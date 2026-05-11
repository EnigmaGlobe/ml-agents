#!/usr/bin/env python3
"""
Create an MP4 from a run's screenshot frames.

Usage examples (PowerShell):
  # Use a specific run folder
  python .\tools\make_video_from_frames.py "Project\Assets\ML-Agents\Examples\PushBlock\metadata\recordings\20260511_143036" --fps 30

  # Auto-detect latest run under the default recordings folder
  python .\tools\make_video_from_frames.py --auto --fps 30

The script attempts, in order:
  1) Call an external `ffmpeg` binary if found in PATH.
  2) Use OpenCV (`cv2`) if installed.
  3) Use `imageio` as a final fallback.

It writes an MP4 named `recording.mp4` (or custom via --output) into the provided run folder.
"""

import argparse
import sys
import os
from pathlib import Path
import glob
import re
import subprocess
import tempfile


def find_default_recordings_dir():
    # Path relative to repo root expected by the example code
    # (works if script is run from repo root)
    candidate = Path("Project") / "Assets" / "ML-Agents" / "Examples" / "PushBlock" / "metadata" / "recordings"
    return candidate


def find_latest_run_folder(recordings_root: Path):
    if not recordings_root.exists():
        return None
    dirs = [d for d in recordings_root.iterdir() if d.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime)


def collect_pngs(run_dir: Path):
    # Recursively collect PNGs and sort by the numeric part of the filename if present.
    files = list(run_dir.rglob("*.png"))
    if not files:
        return []

    def sort_key(p: Path):
        m = re.search(r"(\d+)(?=\.png$)", p.name)
        if m:
            return int(m.group(1))
        return p.name

    files.sort(key=sort_key)
    return files


def try_ffmpeg(png_paths, output_path: Path, fps: int):
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        return False, "ffmpeg not found in PATH"

    # If files are in a single folder and follow the pattern frame_XXXX.png, use -framerate and -start_number
    first_parent = Path(png_paths[0]).parent
    same_dir = all(Path(p).parent == first_parent for p in png_paths)
    pattern_match = None
    padding = None
    if same_dir:
        m = re.search(r"frame_(\d+)\.png$", png_paths[0].name)
        if m:
            padding = len(m.group(1))
            pattern_match = True

    if same_dir and pattern_match:
        # Use numeric pattern input with start_number set to the smallest index
        nums = []
        for p in png_paths:
            mm = re.search(r"frame_(\d+)\.png$", Path(p).name)
            if mm:
                nums.append(int(mm.group(1)))
        if nums:
            start = min(nums)
            pattern = str(first_parent / (f"frame_%0{padding}d.png"))
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-start_number",
                str(start),
                "-i",
                pattern,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ]
            try:
                subprocess.run(cmd, check=True)
                return True, None
            except subprocess.CalledProcessError as e:
                return False, f"ffmpeg failed: {e}"

    # Fallback: create a temporary list file for ffmpeg concat input (robust to non-consecutive numbering)
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as lf:
        list_path = Path(lf.name)
        for p in png_paths:
            # ffmpeg concat demuxer expects paths like: file '/absolute/path.png'
            lf.write("file '{}'\n".format(str(p).replace("'", "'\\''")))

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True)
        try:
            list_path.unlink()
        except Exception:
            pass
        return True, None
    except subprocess.CalledProcessError as e:
        return False, f"ffmpeg failed: {e}"


def try_opencv(png_paths, output_path: Path, fps: int):
    try:
        import cv2
    except Exception as e:
        return False, f"opencv (cv2) not available: {e}"

    if not png_paths:
        return False, "no pngs"

    first = cv2.imread(str(png_paths[0]))
    if first is None:
        return False, "cv2 failed to read first image"
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, float(fps), (w, h))
    if not out.isOpened():
        return False, "cv2 VideoWriter failed to open"

    for p in png_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"Warning: cv2 failed to read {p}; skipping", file=sys.stderr)
            continue
        out.write(img)
    out.release()
    return True, None


def try_imageio(png_paths, output_path: Path, fps: int):
    try:
        import imageio
    except Exception as e:
        return False, f"imageio not available: {e}"

    if not png_paths:
        return False, "no pngs"

    try:
        writer = imageio.get_writer(str(output_path), fps=fps)
        for p in png_paths:
            img = imageio.imread(str(p))
            writer.append_data(img)
        writer.close()
        return True, None
    except Exception as e:
        return False, f"imageio write failed: {e}"


def main():
    p = argparse.ArgumentParser(description="Make an MP4 from screenshot frames in a run folder")
    p.add_argument('run_folder', nargs='?', default=None, help='Path to a run folder (recordings/{runstamp}). If omitted, use --auto to pick latest under default recordings location.')
    p.add_argument('--auto', action='store_true', help='Auto-detect the latest run under the default recordings folder')
    p.add_argument('--fps', type=int, default=30, help='Output video frames per second (default: 30)')
    p.add_argument('--output', default='recording.mp4', help='Output MP4 filename (placed inside the run folder)')
    args = p.parse_args()

    run_folder = None
    if args.auto:
        rec_root = find_default_recordings_dir()
        if rec_root is None or not rec_root.exists():
            print(f"Default recordings folder not found: {rec_root}")
            sys.exit(2)
        latest = find_latest_run_folder(rec_root)
        if latest is None:
            print(f"No run folders found under: {rec_root}")
            sys.exit(2)
        run_folder = latest
    elif args.run_folder:
        run_folder = Path(args.run_folder)
    else:
        print("Either provide a run_folder path or use --auto to pick the latest run.")
        p.print_help()
        sys.exit(1)

    run_folder = Path(run_folder)
    if not run_folder.exists() or not run_folder.is_dir():
        print(f"Run folder does not exist: {run_folder}")
        sys.exit(2)

    pngs = collect_pngs(run_folder)
    if not pngs:
        print(f"No PNG files found under run folder: {run_folder}")
        sys.exit(2)

    output_path = run_folder / args.output
    print(f"Found {len(pngs)} frame PNGs. Will write: {output_path} (fps={args.fps})")

    # Try ffmpeg first
    ok, msg = try_ffmpeg(pngs, output_path, args.fps)
    if ok:
        print(f"Wrote video with ffmpeg: {output_path}")
        return
    else:
        print(f"ffmpeg approach failed/unused: {msg}")

    # Then try OpenCV
    ok, msg = try_opencv(pngs, output_path, args.fps)
    if ok:
        print(f"Wrote video with OpenCV: {output_path}")
        return
    else:
        print(f"OpenCV approach failed: {msg}")

    # Then try imageio
    ok, msg = try_imageio(pngs, output_path, args.fps)
    if ok:
        print(f"Wrote video with imageio: {output_path}")
        return
    else:
        print(f"imageio approach failed: {msg}")

    print("All available methods failed. Please install ffmpeg or opencv or imageio and retry.")
    sys.exit(2)


if __name__ == '__main__':
    main()
