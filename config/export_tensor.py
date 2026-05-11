from __future__ import annotations

import argparse
import csv
import struct
from pathlib import Path


DEFAULT_TAG = "Environment/Cumulative Reward"
DEFAULT_TAGS = [
    "Environment/Cumulative Reward",
    "Environment/Episode Length",
    "Policy/Extrinsic Reward",
    "Policy/Learning Rate",
    "Losses/Policy Loss",
    "Losses/Value Loss",
]


def read_varint(buf: bytes, index: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        b = buf[index]
        index += 1
        value |= (b & 0x7F) << shift
        if not (b & 0x80):
            return value, index
        shift += 7


def parse_fields(buf: bytes) -> list[tuple[int, int, object]]:
    index = 0
    fields: list[tuple[int, int, object]] = []
    while index < len(buf):
        key, index = read_varint(buf, index)
        field_no = key >> 3
        wire_type = key & 7
        if wire_type == 0:
            value, index = read_varint(buf, index)
        elif wire_type == 1:
            value = struct.unpack_from("<Q", buf, index)[0]
            index += 8
        elif wire_type == 2:
            size, index = read_varint(buf, index)
            value = buf[index : index + size]
            index += size
        elif wire_type == 5:
            value = struct.unpack_from("<I", buf, index)[0]
            index += 4
        else:
            raise ValueError(f"Unsupported wire type {wire_type}")
        fields.append((field_no, wire_type, value))
    return fields


def iter_tfrecords(path: Path):
    with path.open("rb") as f:
        while True:
            length_bytes = f.read(8)
            if not length_bytes:
                return
            if len(length_bytes) != 8:
                raise EOFError(f"Truncated TFRecord length in {path}")
            (length,) = struct.unpack("<Q", length_bytes)
            f.read(4)  # masked crc for the length
            data = f.read(length)
            if len(data) != length:
                raise EOFError(f"Truncated TFRecord payload in {path}")
            f.read(4)  # masked crc for the payload
            yield data


def parse_event(buf: bytes) -> tuple[float | None, int | None, bytes | None]:
    wall_time = None
    step = None
    summary = None
    for field_no, wire_type, value in parse_fields(buf):
        if field_no == 1 and wire_type == 1:
            wall_time = struct.unpack("<d", struct.pack("<Q", value))[0]
        elif field_no == 2 and wire_type == 0:
            step = int(value)
        elif field_no == 5 and wire_type == 2:
            summary = bytes(value)
    return wall_time, step, summary


def parse_summary_values(summary_buf: bytes):
    for field_no, wire_type, value in parse_fields(summary_buf):
        if field_no == 1 and wire_type == 2:
            yield bytes(value)


def parse_scalar_value(value_buf: bytes) -> tuple[str | None, float | None]:
    tag = None
    scalar = None
    for field_no, wire_type, value in parse_fields(value_buf):
        if field_no == 1 and wire_type == 2:
            tag = bytes(value).decode("utf-8", errors="replace")
        elif field_no == 2 and wire_type == 5:
            scalar = struct.unpack("<f", struct.pack("<I", int(value)))[0]
    return tag, scalar


def extract_series(event_path: Path, target_tag: str):
    rows = []
    for record in iter_tfrecords(event_path):
        wall_time, step, summary_buf = parse_event(record)
        if summary_buf is None or wall_time is None or step is None:
            continue
        for value_buf in parse_summary_values(summary_buf):
            tag, scalar = parse_scalar_value(value_buf)
            if tag == target_tag and scalar is not None:
                rows.append((wall_time, step, scalar))
    return rows


def sanitize_filename(tag: str) -> str:
    if tag == "Environment/Cumulative Reward":
        return "Cumulative Reward.csv"
    if tag == "Environment/Episode Length":
        return "Episode Length.csv"
    if tag == "Policy/Extrinsic Reward":
        return "Extrinsic Reward.csv"
    if tag == "Policy/Learning Rate":
        return "Learning Rate.csv"
    if tag == "Losses/Policy Loss":
        return "Policy Loss.csv"
    if tag == "Losses/Value Loss":
        return "Value Loss.csv"
    return tag.replace("/", "_") + ".csv"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a scalar series from a TensorBoard event file.")
    parser.add_argument("--event-file", required=True, type=Path, help="Path to the raw .tfevents file")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory to write CSV files")
    parser.add_argument("--tag", default=DEFAULT_TAG, help="TensorBoard scalar tag to export")
    parser.add_argument("--all", action="store_true", help="Export all known scalar series")
    args = parser.parse_args()

    tags = DEFAULT_TAGS if args.all else [args.tag]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for tag in tags:
        rows = extract_series(args.event_file, tag)
        out_csv = args.out_dir / sanitize_filename(tag)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Wall time", "Step", "Value"])
            for wall_time, step, value in rows:
                writer.writerow([wall_time, step, value])
        print(f"Wrote {len(rows)} rows to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
