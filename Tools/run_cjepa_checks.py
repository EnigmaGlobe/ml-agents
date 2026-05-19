import sys
import traceback

print('Using python:', sys.executable)
OUT_PATH = 'tools/run_cjepa_checks.out'
out_lines = []
def o(s):
    print(s)
    out_lines.append(str(s))

# Check torch
try:
    import torch
    o('torch: ' + getattr(torch, '__version__', repr(torch)))
except Exception as e:
    print('torch import failed:', repr(e))

# Run the evaluate test directly
try:
    from mlagents.trainers.torch_entities.components.reward_providers.test_c_jepa_reward_provider import (
        test_c_jepa_evaluate_preextracted_slots,
        test_c_jepa_update_stub_noop,
    )
    o('\nRunning evaluate test...')
    try:
        test_c_jepa_evaluate_preextracted_slots()
        o('evaluate test: PASS')
    except Exception:
        o('evaluate test: FAIL')
        out_lines.append(traceback.format_exc())

    o('\nRunning update test...')
    try:
        test_c_jepa_update_stub_noop()
        o('update test: PASS')
    except Exception:
        o('update test: FAIL')
        out_lines.append(traceback.format_exc())
except Exception:
    o('Could not import tests:')
    out_lines.append(traceback.format_exc())

# Run smoke script
o('\nRunning smoke script...')
try:
    import runpy
    runpy.run_path('tools/smoke_cjepa.py', run_name='__main__')
    o('smoke: DONE')
except Exception:
    o('smoke: FAILED')
    out_lines.append(traceback.format_exc())

# write output file
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out_lines))
