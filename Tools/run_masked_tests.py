import subprocess, sys
p = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'mlagents/trainers/torch_entities/components/reward_providers/test_c_jepa_masked_update.py'], capture_output=True, text=True)
with open('tools/run_masked_tests.out', 'w', encoding='utf-8') as f:
    f.write(p.stdout)
    f.write('\n')
    f.write(p.stderr)
print('wrote tools/run_masked_tests.out')
