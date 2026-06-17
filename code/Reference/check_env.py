"""
check_env.py - verify the Python environment is set up correctly to run the
MUA detectability sweep (LFPy 2.3.6 + NEURON).

Run it INSIDE the environment you intend to use (e.g. the mua_env venv):

    python code/check_env.py

For each dependency it reports the version, whether it meets LFPy 2.3.6's
minimum, and whether it is loaded from THIS env or inherited from the base
Python (relevant when the venv was created with --system-site-packages). It
warns if NumPy is >= 2 (LFPy 2.3.6 and the project's documented stack expect
NumPy 1.x), and finishes with a tiny LFPy+NEURON simulation as a functional
smoke test. Exit code is 0 if everything is OK, 1 otherwise.
"""
import os
import sys

# LFPy 2.3.6 minimum versions (from its requirements.txt / PyPI metadata).
MINIMUMS = {
    'numpy':      '1.8',
    'scipy':      '0.14',
    'h5py':       '2.5',
    'matplotlib': '2.0',
    'Cython':     '0.20',
    'lfpykit':    '0.5',
    'neuron':     '7.7.2',
    'LFPy':       '2.3.6',
}


def parse(v):
    """Leading numeric components of a version, e.g. '8.0.0-2-g0e9a' -> (8,0,0,2)."""
    nums = []
    for part in str(v).replace('-', '.').split('.'):
        if part.isdigit():
            nums.append(int(part))
        else:
            break  # stop at first non-numeric chunk (rc1, dev, g<hash>, ...)
    return tuple(nums)


def meets(version, minimum):
    try:
        return parse(version) >= parse(minimum)
    except Exception:
        return None


def location(mod):
    """'this env' if the module lives under sys.prefix, else 'BASE (inherited)'."""
    f = getattr(mod, '__file__', '') or ''
    try:
        prefix = os.path.abspath(sys.prefix)
        return 'this env' if os.path.commonpath([os.path.abspath(f), prefix]) == prefix \
               else 'BASE python (inherited)'
    except Exception:
        return '(unknown location)'


print('=' * 72)
print('Python     :', sys.version.split()[0])
print('Executable :', sys.executable)
print('Env prefix :', sys.prefix)
print('In a venv  :', 'YES' if sys.prefix != sys.base_prefix else 'NO (this is the base Python!)')
print('=' * 72)

ok = True
versions = {}
for name, minimum in MINIMUMS.items():
    try:
        mod = __import__(name)
        v = getattr(mod, '__version__', '(unknown)')
        versions[name] = v
        m = meets(v, minimum)
        flag = 'OK' if m else ('??' if m is None else 'TOO OLD')
        if m is False:
            ok = False
        print(f'{name:11} {v:15} (need >= {minimum:8}) [{flag:7}] <- {location(mod)}')
    except Exception as e:
        ok = False
        print(f'{name:11} NOT INSTALLED / import failed: {type(e).__name__}: {e}')

print('-' * 72)

# NumPy 2.x check
nv = versions.get('numpy')
if nv and parse(nv) and parse(nv)[0] >= 2:
    ok = False
    print(f'WARNING: numpy {nv} is 2.x. LFPy 2.3.6 and the project baseline use NumPy 1.x.')
    print('         Pin it with:  python -m pip install --ignore-installed "numpy<2"')
elif nv:
    print(f'numpy {nv} is 1.x - matches the documented stack. Good.')

# Functional smoke test: load the bundled morphology and simulate briefly.
print('-' * 72)
print('Functional test: building a cell and simulating 5 ms ...')
try:
    import LFPy
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    morph = os.path.join(repo, 'LFPy-2.3.6', 'examples', 'morphologies',
                         'L5_Mainen96_LFPy.hoc').replace('\\', '/')
    if not os.path.isfile(morph):
        ok = False
        print(f'  morphology not found at {morph} - is the repo fully cloned?')
    else:
        cell = LFPy.Cell(morphology=morph, passive=True, dt=2**-4, tstart=0, tstop=5)
        cell.simulate()
        print(f'  OK: LFPy + NEURON ran. Simulated {cell.tvec.size} time points.')
except Exception as e:
    ok = False
    print(f'  FAILED: {type(e).__name__}: {e}')

print('=' * 72)
print('RESULT:', 'ALL GOOD - environment looks ready.' if ok
      else 'PROBLEMS found above - see the flagged lines.')
print('=' * 72)
sys.exit(0 if ok else 1)
