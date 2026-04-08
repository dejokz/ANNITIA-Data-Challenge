"""
Fast local pre-screen before committing to a full experiment.
Checks syntax and tries a quick import of train.py.
"""

import subprocess
import sys
import py_compile

def simulate():
    """Return True if train.py passes basic sanity checks."""
    try:
        py_compile.compile('train.py', doraise=True)
    except Exception as e:
        print(f"SYNTAX ERROR: {e}")
        return False

    # Try importing train.py in a subprocess to catch runtime import errors
    result = subprocess.run(
        [sys.executable, "-c", "import train; print('OK')"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        print(f"IMPORT ERROR: {result.stderr}")
        return False
    return True


if __name__ == '__main__':
    ok = simulate()
    sys.exit(0 if ok else 1)
