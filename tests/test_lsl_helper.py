import sys, os

# Ensure src is importable
ROOT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if ROOT_SRC not in sys.path:
    sys.path.insert(0, ROOT_SRC)

import pytest

from app.eeg.streamlit_app import _lsl_connect_or_error


def test_lsl_helper_missing_pylsl(monkeypatch):
    # Simulate pylsl missing by ensuring import raises
    monkeypatch.setitem(sys.modules, 'pylsl', None)

    inlet, err = _lsl_connect_or_error('EEG', timeout=0.1)
    assert inlet is None
    assert isinstance(err, str)
    assert 'pylsl' in err or 'Missing' in err
