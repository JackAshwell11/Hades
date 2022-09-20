"""Tests all functions in extensions/extensions.py."""
from __future__ import annotations

# Pip
import pytest

__all__ = ()


@pytest.mark.xfail
def test_extensions_script() -> None:
    """Test the extensions.py script."""
