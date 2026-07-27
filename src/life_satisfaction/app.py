"""Compatibility wrapper for the optional research demonstrator."""

from .deployment import FEATURE_SPECS as INPUTS
from .deployment import build_gradio_app as create_demo
from .deployment import main

__all__ = ["INPUTS", "create_demo", "main"]


if __name__ == "__main__":
    main()
