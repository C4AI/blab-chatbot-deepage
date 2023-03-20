#!/usr/bin/env python3

"""This script can be used to run the main module."""

from pathlib import Path
from runpy import run_path
from sys import path

MODULE = "blab_chatbot_deepage"

src_dir = Path(__file__).parent.resolve() / "src"
path.append(str(src_dir))
run_path(str(src_dir / MODULE))
