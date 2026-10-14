"""Shared test settings; trained-model checks must be requested explicitly."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import torch

torch.set_num_threads(4)


def pytest_addoption(parser):
    parser.addoption("--run-trained", action="store_true", help="Verify the local trained checkpoint")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-trained"):
        for item in items:
            if "trained" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Use --run-trained; requires local checkpoint"))
