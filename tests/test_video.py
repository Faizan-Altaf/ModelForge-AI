import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modelforge.video import calculate_frame_stride


def test_frame_stride_one_second():

    assert calculate_frame_stride(30, 1.0) == 30


def test_frame_stride_half_second():

    assert calculate_frame_stride(30, 0.5) == 15


def test_frame_stride_minimum_one():

    assert calculate_frame_stride(1, 0.1) == 1


def test_invalid_fps():

    with pytest.raises(ValueError):
        calculate_frame_stride(0, 1)


def test_invalid_interval():

    with pytest.raises(ValueError):
        calculate_frame_stride(30, 0)
