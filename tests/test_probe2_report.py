import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('probe2_report', Path(__file__).parents[1] / 'scripts/pt/probe2_report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_positive_control_requires_effect_and_noise_clearance():
    assert report.control_result([.49, .50, .48], [.40, .41, .39])['passed']
    assert not report.control_result([.42] * 3, [.40] * 3)['passed']
    assert not report.control_result([.3, .6, .9], [.4, .4, .4])['passed']
    assert not report.control_result([.39] * 3, [.40] * 3)['passed']


def test_missing_seed_fails_closed():
    with pytest.raises(ValueError):
        report.control_result([.5, .5], [.4] * 3)
