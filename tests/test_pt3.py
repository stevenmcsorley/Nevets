import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('pt3',Path(__file__).parents[1]/'scripts/pt/pt3.py')
pt3=importlib.util.module_from_spec(spec); spec.loader.exec_module(pt3)


def row(a,b,c): return {'metrics':dict(in_format=a,table=b,cf_both=c)}


def test_plateau_requires_two_readouts_and_accepts_gain_on_any_metric():
    assert not pt3.plateau([row(.5,.5,.5),row(.5,.5,.5)])
    assert pt3.plateau([row(.5,.5,.5),row(.5,.5,.5),row(.5,.5,.5)])
    assert not pt3.plateau([row(.5,.5,.5),row(.5,.5,.5),row(.5,.5,.51)])


def test_plateau_compares_with_best_and_does_not_reset_on_noise():
    assert pt3.plateau([row(.5,.5,.5),row(.48,.49,.49),row(.501,.501,.501)])
    assert not pt3.plateau([row(.5,.5,.5),row(.51,.5,.5),row(.51,.5,.5)])
