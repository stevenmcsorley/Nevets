import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('repair',Path(__file__).parents[1]/'scripts/general_repair.py')
repair=importlib.util.module_from_spec(spec); spec.loader.exec_module(repair)


def test_queue_requires_completion_and_clear_shared_markers(tmp_path):
    assert not repair.ready(tmp_path)
    marker=tmp_path/'reports/pt/pt3/COMPLETE'; marker.parent.mkdir(parents=True); marker.touch()
    assert repair.ready(tmp_path)
    for name in ['reports/pt/PT2_RUNNING','reports/general/GPU_BUSY','reports/p1/ARM_RUNNING']:
        p=tmp_path/name; p.parent.mkdir(parents=True,exist_ok=True); p.touch()
        assert not repair.ready(tmp_path)
        p.unlink()


def test_sampler_comparison_changes_only_flag():
    arm=dict(init='s30',semantic=False,seed=7)
    base=repair.train_args(arm,Path('output.pt'))
    fixed=repair.train_args({**arm,'semantic':True},Path('output.pt'))
    assert fixed==base+['--semantic-infogather']
    assert base[base.index('--steps')+1]=='10000'
    assert '--resume' in base


def test_queue_may_start_while_pt3_paused_but_not_while_its_gpu_marker_is_held(tmp_path):
    marker=tmp_path/'reports/pt/pt3/PAUSED_FOR_GENERAL'; marker.parent.mkdir(parents=True); marker.touch()
    assert repair.ready(tmp_path)
    busy=tmp_path/'reports/pt/PT2_RUNNING'; busy.touch()
    assert not repair.ready(tmp_path)
