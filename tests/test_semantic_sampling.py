import pytest
from systemone_lab.training import domain_balanced_weights


def test_semantic_balance_preserves_domain_mass_and_balances_decisions():
    records = [{'meta': {'domain':'infogather'}, 'labels': {'answer':f'repair the part{i}'}}
               for i in range(10)]
    records += [{'meta': {'domain':'infogather'}, 'labels': {'answer':'run the diagnostic'}}] * 20
    records += [{'meta': {'domain':'spatial'}, 'labels': {'answer':'north'}}] * 3
    legacy = domain_balanced_weights(records, 'infogather=0.4,spatial=0.6')
    fixed = domain_balanced_weights(records, 'infogather=0.4,spatial=0.6', True)
    assert sum(legacy[10:30]) == pytest.approx(.4/11)
    assert sum(fixed[:10]) == pytest.approx(.2)
    assert sum(fixed[10:30]) == pytest.approx(.2)
    assert sum(fixed[30:]) == pytest.approx(.6)
    assert sum(fixed) == pytest.approx(1.)


def test_unknown_semantic_label_is_rejected():
    with pytest.raises(ValueError, match='unknown infogather'):
        domain_balanced_weights([{'meta':{'domain':'infogather'}, 'labels':{'answer':'bad'}}],
                                'infogather=1', True)
