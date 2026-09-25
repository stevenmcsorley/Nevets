import random
from systemone_lab.data.spatial_worlds import make_counterfactual_pair

def test_counterfactual_changes_answer_and_is_minimalish():
    r=random.Random(3)
    for _ in range(50):
        a,b=make_counterfactual_pair(r,4,1)
        assert a.label != b.label
        assert a.meta['query'] == b.meta['query']
        assert a.meta['pair_id'] == b.meta['pair_id']
