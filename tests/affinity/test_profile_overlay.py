import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('profile_overlay', Path(__file__).parents[2]/'src/kfold/affinity/profile_overlay.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
merge = module.merge_profile_overlay


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(protein_key='key', target_input_key='target', target_training_ligands=5,
            candidate_rank=i, input_id=str(i), system_index=i+1, input_sha='seq', apo_sha='apo', ligand_sha=str(i))
            for i in range(4)]
        self.profile = {k:self.rows[2][k] for k in ('input_id','system_index','target_input_key','input_sha','apo_sha','ligand_sha')}
        self.profile['record_checksums_verified'] = True

    def test_reuse_is_immutable_and_missing_earlier_stays_missing(self):
        original = {'0': {'existing': True}}
        before = copy.deepcopy(original)
        merged, digest = merge(self.rows[:2], self.rows, original, {'2':self.profile})
        self.assertEqual(original, before)
        self.assertEqual(set(merged), {'0','2'})
        self.assertEqual(len(digest), 64)

    def test_changed_prefix_and_rank_gaps_fail(self):
        for field, value in [('ligand_sha','changed'),('candidate_rank',9)]:
            rows = copy.deepcopy(self.rows); rows[0][field] = value
            with self.assertRaises(ValueError): merge(self.rows[:2], rows, {}, {})

    def test_mixed_target_fails(self):
        rows = copy.deepcopy(self.rows);rows[3]['protein_key'] = 'other'
        with self.assertRaises(ValueError): merge(self.rows[:2], rows, {}, {})

    def test_identity_and_checksum_claim_required(self):
        for field,value in [('apo_sha','other'),('record_checksums_verified',False)]:
            p = dict(self.profile);p[field] = value
            with self.assertRaises(ValueError): merge(self.rows[:2], self.rows, {}, {'2':p})

    def test_conflict_and_unrelated_profile_rejected(self):
        with self.assertRaises(ValueError): merge(self.rows[:2], self.rows, {'2':{'different':1}}, {'2':self.profile})
        with self.assertRaises(ValueError): merge(self.rows[:2], self.rows, {}, {'other':self.profile})


if __name__ == '__main__':
    unittest.main()
