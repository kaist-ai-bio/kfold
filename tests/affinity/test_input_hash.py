from types import SimpleNamespace
import torch
from kfold.affinity.features import hash_prepared_input

def fake(tensor):
    group = SimpleNamespace(to_dict=lambda: {'x': tensor})
    return SimpleNamespace(**{k: group for k in ('chain', 'token', 'atom', 'bond', 'sequence')})

def test_layout_independent_empty_expanded_and_bf16_hashes():
    for dtype in (torch.int64, torch.float32, torch.bfloat16):
        for size in (0, 1, 3):
            expanded = torch.ones(1, dtype=dtype).expand(size)
            assert hash_prepared_input(fake(expanded)) == hash_prepared_input(fake(torch.ones(size, dtype=dtype)))
    assert hash_prepared_input(fake(torch.ones(1))) != hash_prepared_input(fake(torch.zeros(1)))
    assert hash_prepared_input(fake(torch.empty(0))) != hash_prepared_input(fake(torch.empty(0, 2)))
