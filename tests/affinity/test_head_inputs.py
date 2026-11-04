import torch

from kfold.affinity.compact import pack_crop_union
from kfold.affinity.features import pack_full_cross
from kfold.affinity.head_inputs import compact_head_inputs
from kfold.affinity.shards import ShardStore
from kfold.model.modules.affinity_pairformer import AffinityPairformer


def test_serialized_union_matches_direct_dense_head_input_and_forward(tmp_path):
    torch.manual_seed(42)
    n = 14
    p = torch.arange(n) < 12
    s, lm, z = torch.randn(n, 16), torch.randn(n, 16), torch.randn(n, n, 16)
    logits = torch.randn(n, n, 64)
    logits = logits + logits.transpose(0, 1)
    source = pack_full_cross(s, lm, z, logits, protein_mask=p, ligand_mask=~p,
                             bin_boundaries=torch.linspace(2+20/64, 22-20/64, 63),
                             token_mapping={'residue_index': torch.arange(n)})
    source.update(receipt={'seed': 17}, cache_identity_sha256='a'*64)
    indices = torch.tensor([0, 2, 3, 5, 8, 12, 13])
    compact = pack_crop_union(source, dict(query_200=indices, target_consensus_200=indices,
                                          query_350=torch.arange(n)),
                              crop_receipt={'selector_version': 'test', 'config': {'test': True}})
    store = ShardStore(tmp_path, 'head-test')
    store.add('sample', compact); store.seal()
    actual = compact_head_inputs(ShardStore(tmp_path, 'head-test').get('sample'), 'query_200')
    active = (~p[indices])[:, None] | (~p[indices])[None, :]
    expected_z = z.to(torch.bfloat16).float()[indices][:, indices] * active[..., None]
    assert torch.equal(actual['z'][0], expected_z)
    prob = logits.to(torch.bfloat16).float()[indices][:, indices].softmax(-1)
    centers = torch.linspace(2+10/64, 22-10/64, 64)
    expected_dist = torch.stack((prob[..., centers<=8].sum(-1), (prob*centers).sum(-1),
                                -(prob*prob.clamp_min(1e-8).log()).sum(-1)/torch.tensor(64.).log()), -1)
    expected_dist *= active[..., None]
    assert torch.allclose(actual['distogram_features'][0], expected_dist, atol=1e-6, rtol=0)
    assert torch.equal(actual['s_inputs'][0], s.to(torch.bfloat16).float()[indices])
    head = AffinityPairformer(AffinityPairformer.Config(channel_s=16, channel_z=16,
                             num_heads_attn=4, num_blocks=1, dropout=0.)).eval()
    # Nonzero readout checks more than the final zero-initialized bias.
    torch.nn.init.normal_(head.readout[-1].weight, std=.01)
    with torch.no_grad():
        output = head(**actual)
        direct = dict(actual, z=expected_z[None], distogram_features=expected_dist[None])
        expected = head(**direct)
    assert torch.isfinite(output).all()
    assert torch.allclose(output, expected, atol=1e-6, rtol=0)
