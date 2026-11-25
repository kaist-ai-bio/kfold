# Predict protein–ligand affinity

Install this K-Fold checkout and use a CUDA GPU. Obtain the Ratio745 joint affinity checkpoint separately. The 1,000-step training checkpoint `step-00001000.pt` has SHA-256 `e323ab653071c264c4dcd71009a99a5d4141dad3973877de7f8896048260a910`.

Write a [K-Fold query](inference.md#input-format) with one protein chain and one noncovalent small molecule. Use the same protein sequence in the query and the supplied apo structure. A query can specify a ligand by SMILES or by a single CCD code.

```yaml
name: my_complex
sequences:
  - protein:
      id: A
      sequence: MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ
      apo: [apo.pdb]
  - ligand:
      id: B
      smiles: CC(=O)NC1=CC=C(O)C=C1
```

Replace the example sequence, apo path, and SMILES with your inputs. Run the apo stage once to prepare the input directory. With an existing apo preparation, use the second command with the same query, output directory, and seed.

```bash
kfold --stage apo --input query.yaml --out-dir predictions --seeds 1
kfold --stage affinity --input query.yaml --out-dir predictions --seeds 1 \
  --affinity-checkpoint ./step-00001000.pt
```

The affinity stage writes `predictions/my_complex/my_complex_seed-1/affinity.json` per query and seed. It uses the `query_200` crop and defaults to three recycles. Use `--overwrite` to replace saved predictions after changing the model or input.

| Output | Interpretation |
|---|---|
| `p_activity` | Regression prediction on the negative log10 molar activity scale. A higher value predicts a lower activity concentration. |
| `binary_logit` | Joint head's binary binding logit. |
| `binding_probability` | Sigmoid of the logit for ranking or thresholding. The value reflects the classifier's training label mixture. |

The existing Ratio745 benchmark used features from a frozen 71K EMA backbone. The v1.1 feature pathway uses the selected v1.1 backbone before apo representation mixing and needs its own benchmark evaluation.
