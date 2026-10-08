# K-Fold inference

K-Fold accepts YAML or JSON files describing proteins, DNA, RNA, and ligands.
Inference requires a CUDA GPU.
No MSA search or sequence database is required.

- [Installation](#installation)
- [Running predictions](#running-predictions)
- [Selecting stages](#selecting-stages)
- [Multi-GPU inference](#multi-gpu-inference)
- [Command-line options](#command-line-options)
- [Input format](#input-format)
- [Apo structures](#apo-structures)
- [Outputs and confidence](#outputs-and-confidence)
- [Python API](python_api.md)

## Installation

K-Fold requires Python 3.11 or later.

Install from PyPI:

```bash
pip install kfold
```

Or install from source:

```bash
git clone https://github.com/kaist-ai-bio/kfold.git
cd kfold
pip install -e .
```

K-Fold uses custom Triton kernels by default when Triton is installed and CUDA is available.

## Running predictions

Save the [input example](#input-format) as `query.yaml`, then run:

```bash
kfold --input query.yaml --out-dir predictions/
```

By default, K-Fold generates protein apo structures for all selected queries with [AtlasFold](https://github.com/SeonghwanSeo/atlasfold), then predicts five complex structures per query with seed 1.
You can also provide apo structures from experiments or other prediction tools.
Model weights and the chemical component dictionary (CCD) are downloaded automatically.
Use `--cache-dir` to select a download cache.

The default backend, `auto`, prefers Triton, then cuEquivariance, then PyTorch, depending on availability.
Use `--kernel {auto,triton,cuequiv,torch}` to select the backend for K-Fold complex prediction and AtlasFold apo preparation:

```bash
kfold --input query.yaml --out-dir predictions/ --kernel triton
```

The selected backend is used for both stages.
The `cuequiv` backend requires cuEquivariance to be installed separately.

From the repository root, the script entry point provides the same interface:

```bash
python run_kfold.py --input query.yaml --out-dir predictions/
```

### Multiple queries

Pass a directory to process its immediate `.yaml`, `.yml`, and `.json` files.
Each file contains one query, and query names must be unique:

```bash
kfold --input queries/ --out-dir predictions/
```

### Checking and resuming runs

Validate query files and referenced paths before loading models:

```bash
kfold --input queries/ --out-dir predictions/ --dry-run
```

A dry run checks query files, structure paths, and CCD codes, and reports pending and completed jobs without loading models or writing prediction outputs.
The CCD is downloaded if needed; structure contents are checked during inference.

Rerun the same command to continue an interrupted run.
Completed predictions and apo structures are reused; unfinished predictions are rerun.

Use `--overwrite` to replace existing results after changing inputs, sampling settings, or output options.

To add predictions and rank them with existing results, include both old and new seeds: with `--seeds 1 2 3`, a completed seed 1 is skipped while seeds 2 and 3 run.

## Selecting stages

The default `--stage all` prepares apo structures and predicts complexes.
Use `--stage apo` to prepare apos only, or `--stage complex` to predict complexes from prepared apos:

```bash
kfold --stage apo --input query.yaml --out-dir predictions/ --seeds 1
kfold --stage complex --input query.yaml --out-dir predictions/ --seeds 1
```

Use the same input, output directory, and seeds for both commands.
With `--share-apo-seeds`, pass the same shared apo seeds to both stages; the complex stage can use any inference seeds specified by `--seeds`.
`--stage complex` reads the prepared queries and structures from `--out-dir` for the queries selected by `--input`.

## Multi-GPU inference

Use `--gpu-ids` to distribute queries and seeds across the selected GPUs:

```bash
kfold --input queries/ --out-dir predictions/ --seeds 1 2 3 --gpu-ids 0 1
```

Both stages use the selected GPUs.
Each prediction runs entirely on one GPU; additional GPUs process other queries or seeds.
You can also pass a single query file with multiple seeds.
Without `--gpu-ids`, inference uses GPU 0.
Pass a single ID, such as `--gpu-ids 2`, to select one GPU.
IDs refer to visible CUDA devices; with `CUDA_VISIBLE_DEVICES=2,3`, `--gpu-ids 0 1` selects physical GPUs 2 and 3.

## Command-line options

Run `kfold --help` for all options.
The default `--stage all` prepares apos and predicts complexes.

| Option | Default | Description |
| --- | --- | --- |
| `-i`, `--input` | Required | Query JSON/YAML file or directory. |
| `-o`, `--out-dir` | Required | Output directory; prepared inputs for `--stage complex`. |
| `--stage` | `all` | Both stages (`all`), apo preparation (`apo`), or prediction from prepared inputs (`complex`). |
| `--seeds` | `1` | One or more unique positive complex inference seeds. |
| `--apo-config` | Built-in defaults | YAML settings for apo batching and AtlasFold sampling. |
| `--num-apos` | `1` | Generated apos per protein entry per inference seed without shared seeds (1–5). |
| `--share-apo-seeds` | Off | Unique positive AtlasFold seeds for generating an apo ensemble shared across inference seeds. |
| `--num-shared-apos` | `5` | Maximum apo candidates used per protein entry with shared seeds (1–5). |
| `--num-samples` | `5` | Complex predictions per query/seed. |
| `--num-recycles` | `10` | Model recycling iterations. |
| `--num-steps` | `100` | Diffusion steps. |
| `--kernel` | `auto` | Backend for K-Fold and AtlasFold apo preparation: `auto`, `triton`, `cuequiv`, or `torch`; `auto` prefers Triton, then cuEquivariance, then PyTorch. |
| `--gpu-ids` | `0` | Unique non-negative visible CUDA device IDs. |
| `--disable-struct-encoder` | Off | Disable the protein structure encoder to save memory. |
| `--disable-rna-encoder` | Off | Disable the RNA encoder; only for queries without RNA. |
| `--cpu-offload` | Off | Offload encoders to CPU: less GPU memory, more CPU memory and transfer time. |
| `--cache-dir` | Hugging Face default | Download cache for model weights and CCD. |
| `--dry-run` | Off | Validate query files and paths, and report pending jobs. |
| `--overwrite` | Off | Rerun completed query/seed jobs. |
| `--save-confidence` | Off | Save per-atom and token-pair confidence arrays. |
| `--save-embeddings` | Off | Save single and pair embeddings. |
| `--save-distogram` | Off | Compute and save distogram arrays. |
| `--save-trajectory` | Off | Compute and save diffusion trajectories. |

### Sampling

Use `--seeds` for independent runs and `--num-samples` for the number of predictions per seed:

```bash
kfold --input query.yaml --out-dir predictions/ --seeds 1 2 --num-samples 5
```

### Apo configuration

Use `--apo-config apo.yaml` to customize AtlasFold settings.
The default configuration is shown below.

```yaml
max_tokens_per_batch: 1024
monomer:
  num_recycles: 4
  mlm_prob: 0.15
  num_samples: 5
  num_steps: null # AtlasFold's dynamic scheduling
multimer:
  num_recycles: 4
  mlm_prob: 0.15
  num_samples: 5
  num_steps: 100
```

`max_tokens_per_batch` controls the trade-off between throughput and memory use ([see AtlasFold](https://github.com/SeonghwanSeo/atlasfold/tree/main#performance-and-gpu-memory)).

## Input format

Every query requires a `name` and a non-empty `sequences` list.
Each entry contains one component type: `protein`, `protein_pair`, `dna`, `rna`, or `ligand`.
YAML and JSON use the same schema.

```yaml
name: test
sequences:
  - protein:
      id: [A, B]
      sequence: MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANL
```

### Proteins

| Field | Required | Description |
| --- | --- | --- |
| `id` | Yes | Chain ID or list of IDs for identical copies. |
| `sequence` | Yes | Amino-acid sequence; `X` represents an unknown residue. |
| `modifications` | No | List of [residue modifications](#residue-modifications). |
| `apo` | No | Optional apo structure paths to use instead of automatic generation. |
| `prior` | No | List of separate prior structure paths; requires `apo`. |

The copies in one entry use the same set of apo and prior structures.
To provide different structures for chains with the same sequence, put them in separate entries.

### Protein pairs

Use `protein_pair` for two proteins whose relative positions are preserved in the starting structure, such as the heavy and light chains of a Fab.
Pair structures are generated automatically with AtlasFold-Multimer, or you can provide your own.

| Field | Required | Description |
| --- | --- | --- |
| `id1`, `id2` | Yes | Chain ID or list of IDs for each component, corresponding to `sequence1` and `sequence2`. |
| `sequence1`, `sequence2` | Yes | Amino-acid sequences in component order. |
| `modifications1`, `modifications2` | No | Residue modifications for each component. |
| `apo` | No | Optional two-chain structure paths to use instead of automatic generation. |
| `prior` | No | List of separate two-chain prior paths; requires `apo`. |

Use `id1: H` and `id2: L` for one pair.
For multiple copies, the two ID lists must have the same length and are paired by position.
For `id1: [H, M]` and `id2: [L, N]`, chains H and M use `sequence1`, and chains L and N use `sequence2`, forming pairs H/L and M/N.
The two copies are placed independently.
Provided structures must contain exactly two non-empty protein chains in `sequence1`, `sequence2` order.

### DNA and RNA

DNA and RNA entries require `id` and `sequence`, and accept optional `modifications`.
DNA uses `A`, `C`, `G`, `T`, and `N`; RNA uses `A`, `C`, `G`, `U`, and `N`.

```yaml
name: nucleic_acids
sequences:
  - dna:
      id: A
      sequence: ACGTACGT
  - rna:
      id: B
      sequence: ACGUACGU
```

### Ligands

Ligands require `id` and exactly one of `ccd` or `smiles`.
Use a CCD code for a known chemical component, including ions, or a SMILES string for a custom ligand.
The following entries can be added to a query's `sequences` list:

```yaml
- ligand:
    id: C
    ccd: ATP
- ligand:
    id: [D, E]
    ccd: MG
- ligand:
    id: F
    smiles: 'CC(=O)Oc1ccccc1C(=O)O'
```

`ccd` also accepts a non-empty list such as `[ATP]`, or multiple CCD codes for a multi-residue component.
CCD codes must exist in the loaded dictionary.

### Residue modifications

Add `modifications` to a protein, DNA, or RNA entry.
Each modification specifies a 1-based `index` within the sequence and the replacement `ccd` code.
For example, this replaces the fourth residue with phosphoserine:

```yaml
name: modified_protein
sequences:
  - protein:
      id: A
      sequence: MKTSA
      modifications:
        - index: 4
          ccd: SEP
```

Each residue index may appear only once.
Protein pairs use the same format in `modifications1` and `modifications2` for their respective sequences.

### Covalent bonds

The optional top-level `bonds` list connects pairs of atoms.
Each atom reference is `[chain_id, residue_index, atom_name]`, with 1-based residue indices:

```yaml
bonds:
  - [[A, 20, NZ], [C, 1, C08]]
```

This connects atom `NZ` of chain A residue 20 to atom `C08` of chain C residue 1.
The chains, residues, and atom names must exist in the query's prepared structure.
Self-bonds and duplicate bonds are rejected.

## Apo structures

K-Fold uses apo structures as starting structures for proteins.
These are generated automatically, or you can provide your own.

### Automatic generation

K-Fold generates apo structures with AtlasFold for individual proteins and AtlasFold-Multimer for protein pairs.

By default, apo structures are generated separately for each complex inference seed.
Use `--num-apos` to choose how many apo structures to generate for each protein entry per inference seed (1–5; default: 1).
For example, with `--seeds 42 --num-apos 3`, AtlasFold uses seeds `421`, `422`, and `423`, generating five diffusion samples per apo seed by default.

Generating apos separately can be costly when running many inference seeds.
Use `--share-apo-seeds` to generate an apo ensemble for each protein entry and reuse it across all complex inference seeds, with `--num-shared-apos` controlling how many candidates each entry uses:

```bash
kfold --input query.yaml --out-dir predictions/ --seeds 1 2 3 4 5 6 7 8 9 10 --share-apo-seeds 7 11 42 --num-shared-apos 2
```

This generates three apo candidates per entry using AtlasFold seeds `7`, `11`, and `42`, then randomly selects two candidates from each entry's ensemble for each complex inference seed.
Selection is reproducible for a given inference seed, and entries with fewer candidates use all available candidates.
By default, up to five candidates are used per entry.
Selecting a subset leaves the prior ensemble unchanged.

### Providing apo structures

Use `apo` to provide experimental structures or predictions from tools such as AlphaFold2:

```yaml
name: provided_apo
sequences:
  - protein:
      id: A
      sequence: MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANL
      apo:
        - structures/protein.pdb
```

List one or more PDB or mmCIF files under `apo`.
Relative paths are resolved from the query file's directory.
If a file contains multiple models, each model is included as an apo candidate.
Provided structures replace automatic generation for that entry.
With `--share-apo-seeds`, `--num-shared-apos` also limits the provided apo candidates used per entry during complex prediction.
Without shared seeds, all provided candidates are used.

## Outputs and confidence

For a query named `test`, the default command writes:

```text
predictions/test/
├── test_model.cif
├── test_confidence.json
├── test_summary.csv
├── kfold_settings.json
└── test_seed-1/
    ├── done.txt
    ├── query.json
    ├── kfold_settings.json
    ├── apo_setting.json
    ├── apo/
    │   ├── monomer-1.done
    │   ├── monomer-1_apo.pdb
    │   └── monomer-1_prior.pdb
    ├── test_seed-1_sample-0_model.cif
    ├── test_seed-1_sample-0_confidence.json
    ├── ...
    ├── test_seed-1_sample-4_model.cif
    └── test_seed-1_sample-4_confidence.json
```

By default, each seed directory contains its predictions and prepared inputs.
With `--share-apo-seeds`, `query.json`, `apo_setting.json`, and `apo/` are saved directly under `predictions/test/`; predictions remain in their respective seed directories.
Sample indices start at 0.
`query.json` references the saved apo and prior PDB files using relative paths; move the whole seed directory, or the whole target directory when sharing apos, to keep these references valid.
Apo files use `monomer-{i}_apo.pdb` and `monomer-{i}_prior.pdb`, or `multimer-{i}_apo.pdb` and `multimer-{i}_prior.pdb` for protein pairs.
The number i is the 1-based position in the full `sequences` list, including non-protein entries.

### Ranking and confidence summaries

After all jobs finish, the CLI copies the highest-ranked structure and its confidence JSON to the query directory.
The summary CSV lists `seed`, `sample`, `ranking_score`, `plddt`, `ptm`, `iptm`, `pde`, and `has_clash`, sorted by descending ranking score.

Confidence JSON files group scores under `complex`, `chains`, and `interfaces`.
Chain summaries contain mean pLDDT, mean PDE, and pTM; interface summaries contain pairwise ipTM.
A token represents one standard protein, DNA, or RNA residue, or one atom in a ligand or modified residue.

| Metric | Meaning | Scale |
| --- | --- | --- |
| `plddt` | Local confidence, predicted per atom and averaged in summaries. | 0–100; higher is better. |
| `ptm` | Predicted TM-score for overall structure accuracy. | 0–1; higher is better. |
| `iptm` | Interface predicted TM-score for relative chain placement. | 0–1; higher is better. |
| `pae` | Predicted aligned error between tokens, available in raw confidence arrays. | Å; lower is better. |
| `pde` | Predicted distance error between tokens, averaged in summaries. | Å; lower is better. |
| `has_clash` | Whether the predicted complex triggers the inter-chain clash check. | 0 or 1. |

The complex ranking score is `0.8 * iptm + 0.2 * ptm - 100 * has_clash`.
Structure mmCIF files store per-atom pLDDT in the B-factor field.

Ranking covers the requested seeds and sample count, including reused results.

### Optional outputs

Structure mmCIF and confidence-summary JSON files are always saved.
Enable additional outputs with these flags:

| Flag | File suffix | Contents |
| --- | --- | --- |
| `--save-confidence` | `_sample-{i}_confidence.npz` | Per-atom `plddt` and token-pair `pae` and `pde` arrays. |
| `--save-embeddings` | `_embeddings.npz` | Unpadded `s_inputs`, `s_lm`, and `z` representations, shared across samples. |
| `--save-distogram` | `_distogram.npz` | Shared `logits`, `bin_edges`, `asym_ids`, and `res_ids`. |
| `--save-trajectory` | `_sample-{i}_trajectory.cif` | Diffusion trajectory for each sample. |

All filenames begin with `<name>_seed-<seed>`.
If the best prediction has a saved confidence NPZ, it is also copied to `<name>_confidence.npz` in the query directory.
