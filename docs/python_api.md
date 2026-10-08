# K-Fold Python API

Inference requires a CUDA GPU.
See the [inference guide](inference.md) for installation requirements, input formats, and output descriptions.

## Running predictions

Save the [input example](inference.md#input-format) as `query.yaml`, then load a model and reuse the runner across queries:

```python
from kfold.inference.query import Query
from kfold.inference.runner import KFoldRunner
from kfold.model import KFold

model = KFold.from_pretrained(device="cuda")
runner = KFoldRunner(model)
query = Query.load("query.yaml")

result = runner.predict(query, seed=1)
result.save("predictions/test/test_seed-1/")
```

`predict()` prepares apo structures and predicts complexes, returning a `FoldingResult` with CPU NumPy arrays.
The input query is unchanged, and the runner can be reused across queries and seeds.

## Constructing queries

Use `Query.from_dict(data, base_dir=...)` for the same schema as YAML/JSON files.
`base_dir` resolves relative structure paths; `Query.load(path)` uses the query file's parent directory.

Typed objects are also available:

```python
from kfold.inference.query import LigandSequence, ProteinSequence, Query

query = Query(
    name="test",
    sequences=[
        ProteinSequence(id=["A"], sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANL"),
        LigandSequence(id=["B"], ccd=["ATP"]),
    ],
)
```

Direct constructors require lists for `id`, ligand `ccd`, and `ProteinPair.id1` and `ProteinPair.id2`, such as `id1=["H"]` and `id2=["L"]`.
For multiple protein-pair copies, `id1` and `id2` must have the same length and are paired by position.
Provided `apo` and `prior` values must be lists of absolute `pathlib.Path` objects pointing to existing files.
Modifications use `Modification(index=4, ccd="SEP")`; bonds use `Bond(atom1=("A", 20, "NZ"), atom2=("C", 1, "C08"))`.

## Options

### Model loading

`KFold.from_pretrained()` returns a frozen model in evaluation mode.

| Argument | Default | Description |
| --- | --- | --- |
| `pretrained_model_name_or_path` | `"kaist-ai-bio/kfold"` | Hugging Face repo or local directory with `config.yaml` and `weights/kfold.pth`. |
| `device` | `"cuda"` | Model device, e.g. `"cuda:1"`; inference requires CUDA. |
| `cache_dir` | `None` | Model and encoder weight cache; defaults to the Hugging Face cache. |
| `use_struct_encoder` | `True` | Load the protein structure encoder; disabling saves memory while retaining apo coordinates. |
| `use_rna_encoder` | `True` | Load the RNA encoder; disable only for queries without RNA. |
| `cpu_offload` | `False` | Offload encoders to CPU: less GPU memory, more CPU memory and transfer time. |
| `kernel_backend` | `"auto"` | `"auto"`, `"triton"`, `"cuequiv"`, or `"torch"`; automatic selection prefers Triton, then cuEquivariance, then PyTorch. |

Omit `kernel_backend` or pass `"auto"` for automatic selection; `None` is not accepted.
`KFoldRunner` uses the model's selected backend for AtlasFold apo preparation as well.
Triton requires CUDA and supports inference only.

With `cpu_offload=True`, use the returned model directly with `KFoldRunner`; calling `.cuda()` or `.to("cuda")` afterward moves the offloaded encoders back to GPU.

### Runner initialization

`KFoldRunner()` can be reused across queries and seeds.

| Argument | Default | Description |
| --- | --- | --- |
| `model` | Required | Loaded `KFold` model on CUDA. |
| `ccd` | `None` | Custom `CCD` object; loads the release CCD when omitted. |
| `apo_config` | `None` | `ApoConfig` for apo generation; same defaults as the CLI. |
| `cache_dir` | `None` | Download cache for CCD and AtlasFold assets. |
| `share_atlaslm` | `True` | Reuse K-Fold's AtlasLM; otherwise load one shared by the apo samplers. |
| `verbose` | `True` | Emit INFO logs for initialization and prediction stages. |

Pass `cache_dir` to both `KFold.from_pretrained()` and `KFoldRunner()` to use the same custom cache throughout.
Apo models are loaded on first use and cached on the GPU for subsequent predictions.

### Prediction

`runner.predict()` predicts one query with one seed.
Sampling defaults match the CLI.

| Argument | Default | Description |
| --- | --- | --- |
| `query` | Required | A `Query` object. |
| `seed` | Required | Positive integer seed for input preparation and prediction. |
| `num_apos` | `1` | Generated apos per protein entry (1–5); provided structures are used directly. |
| `num_samples` | `5` | Predictions per seed. |
| `num_recycles` | `10` | Number of model recycling iterations. |
| `num_steps` | `100` | Number of diffusion steps. |
| `return_embeddings` | `False` | Include shared single and pair embeddings in `result.embeddings`. |
| `return_trajectory` | `False` | Include per-sample diffusion trajectories in `result.trajectory`. |
| `return_distogram` | `False` | Include shared distance logits and metadata in `result.distogram`. |

Sampling counts must be positive.
See [apo structures](inference.md#apo-structures) for how provided and generated structures are used.

### Saving results

`result.save(out_dir)` writes structures and confidence files directly into the provided directory, overwriting existing results for the same query and seed.
Raw confidence NPZ files are saved by default; use `save_confidence=False` to omit them.

| Argument | Default | Description |
| --- | --- | --- |
| `out_dir` | Required | Destination directory, created if needed. |
| `save_confidence` | `True` | Save per-sample pLDDT, PAE, and PDE arrays as NPZ. |
| `save_embeddings` | `False` | Save embeddings when returned by `predict()`. |
| `save_distogram` | `False` | Save distogram arrays when returned by `predict()`. |
| `save_trajectory` | `False` | Save per-sample trajectories as mmCIF when returned by `predict()`. |

## Accessing optional arrays

Enable optional outputs when predicting and again when saving:

```python
result = runner.predict(
    query,
    seed=1,
    return_embeddings=True,
    return_distogram=True,
)
print(result.embeddings["z"].shape)
result.save("predictions/test/", save_embeddings=True, save_distogram=True)
```

Single embeddings describe individual tokens; pair embeddings describe token pairs.
These arrays are shared across prediction samples and have padding removed:

| Key | Shape | Dtype |
| --- | --- | --- |
| `s_inputs` | `(num_tokens, single_channels)` | FP16 |
| `s_lm` | `(num_tokens, lm_channels)` | FP16 |
| `z` | `(num_tokens, num_tokens, pair_channels)` | FP16 |

Use `return_trajectory=True` with `save_trajectory=True` for trajectories.

## Preparing inputs separately

For a query with provided apo structures for every protein entry, load them and build the model input before prediction:

```python
apos, priors = runner.load_apo_and_prior(query)
item = runner.build_input(query, seed=1, num_samples=5, apos=apos, priors=priors)
result = runner.predict_from_input(item, seed=1, num_samples=5)
```

To generate apo structures independently, use `ApoRunner`.
Save the [apo configuration example](inference.md#apo-configuration) as `apo.yaml`:

```python
from kfold.inference.apo_runner import ApoConfig, ApoRunner

config = ApoConfig.load("apo.yaml")
apo_runner = ApoRunner(device="cuda", config=config)
prediction = apo_runner.predict(
    "protein", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANL", seeds=[11]
)
```

Pass a single sequence for a monomer or a tuple of two sequences for a protein pair.
The result contains `apos` and `priors`.
To use this configuration for automatic generation, pass `apo_config=config` to `KFoldRunner`.

Apo models stay loaded for repeated calls.
Call `apo_runner.unload_models()` when finished to release the folding models while retaining AtlasLM.
