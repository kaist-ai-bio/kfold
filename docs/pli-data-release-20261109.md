# Frozen PLI feature/cache release — 2026-11-09

The frozen71K EMA/source368a9d6e backbone produced one indexed full-input representation per record (three recycles, LM dropout0.25, immutable index seed). Stored BF16 tensors include s_inputs/s_lm, directedPL/LP/LLpairs and raw distograms; PP is not retained. Query200, frozen training-consensus200 and query350 share one tensor union and explicit indices. Historical aliases retain separate routing; unavailable/unstable evidence is explicitly marked, never silently replaced.

The release contains2,817,891physical cached input records:2,795,993original/decoy records after383approved validation-only exclusions, plus21,898BioLiP3additional feature inputs. There are16,731payload files totaling25,922,781,806,592bytes. BioLiP3 adds23,576training supervision rows (23,573main and3censored rows on existing inputs), no new temporal validation. The admitted measurements supplement our frozen corpus and are corroborated in officialChEMBL35; they are not claimed new relative to the full parent database. Fifteen ambiguous duplicate-measurement candidates remain outside the release.

GPU1245739completed0:0. CPU1245909completed0:0 and checked exact new input coverage, all new sourceSHA, retained tensor equality, readback, actual head loaders and unique catalog coverage. The prior parent audit verified all16,686cache payload fileSHAs and sampled25,115decoded records; it did not decode every record again. Existing payloads, plans, sources, labels and apo provenance are unchanged.

The main and supplementary label paths and frozen science context are linked in configs/affinity/data_release_v1.json. This commit freezes feature preparation, not a final training membership claim: legacy val overlaps train, external novel-ligand and known-drug kinase evaluations must remain distinct, censored relations and synthetic provenance require explicit task masks. The next stage constructs and audits head-training manifests.

A separate pre-trunk apo side-cache proposal was withdrawn by the user before production. Existing s_lm may supply a trainable outer-product head ablation. Completed producer probes remain operational evidence and do not change this cache contract.

Raw originals are being copied with destination readbackSHA to the user's Hangang/r902a13 account; preservation copy completion and source-retirement gates are separate. No raw or cache deletion is enabled. Tensor/model/data files and authentication material are not included in Git.

Validation:166CPU affinity tests passed before the release commit. SQLite bundles retain lossless per-record compression and permit future container export; no LMDB repack was performed.
