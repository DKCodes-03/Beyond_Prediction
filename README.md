# Beyond_Prediction
Beyond Prediction: Exploring Knowledge Graphs and Latent Reasoning in AI

## Status

The ICEWS14 structured source loader, temporal fact preprocessing,
deterministic graph-derived QA preparation, in-memory temporal graph
construction, temporal filtering, bounded subgraph retrieval, and a
graph-path sanity baseline are implemented. The question and temporal graph
encoder and learned non-SELR answer decoder are implemented; the SELR answer-only
training loop, validation, and resumable model/optimizer checkpoints are also
implemented. The SELR training entry point now loads prepared facts/splits,
retrieves and encodes train/validation examples, pads batches, logs callbacks,
and writes run metadata. Tiny CPU integration and controlled-learning tests
cover the end-to-end answer path and verify that SELR can fit a small synthetic
candidate task. A matched baseline-versus-SELR comparison runner now verifies
the frozen pilot artifacts, trains paired three-seed runs, and records the
test-set results. A bounded latent-conditioned explanation decoder and an
opt-in second-stage explanation trainer are implemented. Structured
explanation evaluation measures graph edge support, temporal consistency, and
supporting-path validity; faithfulness under perturbations remains future work.
The split directory contains only a Git placeholder; use the generated pilot
artifacts for reproduction.

The original proposal is the long-term vision. The starting scope follows the
Suggested MVP, using the full modular layout from Suggested MVP File Structure.

## Starting objective

Investigate whether Self-Explainable Latent Reasoning (SELR) improves temporal
multi-hop QA over a matched baseline:

ICEWS14 -> temporal graph -> subgraph retrieval -> graph encoder -> SELR -> answer.

Start with structured data and answer supervision. Add explanation generation
only after the answer-only pipeline works. Universal ingestion, training new
extractors, Neo4j, TimelineKGQA, and Qwen/QLoRA integration are not prerequisites
for the first experiment.

## Project layout

| Location | Purpose |
| --- | --- |
| `configs/` | Dataset, retrieval, model, and training settings |
| `data/raw/` | Local source datasets |
| `data/processed/` | Generated triples, graphs, retrieval caches, and embeddings |
| `data/splits/` | Generated train, validation, and test QA records |
| `src/ingestion/` | Structured loaders and preprocessing |
| `src/graph/` | Temporal graph construction and optional Neo4j adapter |
| `src/retrieval/` | Question-conditioned, temporally filtered subgraphs |
| `src/encoder/` | Reused graph models or simple embedding-based encoder |
| `src/selr/` | Latent bottleneck, reasoning, answer scoring, and explanation decoding |
| `src/llm/` | Answer decoding and later Qwen integration |
| `src/training/` | Answer-first SELR training and later QLoRA |
| `src/evaluation/` | Answer, explanation, and baseline evaluation |
| `src/utils/` | Logging, reproducibility, and checkpoint support |
| `notebooks/` | Empty exploration and analysis notebooks |
| `checkpoints/` | Local SELR, QLoRA, and selected model artifacts |
| `outputs/` | Local experiments, logs, figures, and reports |
| `tests/` | CPU temporal fixture and graph, retrieval, SELR, and pipeline tests |

See [plan.md](plan.md) for scope, design, and success criteria, and
[tasks.md](tasks.md) for the phased implementation checklist.
See [project_details.md](project_details.md) for the current project direction
and the rationale behind key choices.

The Phase 0 record contracts for canonical facts, QA records, retrieved
subgraphs, model outputs, and run metadata are defined in
[schemas/data_contracts.schema.json](schemas/data_contracts.schema.json).
Dates use ISO 8601 calendar-date form (`YYYY-MM-DD`); QA cutoffs are inclusive,
and failed retrieval/model outputs must be represented explicitly rather than
as successful-looking answers.

The four starter configurations in `configs/` follow the definitions in
[schemas/configs.schema.json](schemas/configs.schema.json). They encode the
synthetic historical-QA protocol and bounded CPU baseline defaults. The
acquired ICEWS14 revision and archive checksum are recorded in
`configs/dataset_config.yaml`; the SELR CLI parses its training, model, and
retrieval YAML configs and validates the fields it consumes.

## Setup

The verified starter environment uses Python 3.11.9, CPU-only PyTorch 2.7.1,
NumPy 2.2.6, NetworkX 3.4.2, pytest 9.1.1, and PyYAML 6.0.2. CUDA is not
required. The requirements file selects PyTorch's CPU wheel index so
installation does not pull a CUDA build.

With Python 3.11 available, create and activate a local virtual environment in
PowerShell, then install the pinned packages:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Alternatively, create the Conda environment from the repository root:

```powershell
conda env create -f environment.yml
conda activate beyond-prediction
```

Run this CPU smoke check to verify imports, PyTorch autograd/NumPy
interoperability, and a directed NetworkX path:

```powershell
python -c "import networkx as nx, numpy as np, pytest, torch; x = torch.tensor(2.0, requires_grad=True); (x * x).backward(); graph = nx.DiGraph([(1, 2), (2, 3)]); assert x.grad.item() == 4 and x.grad.numpy().item() == 4 and nx.has_path(graph, 1, 3); print(f'torch={torch.__version__}, numpy={np.__version__}, networkx={nx.__version__}, pytest={pytest.__version__}, cuda={torch.cuda.is_available()}')"
```

For reproducible runs, seed supported runtimes once near experiment startup:

```python
from src.utils.seed import set_seed
from src.utils.logging_utils import setup_logger

set_seed(17, deterministic=True, num_threads=2)
logger = setup_logger("beyond_prediction", log_dir="outputs/logs")
logger.info("experiment started")
```

The hand-checkable CPU graph fixture covers a two-hop path, inclusive temporal
cutoff boundaries, and missing evidence. `build_temporal_graph()` validates
canonical facts and builds a directed NetworkX multigraph keyed by stable fact
IDs, with deterministic adjacency indexes. Temporal cutoff filtering is
provided by `filter_facts_by_temporal_constraint()` in
`src/retrieval/temporal_filter.py`. Point constraints match the exact date;
interval constraints declare start- and end-boundary inclusivity independently.
The existing historical `on_or_before` cutoff includes facts on the cutoff date.
`retrieve_subgraph()` in `src/retrieval/subgraph_retriever.py` traverses the
query's ordered relation sequence from its seed entities. It returns a
schema-compliant subgraph with configurable hop/node/edge limits and explicit
`retrieval_status`, missing seed IDs, and a `limits_applied` indicator. Missing
seeds and present seeds with no matching evidence are distinct retrieval
outcomes.
`retrieval_cache_key()` in `src/retrieval/retrieval_utils.py` returns a stable
SHA-256 key over dataset and graph versions, split, normalized query, effective
retrieval settings, and example ID; equivalent default and explicit settings
share a key, while changes to any retrieval input isolate the cached result.
`assess_retrieval_support()` in the same module measures coverage of each QA
record's declared witness path in the retrieved fact IDs. It attributes an
incomplete witness to a retrieval miss, a wrong or unavailable answer despite
complete evidence to a reasoning error, and an explicit model failure
separately. `summarize_support_diagnostics()` reports complete-path coverage,
edge-level coverage, exact answer accuracy, and the per-category rates. These
are witness-path diagnostics, not a claim that every possible supporting path
or every accepted answer's evidence was retrieved.
`graph_path_sanity_baseline()` in `src/evaluation/path_baseline.py` independently
traverses the complete temporal graph using each query's relation sequence and
time constraint. It returns every reachable endpoint and one deterministic
witness path per endpoint, without reading answer labels. This graph-oracle
sanity check is separate from the learned non-SELR baseline implemented with
`CandidateAnswerDecoder`.
Verify graph construction, temporal filtering, and bounded retrieval with
`python -m pytest tests/test_graph.py tests/test_retrieval.py tests/test_path_baseline.py`.

`build_encoder_vocabulary()` in `src/encoder/node_embeddings.py` builds entity
and relation IDs from the complete permitted graph (a transductive vocabulary)
and fits question tokens only from records explicitly marked `split: train`.
`GraphQuestionEncoder` encodes a question and one retrieved subgraph using
learned entity/relation/token embeddings, a continuous calendar-date encoder,
and one directed mean-aggregation message-passing layer. Out-of-vocabulary
IDs/tokens use a shared unknown embedding, and candidate vectors preserve the
retriever's candidate ID order. PyTorch Geometric is not required. Run
`python -m pytest tests/test_encoder.py tests/test_retrieval.py` to check the
encoder contract and its retrieval inputs.

`CandidateAnswerDecoder` in `src/llm/answer_decoder.py` is the learned
non-SELR answer head. Its candidate set is exactly the retrieved subgraph's
`candidate_answer_ids` (the endpoints reached on the query's final hop), not
every entity in the graph. Training uses binary cross-entropy over candidates:
every accepted answer must be present in the retrieved candidate set and is
marked positive, so multiple correct answers are trained together; an omitted
accepted answer raises an explicit retrieval-coverage error instead of being
treated as a negative. At inference, all candidates with sigmoid score at least
0.5 are returned; if none cross the threshold, the top-scoring candidate is
returned. An empty candidate set is reported as `no_candidates`. Run
`python -m pytest tests/test_answer_decoder.py tests/test_encoder.py` to check
the baseline decoder and encoder.

`compute_answer_metrics()` in `src/evaluation/metrics.py` canonicalizes answer
IDs and compares prediction/label sets, reporting exact-match accuracy,
micro/macro precision, recall, and F1. Failed model outputs count as empty
predictions. `export_predictions()` writes schema-shaped prediction arrays;
the benchmark writes `predictions.json`, `metrics.json`, and `run.json`
atomically. Run the deterministic graph-path sanity benchmark on a generated
split with:

```powershell
python -m src.evaluation.benchmark --split test --seed 17 `
  --facts data/processed/facts.json `
  --split-file data/splits/test.json `
  --output-dir outputs/experiments/graph_path_sanity_test
```

This command is a graph/QA sanity check, separate from the learned
baseline-versus-SELR comparison described below.

### Task 34 baseline pilot

The first baseline run used a deterministic uniform sample of 2,000 of the
90,730 available ICEWS14 `Pt` fact rows (fact-sampling seed 17), then generated
and path-disjointly split synthetic QA with split seed 17 and fractions
0.8/0.1/0.1. The resulting graph contains 2,000 canonical facts and 17,696 QA
records; the held-out test split contains 1,770 records. On that test split,
the graph-path sanity baseline achieved 1.0 exact-match accuracy, micro/macro
precision, recall, and F1, with 4,022 gold and predicted answers and no failed
outputs. This graph oracle traverses the same sampled graph used to construct
the QA labels, so these perfect scores are a data/protocol sanity check, not
evidence of learned-model accuracy or full-dataset performance.

The full-data QA preparation attempt exceeded 3 GB resident memory while still
processing and was stopped before completion to avoid exhausting local memory,
so this pilot is explicitly not a replacement for a full-data run.
Its facts, splits, manifest, predictions, metrics, and run configuration are
preserved under
`outputs/experiments/graph_path_sanity_test_pilot/`. Re-run the recorded test
benchmark from those inputs with:

```powershell
python -m src.evaluation.benchmark --split test --seed 17 `
  --facts outputs/experiments/graph_path_sanity_test_pilot/input_artifacts/processed/facts.json `
  --split-file outputs/experiments/graph_path_sanity_test_pilot/input_artifacts/splits/test.json `
  --output-dir outputs/experiments/graph_path_sanity_test_pilot/reproduced_results
```

The run metadata records the ICEWS14 revision and archive checksum, sampled
fact count and seed, artifact hashes, split counts and seed, and evidence and
temporal policies. Generated experiment artifacts remain local and are not
tracked by Git.

### Task 35 frozen learned-model comparison

Before SELR experiments, the numeric targets and comparison protocol were
frozen in [`project_details.md`](project_details.md#frozen-task-35-baseline-versus-selr-protocol):
across paired seeds 17, 23, and 42, target at least a 0.05 (5 percentage-point)
improvement in mean test exact match and no more than 2.0x baseline p95
inference latency.
The comparison is limited to the exact 2,000-fact pilot artifacts above; it is
not a full-data or natural-language QA result. The Task 44 run completed all
six baseline/SELR trainings at the 2,000-update budget and evaluated the test
split after all checkpoints were fixed. Both variants achieved 1.0 mean exact
match across seeds 17, 23, and 42 (sample standard deviation 0.0); the paired
SELR-minus-baseline difference was 0.0, so the predeclared +0.05 SELR target
was not met. The pilot result is recorded in
`outputs/experiments/task44_learned_comparison/comparison.json`, including
input hashes, configs, training histories, test metrics, per-seed predictions,
and a source-tree checksum. It does not establish full-data or
natural-language QA generalization. Task 45's hop/time-type and retrieval
coverage breakdowns, plus inference-efficiency measurements, are saved in
`outputs/experiments/task44_learned_comparison/task45_metrics.json`. The report
records the CPU, batch size, timing scope, memory-measurement limitations, and
encoder-token definition.

### Task 46 verified preprocessing, training, evaluation, and test commands

Run these commands from the repository root with the Python 3.11 environment
activated as described above. The preprocessing command rebuilds the bounded
2,000-fact Task 34 pilot from the local ICEWS14 archive; it reads the 90,730
available fact rows, samples with seed 17 before QA generation, and writes to a
separate ignored output directory without replacing the frozen Task 34 inputs.
It requires `data/raw/icews14/tflex_icews14_all.zip` and
`data/raw/icews14/tflex_icews14_meta.json`.

```powershell
@'
import hashlib
import random
from pathlib import Path

from src.ingestion.artifact_store import save_processed_artifacts
from src.ingestion.json_loader import load_icews14_facts, load_icews14_metadata
from src.ingestion.preprocess import preprocess_facts
from src.ingestion.qa_preparation import prepare_qa
from src.ingestion.split_generation import split_qa_records

archive = Path("data/raw/icews14/tflex_icews14_all.zip")
metadata_path = Path("data/raw/icews14/tflex_icews14_meta.json")
raw_facts = load_icews14_facts(archive, metadata_path)
sampled_facts = random.Random(17).sample(raw_facts, 2000)
facts = preprocess_facts(sampled_facts)
metadata = load_icews14_metadata(metadata_path)
qa_records = prepare_qa(
    facts,
    entity_labels={str(key): value for key, value in metadata.entities.items()},
    relation_labels={str(key): value for key, value in metadata.relations.items()},
)
splits = split_qa_records(qa_records, seed=17)
output_root = Path("outputs/experiments/task46_icews14_preprocessing/input_artifacts")
manifest = save_processed_artifacts(
    facts,
    splits,
    output_root / "processed",
    output_root / "splits",
    dataset_id="ICEWS14",
    dataset_source="huggingface:linxy/ICEWS14",
    dataset_revision="462141693b1c9c2ece49384fab41955eae89471a",
    source_checksum_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    split_seed=17,
    split_fractions={"train": 0.8, "validation": 0.1, "test": 0.1},
)
split_counts = {name: len(records) for name, records in splits.items()}
print(
    f"facts={len(facts)}, qa={len(qa_records)}, "
    f"splits={split_counts}, "
    f"manifest={manifest}"
)
'@ | python -
```

The verified output is 2,000 canonical facts and 17,696 QA records, split into
14,156 train, 1,770 validation, and 1,770 test records. `save_processed_artifacts()`
writes the facts, splits, and provenance manifest.

The frozen Task 44 command below trains the matched baseline and SELR models
for seeds 17, 23, and 42, then evaluates their selected checkpoints. It uses
the preserved Task 34 pilot artifacts and writes to
`outputs/experiments/task44_learned_comparison/`:

```powershell
python -m src.evaluation.learned_comparison
```

The Task 34 graph-path evaluation command was also run against the preserved
test split; it writes the predictions, metrics, and run metadata to a separate
verification directory:

```powershell
python -m src.evaluation.benchmark --split test --seed 17 `
  --facts outputs/experiments/graph_path_sanity_test_pilot/input_artifacts/processed/facts.json `
  --split-file outputs/experiments/graph_path_sanity_test_pilot/input_artifacts/splits/test.json `
  --output-dir outputs/experiments/task46_benchmark_verification
```

Run the complete test suite from the activated environment:

```powershell
python -m pytest -q
```

The tests pass with the project's Python 3.11 environment.

### SELR training entry point

The answer-only training CLI consumes canonical facts plus the train and
validation QA arrays; it does not read the test split. Before running it,
explicitly select SELR in `configs/training_config.yaml`
(`model_variant: selr`) and `configs/model_config.yaml`
(`variant: selr`, `selr.enabled: true`). The standard baseline defaults remain
unchanged. For the frozen Task 35 pilot artifacts:

```powershell
python -m src.training.train_selr `
  --facts outputs/experiments/graph_path_sanity_test_pilot/input_artifacts/processed/facts.json `
  --train-split outputs/experiments/graph_path_sanity_test_pilot/input_artifacts/splits/train.json `
  --validation-split outputs/experiments/graph_path_sanity_test_pilot/input_artifacts/splits/val.json `
  --output-dir outputs/experiments/selr_seed17
```

The run uses `configs/training_config.yaml`, `configs/model_config.yaml`, and
`configs/retrieval_config.yaml`; pass `--seed 23` or `--seed 42` for the other
frozen protocol seeds, and use `--resume-from <checkpoint>` to resume. The
graph encoder is initialized from the run seed and frozen during SELR fitting;
the run record states this policy. Epoch and completion events go to the
configured log directory, while `run.json` records metrics, inputs, config,
and the selected checkpoint. QLoRA and explanation training are not invoked.

### Staged explanation training

`train_selr_explanations()` in `src/training/trainer.py` starts stage two from
the answer-only checkpoint and its exact training config. It validates that
checkpoint before enabling the bounded explanation decoder. Stage-two batches
retain the answer fields above and add `explanation_target_ids: [B, T]` and
`explanation_target_mask: [B, T]`; these are target token sequences with
trailing PAD positions and an EOS at each target's final valid position. The
trainer constructs shifted-right decoder inputs. Fit the token vocabulary using
training targets only. Configure the stage with
`objective: answer_plus_explanation` and an `explanation` object
containing `vocab_size`, `max_decode_steps`, `pad_token_id`, `start_token_id`,
`end_token_id`, and positive `loss_weight`. The objective is answer BCE plus
weighted masked token cross-entropy. Validation batches need only answer
fields; checkpoint selection remains based on validation answer exact match.
The standard answer-only CLI/config and checkpoints are unchanged.

### Task 50 structured explanation evaluation

`evaluate_explanations()` in `src/evaluation/evaluate_explanations.py` evaluates
one structured explanation per QA record. Each record contains `qa_id`,
`answer_id`, and `supporting_path`, an ordered list of claimed facts with
`subject_id`, `relation_id`, `object_id`, and `timestamp`; an optional `fact_id`
binds a claim to a specific graph fact. It reports edge-level factual support
and temporal-consistency rates separately, plus the fraction of complete paths
that start at a query seed, follow its ordered relations, connect edge to edge,
end at the stated answer, and use only graph-supported facts within the query's
temporal constraint. Missing, duplicate, or extra QA explanations are rejected.
These metrics measure structured evidence validity, not text fluency or
faithfulness. Verify with:

```powershell
python -m pytest tests/test_explanation_evaluation.py -q
```

### Task 53 answer regression comparison

Run the paired Task 53 experiment from the selected Task 44 seed-17 SELR
checkpoint:

```powershell
python -m src.evaluation.explanation_comparison
```

The runner constructs explanation targets from training-split graph paths only,
fits a training-only token vocabulary, and preserves the answer checkpoint's
training settings and seed. When a QA has multiple accepted answers, one target
is used: the path for its lexicographically first canonical answer ID. The
runner selects the explanation-stage checkpoint by validation answer exact
match, then evaluates both checkpoints on the same test split. Results, paired
predictions, and every post-training answer failure (classified as a regression
or persistent failure) are written to
`outputs/experiments/task53_explanation_comparison/`. Test results are not
written if explanation training reaches its runtime limit. Supply alternate
input paths, checkpoint, seed, or output directory with the corresponding
CLI options; an existing comparison result is never overwritten.

Run the complete frozen three-seed learned baseline-versus-SELR comparison
against the exact pilot artifacts with:

```powershell
python -m src.evaluation.learned_comparison
```

The comparison command verifies all four frozen artifact hashes and the
predeclared model, retrieval, optimizer, and stopping settings. It fits both
variants for seeds 17, 23, and 42, selects checkpoints using validation exact
match only, and evaluates the test split only after all six training runs
finish within budget. A time-capped run is recorded as incomplete and test
labels remain unused. The result manifest includes input hashes, code revision,
source-tree checksum, configs, training histories, and test metrics; per-seed
predictions are exported alongside it under
`outputs/experiments/task44_learned_comparison/`. Use a distinct `--output-dir`
to preserve an earlier run.

### Task 55 Qwen validation

Task 55 found that Qwen2.5-7B-Instruct is Apache 2.0 licensed and provides an
official GPTQ-Int4 variant. A Colab Tesla T4 successfully ran its NF4 4-bit
inference smoke test (1.67 seconds; 5.19 GiB peak PyTorch-allocated VRAM).
There is still no demonstrated research need to justify integration, so
Qwen/QLoRA remains deferred. The full run details and limitations are in
`outputs/reports/task55_qwen_validation.md`.

### Task 56 opt-in Qwen/QLoRA integration

The optional 4-bit inference and prompt helpers are implemented in
`src/llm/qwen_wrapper.py` and `src/llm/prompt_templates.py`. The LoRA-only
training entry point is `src/training/train_qlora.py`; it is not invoked by the
baseline or SELR paths and requires separate Transformers, Accelerate,
Datasets, PEFT, and bitsandbytes dependencies plus a CUDA GPU. Core adoption
remains gated by a predeclared answer or explanation comparison.

Verify the controlled tiny CPU learning check without a GPU:

```powershell
python -m pytest tests/test_training.py::test_selr_learns_controlled_tiny_cpu_fixture -q
```

This synthetic fit check verifies optimization and answer decoding on two
examples; it is not evidence of generalization or a learned-model result on
ICEWS14.

Run metadata can be created and written as JSON with
`create_run_metadata(...)` and `write_run_metadata(...)` from
`src/utils/logging_utils.py`.

No datasets, weights, credentials, or experiment results are tracked in Git.
ICEWS14 is downloaded locally and excluded from Git.

## ICEWS14 acquisition and provenance

The ICEWS14 TFLEX release is downloaded locally under `data/raw/icews14/` and
excluded from Git. It is pinned to Hugging Face dataset
[`linxy/ICEWS14`](https://huggingface.co/datasets/linxy/ICEWS14), revision
`462141693b1c9c2ece49384fab41955eae89471a`:

| Local file | Format and contents | Size | SHA-256 |
| --- | --- | ---: | --- |
| `tflex_icews14_all.zip` | ZIP containing `train.jsonl`, `valid.jsonl`, `test.jsonl` formal-query records | 20,350,171 bytes | `0bf107f5413221fef3703f5025814015d8e329d2fe31ca8446f06561a0b17c77` |
| `tflex_icews14_meta.json` | JSON metadata and entity/relation/timestamp vocabularies | 585,294 bytes | `886604155bc8f6973e0ba5c18f5b01ec001b191ded306ba6b26d350eb873f49b` |

The archive splits contain 759,569 / 158,782 / 170,418 query records
(train/validation/test). Records use integer-ID arrays (`query_name`, `query`,
`answer`, and optional `easy_answer`); the metadata supplies 7,128 entities,
230 relations, and 365 daily timestamp IDs for 2014. Selecting the `Pt` query
records and interpreting their answers as timestamp IDs yields 72,826 /
8,941 / 8,963 dated facts per split (90,730 total), verified against the
metadata counts. This is a query-oriented structured release, not a plain
triple TSV. The structured loader selects/converts the `Pt` records and decodes
their IDs through the vocabularies.

The Hugging Face dataset card declares Apache-2.0 for the TFLEX package, but
the release's own `data/ICEWS14/LICENSE` points to the underlying
[Harvard Dataverse ICEWS Coded Event Data](https://doi.org/10.7910/DVN/28075).
Its current [Dataverse terms](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/28075)
restrict use to research and education and prohibit commercial use; this
upstream restriction is distinct from the TFLEX package license (terms checked
2026-11-18). Retain the data locally for this academic research project and do
not redistribute it.
The source metadata and exact revision are also recorded in
`configs/dataset_config.yaml`. The structured loader in
`src/ingestion/json_loader.py` reads the vocabularies and archive, selects the
`Pt` query records, and expands each returned timestamp into a dated fact
record. It preserves vocabulary labels and source-split/row provenance; ID
canonicalization and duplicate handling are performed by `preprocess_facts()`
in `src/ingestion/preprocess.py`. Preprocessing trims IDs, stringifies integer
IDs, normalizes ISO 8601 dates/datetimes to calendar dates (timezone-aware
datetimes are converted to UTC first), and emits stable content-derived fact
IDs. Duplicate subject/relation/object/date facts are collapsed with the
lexicographically smallest source ID retained; malformed records and invalid
timestamps raise `ValueError` rather than being skipped.

TFLEX's additional formal temporal-logic query labels are not natural-language
QA supervision and will not be used as such. The project will generate its own
deterministic graph-derived questions as defined below.

## First QA task and evidence policy

The first experiment is historical temporal QA over ICEWS14, not forecasting.
Each query has an explicit cutoff and may retrieve any ICEWS14 fact dated on or
before it; facts after the cutoff are excluded. The complete fact graph is
available to train, validation, and test queries, so a source fact supporting
a held-out answer is allowed evidence. This is closed-knowledge-base QA, not
prediction of unseen facts, and results must state this policy.

QA question/path instances remain disjoint across splits. Training uses only
training labels, validation labels are for model selection, and test labels are
reserved for final evaluation. Retrieval caches must be isolated by query and
cutoff.

### QA supervision decision

Use deterministic synthetic questions generated from the ICEWS14 fact graph;
do not treat TFLEX formal-query labels as natural-language QA. This choice
keeps every answer independently checkable against the declared graph and
avoids presenting a different query language as natural language. The dataset
must be labeled synthetic in reports and prediction artifacts.

Generate two query types, with directed subject-to-object traversal, using
`prepare_qa()` from `src/ingestion/qa_preparation.py` after fact preprocessing:

| Template ID | Hops | Question template | Answer rule |
| --- | ---: | --- | --- |
| `one_hop` | 1 | `Which entity is connected to {subject} by relation {relation} on or before {cutoff}?` | Every object reached by one matching outgoing fact dated on or before the inclusive cutoff. |
| `two_hop` | 2 | `Which entity is reached from {subject} by relation {relation_1} followed by {relation_2}, using facts dated on or before {cutoff}?` | Every endpoint reached by exactly two matching outgoing facts, each dated on or before the inclusive cutoff. |

Render entity and relation placeholders from the dataset vocabularies when
available; retain canonical IDs as the actual query seeds, relation
constraints, answers, and provenance. The cutoff is an ISO 8601 calendar date.
The generator uses the latest date on a supporting path as the query cutoff.
Pass `entity_labels` and `relation_labels` maps when labels are not retained in
the preprocessed facts. Candidate queries are keyed by seed, ordered relation
sequence, and cutoff, and receive stable content-derived QA IDs. Prepared
records retain canonical relation sequences and remain unassigned to a split;
Task 19 performs split assignment.
Use `split_qa_records()` and `write_qa_splits()` from
`src/ingestion/split_generation.py` to assign prepared records and write
`train.json`, `val.json`, and `test.json`. The default seed and fractions match
`configs/dataset_config.yaml`. Duplicate QA IDs are rejected, and records
sharing the same ordered supporting-path fact IDs stay in one split. Each
record must retain its inclusive `on_or_before` cutoff. Split assignment
partitions QA labels only:
all splits may use the complete fact graph, filtered at each query's cutoff.
Training labels are for fitting, validation labels for model selection, and
test labels for final evaluation only. Keep generated data local. The tracked
`data/splits/.gitkeep` preserves the directory; generated `train.json`,
`val.json`, and `test.json` files are ignored by Git.
After preprocessing facts and assigning splits, persist them with
`save_processed_artifacts()` from `src/ingestion/artifact_store.py`, passing the
dataset identity/revision/checksum and configured split seed/fractions. It
writes canonical facts to `data/processed/facts.json`, the three existing
split-array files to `data/splits/`, and a `data/processed/manifest.json`
written last. The versioned manifest records record-schema references, source
and split provenance, record counts, file hashes, and hashes of each split's QA
IDs; its contract is `schemas/processed_artifacts.schema.json`. Artifacts
remain local and can be regenerated from the source facts and split settings.
For each template, independently enumerate the full set of matching paths in
the graph after applying the cutoff. The accepted answer IDs are the unique
endpoint IDs across those paths, not just the endpoint of a sampled witness.
Validate that the answer set is nonempty and that each answer has a matching
path whose facts satisfy the relation sequence and cutoff; reject invalid
generated examples explicitly. Store a deterministic supporting witness path
and supporting fact IDs as provenance. Compare predictions and labels as sets
of canonical entity IDs (exact match requires set equality); do not infer
answers by parsing the rendered question or normalizing entity names.

Queries use one or exactly two directed hops; they do not model arbitrary
natural-language paraphrases, inverse relations unless present in the data,
open-ended reasoning, or future-event prediction. Multiple endpoints are all
correct answers. Templates expose relation and entity names directly, which
can make the task easier than real user questions, and graph-derived questions
can favor frequent entities/relations and omit ambiguity in the source data.
Report these synthetic-template limitations and keep query/path instances
disjoint across splits.

Generated data, checkpoints, and outputs are ignored by Git; `.gitkeep` files
preserve the directory layout. Do not commit downloaded datasets or credentials.
