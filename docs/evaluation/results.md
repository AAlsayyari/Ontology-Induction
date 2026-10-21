# Evaluation results - 2026-10-15

The current model is a **graph proposal prototype with weak semantic labeling**.
Structural validation and automatic layout work, but the sample exposes frequent
reversed type assertions. These are measured limitations, not verified ontology
output.

## Setup and provenance

- Model: Qwen/Qwen2.5-3B-Instruct, BF16, greedy decoding, model seed 42.
- Hardware: NVIDIA GeForce RTX 5080, 16 GB; Python 3.12.12, PyTorch 2.7.1+cu128,
  Transformers 4.55.4, LM Format Enforcer 0.11.3.
- Reference: [Princeton WordNet 3.0](https://wordnet.princeton.edu/), frozen into
  [eight cases](benchmark.json) before inference: 129 term instances and 107
  direct hypernym links. Each case is a separate graph, not one 129-node graph.
- The model receives only the terms, not gold links, sense IDs, or definitions.
  Production prompts are unchanged. Default maximum node degree is 3.
- [Benchmark provenance](benchmark.md) records the source checksum, explicit word
  senses, selection policy, and license. [Saved predictions](predictions.json)
  include runtime fingerprints; [results.json](results.json) records scorer
  fingerprints separately, including the later type-alias cycle diagnostic.

## Ground-truth matching

A true positive is an unordered predicted pair that exactly matches a **direct**
WordNet hypernym pair. Duplicate pairs are counted once. Missing reference pairs
are false negatives; predicted pairs absent from the reference are false
positives for this specific recovery task. Such mismatches can still be valid
associations or broader ancestor relationships.

| Version | Completed graphs | TP | FP | FN | Precision % | Recall % | F1 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Before self-link decoding guard | 7/8 | 32 | 35 | 75 | 47.76 | 29.91 | 36.78 |
| Current decoder | 8/8 | 41 | 39 | 66 | 51.25 | 38.32 | 43.85 |

These are micro scores. Current macro Precision/Recall/F1 are
51.35% / 38.46% /
43.41%.

The original mammals run failed because the model proposed `canine -> canine`.
That failed case remains in the baseline recall denominator as empty predictions.
The production JSON schema now excludes each source from its target vocabulary;
all eight cases were rerun with the same frozen inputs. No taxonomic facts were
hardcoded to improve matching. Original [predictions](baseline_predictions.json)
and [scores](baseline_results.json) are preserved. Baseline descriptions were
not generated or reviewed.

| Case | Terms | Predicted pairs | Gold pairs | Precision % | Recall % | F1 % | Isolated | Cycle rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mammals | 16 | 13 | 13 | 69.23 | 69.23 | 69.23 | 1 | 1 |
| birds | 16 | 7 | 11 | 85.71 | 54.55 | 66.67 | 5 | 0 |
| vehicles | 17 | 9 | 12 | 66.67 | 50.00 | 57.14 | 4 | 0 |
| food | 16 | 13 | 14 | 38.46 | 35.71 | 37.04 | 1 | 0 |
| musical instruments | 16 | 10 | 17 | 80.00 | 47.06 | 59.26 | 2 | 0 |
| anatomy | 16 | 11 | 13 | 9.09 | 7.69 | 8.33 | 7 | 3 |
| occupations | 16 | 12 | 14 | 41.67 | 35.71 | 38.46 | 3 | 1 |
| computing | 16 | 5 | 13 | 20.00 | 7.69 | 11.11 | 9 | 0 |

The degree-3 constraint permits at most 102 of these 107 reference links:
a **95.33% recall ceiling from degree limits alone**. Reported recall is not
adjusted upward. This ceiling does not explain the much larger observed number
of missed links.

## Graph structure

| Check | Current observation | Assessment |
| --- | --- | --- |
| Self-links | 0 | Excluded during constrained decoding |
| Duplicate/reverse pairs | 0 | Deduplicated and validated |
| Unknown endpoints | 0 | Input-only endpoint vocabularies |
| Maximum node degree | 3 | Enforced cap; no cap violations |
| Universal hubs | 0 | Largest neighbor fraction within a case: 20% |
| Isolated terms | 32/129 (24.81%) | Coverage remains weak; reference graphs contain only 1 isolate |
| Undirected cycles | Cycle rank 5 across 3 cases | Topology is not cycle-free |
| Sampled directed type cycles | 0 among 32 recognized type edges | Partial check, not full-graph certification |

Cycle rank is `edges - nodes + connected components`: the number of independent
undirected cycles, not a count of every possible circular walk. Cycles occur in
mammals (1), anatomy (3), and occupations (1). The reference instrument taxonomy
itself has two undirected cycles due to multiple inheritance, so an undirected
cycle alone does not establish a contradictory hierarchy.

The directed diagnostic recognizes explicit type aliases such as `isA` and
`subClassOf`, and reverses documented parent-to-child aliases. Predicate rules
and coverage counts are in the JSON report. Other relationship kinds are not
treated as type edges. Only 50 of the 80 connections were described; unselected
edges were not checked for directed semantic cycles. Thus the requested ideal
of a fully connected, cycle-free graph is **not established** by this run.

## Review of 50 generated pairs

All 80 generated connections entered a uniform sample without replacement,
using sampling seed 42. The 50 chosen pairs were fixed before descriptions were
requested. All 50 descriptions completed and asserted support; there were no
abstentions or replacement samples.

The [completed CSV](manual_review.csv) contains every source, predicate, target,
explanation, individual judgment, reason, and evidence reference. This is an
**assistant qualitative review**, not independent human or clinical expert
validation. The [review protocol](review_protocol.md) explains the rubric and a
recorded clarification separating true broader membership from exact-parent
recovery.

| Review measure | Result |
| --- | --- |
| Complete claim plus explanation factually correct | 6/50 (12.00%) |
| Incorrect | 39/50 |
| Uncertain | 5/50 |
| Correct direction | 9/48 applicable pairs (18.75%) |
| Reversed direction | 29/48 |
| Uncertain direction | 10/48 |
| Symmetric predicates, direction not applicable | 2/50 |

Uncertain answers do not count as correct. Direction examines the emitted edge;
a correct sentence cannot repair a reversed edge. Factual correctness requires
both a defensible predicate claim and a compatible explanation.

Examples that distinguish these failures:

- **pair-002:** `big cat isA lion` reverses the correct statement in its own prose.
- **pair-006:** `sparrow isA passerine` points correctly, but the explanation
  falsely says passerine is a subclass of sparrow.
- **pair-033:** `kidney hasPart artery` is defensible; its explanation that a
  kidney produces an artery is incorrect.
- **pair-014:** `helicopter isA aircraft` is true and correctly directed, while
  missing the nearer input category `heavier-than-air craft`.
- **pair-046:** The penguin/sphenisciform labels are nearly coextensive in common
  biology despite their ordered WordNet concepts; the qualitative verdict is
  uncertain. The exact reference score is unchanged.

Weakly supported teaching scenarios and broad predation claims are marked
uncertain. Overall, the dominant defect is incorrect relation direction,
frequently contradicted by otherwise fluent explanations. The current model
should be treated as a proposal generator requiring review.

## Automatic layout verification

[Browser geometry results](layout_results.json) use synthetic fixtures to test
rendering, separately from factual model evaluation. They include 2-200 nodes,
long and unbroken labels, disconnected components, cycles, label toggling,
resizing, obstructed edges, and reversed relation endpoints.

| Geometry check | Preset positions, 36 long labels | Automatic layout, same fixture |
| --- | ---: | ---: |
| Overlapping node boxes | 24 | 0 |
| Edges through unrelated nodes | 42 | 0 |

All tested automatic-layout fixtures had zero node overlaps, label overflow,
and edges through unrelated nodes. A typical disconnected 200-node fixture took
94.4 ms; an adversarial connected 200-node graph with long labels took 3657.6 ms
on this workstation. These are browser layout timings, not model inference
benchmarks. Complex graphs can still have edge-to-edge crossings.

[Saved-graph UI replay](ui_replay_results.json) also renders an actual evaluated
vehicle graph, opens its saved relation description, checks repeat-click caching,
and exercises Fit graph without another GPU run. The README screenshots come
from that replay. This test reuses real predictions; it does not claim fresh
model inference or production server cache testing. Unit API tests cover the
server queue and cache behavior separately.

## Reproduce

Run fresh inference with the GPU environment from the main README:

```powershell
.\.venv\Scripts\python.exe -m evaluation.run --run-dir .cache/evaluation/new-run
```

Interrupted runs can resume with `--resume` and the same run directory/model.
New predictions and raw model output stay in the ignored cache. To recalculate
this saved result without a GPU or corpus download:

```powershell
.\.venv\Scripts\python.exe -m evaluation.run --score-only --checkpoint docs/evaluation/predictions.json --manual-review docs/evaluation/manual_review.csv --run-dir .cache/evaluation/rescore
```

Run browser checks after installing `requirements-dev.txt` and a Playwright
browser (or use `--browser-channel chrome` for installed Chrome):

```powershell
.\.venv\Scripts\python.exe tests/browser_layout.py
.\.venv\Scripts\python.exe tests/browser_topology_races.py
.\.venv\Scripts\python.exe tests/browser_saved_graph.py
```

The 35 unit tests pass; the CPU CI dependencies require neither model weights,
PyTorch, Transformers, nor NLTK. NLTK is optional only for rebuilding/verifying
the checksum-pinned WordNet extraction.

## Limits of these measurements

This is a small curated noun-taxonomy diagnostic, with general categories often
listed before their descendants. Input-order robustness was not measured. The
model may have encountered WordNet during pretraining; no held-out knowledge
claim is made. These scores do not establish performance on mixed-domain lists,
other languages, clinical ontologies, financial ontologies, or arbitrary 200-term
inference. Synthetic 200-node layout tests establish geometry behavior only.
Assistant judgments have no independent human agreement estimate. Runtime
fingerprints and saved outputs support auditing, but a missing upstream model
revision and hardware differences limit bit-for-bit regeneration guarantees.
