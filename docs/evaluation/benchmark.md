# WordNet benchmark provenance and scope

[benchmark.json](benchmark.json) is a small, frozen extraction from **Princeton
WordNet 3.0**: eight subject areas, 129 input terms, and 107 direct hypernym
links. It covers mammals, birds, vehicles, food, musical instruments, anatomy,
occupations, and computing. The original corpus remains in the ignored local
cache; it is not included in the repository.

The subject areas and noun senses were chosen before any benchmark model
predictions were generated. This is a curated diagnostic set, rather than a
random or independently held-out benchmark. WordNet may have appeared in the
model's pretraining data. These measurements should not be described as evidence
of generalization to unseen knowledge.

## Ground truth

WordNet identifies concepts using synonym sets, or synsets. Each fixture term
has an explicit synset identifier, offset, and definition to resolve ambiguity.
For example, `chicken.n.02` denotes the bird, and `compiler.n.02` denotes the
software tool. The model receives only the display terms, so it can still choose
a different sense. The benchmark counts that disagreement as an error.
[NLTK's WordNet interface documentation](https://www.nltk.org/howto/wordnet.html)
explains the sense identifiers and hypernym traversal.

For each case, the extractor includes **every direct hypernym link whose two
endpoints were selected**. It does not select favorable edges, add inferred
commonsense links, include instance hypernyms, or take the transitive closure.
The direction is always **specific concept → direct broader concept**, with
the predicate `subClassOf`. Princeton documents these immediate hypernym
relations in its [WordNet reference manual](https://wordnet.princeton.edu/documentation/wn1wn).

Consequently, these precision, recall, and F1 scores measure **exact taxonomy
recovery**. A valid non-taxonomic connection, such as an ingredient or component
relation, can be absent from this gold set. An absent gold pair is an exact-match
false positive; it does not automatically establish that the connection is
factually false. The separate sample review assesses factual support and
relation direction.

| Case | Terms | Direct gold links | Highest gold node degree |
| --- | ---: | ---: | ---: |
| Mammals | 16 | 13 | 4 |
| Birds | 16 | 11 | 4 |
| Vehicles | 17 | 12 | 3 |
| Food | 16 | 14 | 3 |
| Musical instruments | 16 | 17 | 4 |
| Anatomy | 16 | 13 | 5 |
| Occupations | 16 | 14 | 3 |
| Computing | 16 | 13 | 3 |
| **Total** | **129** | **107** | **5** |

The app normally limits each node to three connections. Some gold graphs have
nodes with four or five links, so perfect recall is incompatible with that
limit. WordNet also permits multiple inheritance: its directed hypernym links
can be acyclic while the corresponding undirected graph contains a cycle.
These facts matter when interpreting structural checks and a sparse layout.

## Reproduce the extraction

Normal evaluation reads the committed JSON and does not require NLTK or network
access. To independently verify its provenance:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-eval.txt
.\.venv\Scripts\python.exe -m evaluation.wordnet --check
```

To rebuild the fixture and its license file, omit `--check`. The extractor
downloads the official NLTK corpus archive if needed and refuses an unexpected
SHA-256 digest. The pinned archive is:

- Source: [NLTK's WordNet corpus package](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip)
- Version: Princeton WordNet 3.0
- SHA-256: `cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59`
- Local download: `.cache/evaluation/nltk_data/corpora/wordnet.zip`
- Extraction code and the fixed sense lists: [evaluation/wordnet.py](../../evaluation/wordnet.py)

## Attribution

WordNet 3.0 Copyright 2006 by Princeton University. All rights reserved.
The fixture contains a modified subset of the WordNet database, including noun
definitions and selected hypernym links. The complete upstream license is
preserved in [WORDNET-LICENSE.txt](WORDNET-LICENSE.txt). Princeton publishes the
[license and commercial-use terms](https://wordnet.princeton.edu/license-and-commercial-use).
