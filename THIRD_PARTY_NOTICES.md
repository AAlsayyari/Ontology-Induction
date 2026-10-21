# Third-party notices

## Cytoscape.js

The desktop graph interface includes [Cytoscape.js](https://js.cytoscape.org/) in
`web/vendor/cytoscape.min.js`. Its original license header is preserved in that
file. Cytoscape.js is distributed under the MIT License:

Copyright (c) 2016-2025, The Cytoscape Consortium.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Qwen model

The default inference model is
[Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).
Model weights and tokenizer files are downloaded separately and are excluded
from this repository.

The model is governed by the
[Qwen Research License Agreement](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE),
which permits non-commercial research and evaluation; commercial use requires
a license from the model provider. That agreement applies to the model materials
independently of this application's source code.

Other installed Python dependencies retain their respective upstream licenses.

## WordNet evaluation fixture

`docs/evaluation/benchmark.json` contains a small modified subset of Princeton
WordNet 3.0 noun senses, definitions, and direct hypernym links, used only for
evaluation. WordNet 3.0 Copyright 2006 by Princeton University. All rights
reserved. Its complete license is retained in
[docs/evaluation/WORDNET-LICENSE.txt](docs/evaluation/WORDNET-LICENSE.txt).
See [the benchmark provenance](docs/evaluation/benchmark.md) for the source,
archive checksum, selection method, and extraction instructions. The full
corpus is downloaded separately into the ignored local cache.
