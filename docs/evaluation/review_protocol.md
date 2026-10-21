# Qualitative review protocol

The benchmark and sampling are fixed before reading the generated descriptions.
This is a small diagnostic, not a claim of clinical validity or performance on
all domains.

## Sampling

Generate every benchmark topology with the application's unchanged prompts,
default BF16 model, greedy decoding, seed 42, and degree cap 3. Pool the generated
pairs across cases and select 50 uniformly without replacement using sampling
seed 42. Request descriptions for those exact pairs in their original term and
cluster context. Keep abstentions and generation failures in the sample; do not
replace difficult examples with successful ones.

## Review

The reviewer is the coding assistant, performing an explicit qualitative review
of the saved outputs against the selected WordNet senses and ordinary factual
knowledge. This is **assistant-reviewed**, not an independent human annotation
study, expert sign-off, or a second automated call to the application model.
The completed CSV records an individual judgment and reason for every pair.

- **Factual accuracy — correct:** The predicate and explanation express a
  defensible relationship between these senses. A plausible explanation alone
  does not excuse a false predicate. A type assertion must express actual subtype
  membership. An ordinary `isA` assertion can name a true broader ancestor;
  failure to recover a direct parent remains an exact-reference mismatch and is
  noted separately from factual truth.
- **Factual accuracy — incorrect:** The claim is contradicted by the selected
  senses, confuses categories with parts, claims unjustified exclusivity, or
  invents an unsupported function or causal relation.
- **Factual accuracy — uncertain:** The terms are ambiguous, evidence is
  insufficient, no description was produced, or the model abstained. Uncertain
  examples do not count as correct.
- **Direction — correct/reversed:** Evaluate the source and target in the order
  required by the claimed predicate. Do not infer direction from the original
  undirected topology pair or from a fluent explanation alone.
- **Direction — uncertain:** The predicate is too unclear or invalid to judge
  its direction reliably.
- **Direction — not applicable:** The model abstained, no predicate was emitted,
  or the predicate is symmetric. Exclude these rows from directed accuracy's
  denominator and report their number separately.

Reference matching and qualitative review answer different questions. The
reference score requires an **exact direct WordNet hypernym pair**. A real
association or a more distant ancestor can fail that score while still being
factually valid. Review notes must identify this distinction rather than changing
the reference or relabeling a mismatch after seeing predictions.

Rubric clarification recorded during review: an earlier draft combined factual
truth with the application's preference for immediate subtype links. The final
review separates these questions consistently across all 50 rows. For example,
`helicopter isA aircraft` is factually true and correctly directed even when
`heavier-than-air craft` is a nearer input category. It still fails the direct
WordNet connection match. This clarification changes neither the fixed sample
nor any reference score; the CSV notes the broader-ancestor issue explicitly.

The review reports correct, incorrect, and uncertain counts, factual accuracy
over all 50 reviewed pairs, and correct direction over direction-applicable
pairs. These are descriptive proportions of this fixed sample; no confidence
interval, inter-rater agreement, or broad accuracy claim is implied.
