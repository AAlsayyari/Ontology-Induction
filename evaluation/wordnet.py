"""Rebuild the small, sense-disambiguated WordNet evaluation fixture.

Only this extraction tool needs NLTK/the original corpus. Normal evaluation
reads the committed JSON fixture and never downloads a dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "evaluation" / "benchmark.json"
CORPUS_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip"
CORPUS_SHA256 = "cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59"
CORPUS_PATH = ROOT / ".cache" / "evaluation" / "nltk_data" / "corpora" / "wordnet.zip"

# Chosen before running the model: familiar noun senses and their intermediate
# categories in eight subject areas. No selection depends on model predictions.
# WordNet's canonical synset identifiers resolve ambiguous words such as
# chicken (the bird), carrot (the food), heart (the organ), and compiler (software).
CASE_SYNSETS = {
    "mammals": (
        "mammal.n.01", "placental.n.01", "carnivore.n.01", "canine.n.02",
        "dog.n.01", "wolf.n.01", "fox.n.01", "feline.n.01", "cat.n.01",
        "big_cat.n.01", "lion.n.01", "equine.n.01", "horse.n.01",
        "cetacean.n.01", "whale.n.02", "bat.n.01",
    ),
    "birds": (
        "bird.n.01", "vertebrate.n.01", "bird_of_prey.n.01", "owl.n.01",
        "eagle.n.01", "hawk.n.01", "anseriform_bird.n.01", "duck.n.01",
        "domestic_fowl.n.01", "chicken.n.02", "columbiform_bird.n.01",
        "pigeon.n.01", "sphenisciform_seabird.n.01", "penguin.n.01",
        "passerine.n.01", "sparrow.n.01",
    ),
    "vehicles": (
        "vehicle.n.01", "wheeled_vehicle.n.01", "motor_vehicle.n.01",
        "car.n.01", "truck.n.01", "motorcycle.n.01", "bicycle.n.01",
        "public_transport.n.01", "bus.n.01", "train.n.01", "vessel.n.02",
        "boat.n.01", "ship.n.01", "aircraft.n.01", "heavier-than-air_craft.n.01",
        "airplane.n.01", "helicopter.n.01",
    ),
    "food": (
        "food.n.01", "produce.n.01", "vegetable.n.01", "root_vegetable.n.01",
        "carrot.n.03", "potato.n.01", "edible_fruit.n.01", "apple.n.01",
        "citrus.n.01", "orange.n.01", "lemon.n.01", "baked_goods.n.01",
        "bread.n.01", "starches.n.01", "grain.n.02", "rice.n.01",
    ),
    "musical_instruments": (
        "musical_instrument.n.01", "stringed_instrument.n.01",
        "bowed_stringed_instrument.n.01", "guitar.n.01", "violin.n.01",
        "cello.n.01", "keyboard_instrument.n.01", "piano.n.01",
        "wind_instrument.n.01", "woodwind.n.01", "flute.n.01", "brass.n.02",
        "cornet.n.01", "trombone.n.01", "percussion_instrument.n.01", "drum.n.01",
    ),
    "anatomy": (
        "body_part.n.01", "organ.n.01", "internal_organ.n.01", "heart.n.02",
        "liver.n.01", "excretory_organ.n.01", "kidney.n.01",
        "respiratory_organ.n.01", "lung.n.01", "blood_vessel.n.01",
        "artery.n.01", "vein.n.01", "connective_tissue.n.01", "bone.n.01",
        "skin.n.01", "body_covering.n.01",
    ),
    "occupations": (
        "person.n.01", "worker.n.01", "skilled_worker.n.01", "professional.n.01",
        "health_professional.n.01", "medical_practitioner.n.01", "doctor.n.01",
        "surgeon.n.01", "dentist.n.01", "nurse.n.01", "educator.n.01",
        "teacher.n.01", "craftsman.n.03", "woodworker.n.01", "carpenter.n.01",
        "electrician.n.01",
    ),
    "computing": (
        "machine.n.01", "computer.n.01", "portable_computer.n.01", "laptop.n.01",
        "personal_computer.n.01", "desktop_computer.n.01", "server.n.03",
        "software.n.01", "code.n.03", "operating_system.n.01", "program.n.07",
        "compiler.n.02", "interpreter.n.04", "device.n.01", "keyboard.n.01",
        "printer.n.03",
    ),
}

# These are existing WordNet lemmas, rather than new definitions/categories.
LABELS = {"brass.n.02": "brass instrument", "cornet.n.01": "trumpet"}


def load_corpus():
    """Download a checksum-pinned corpus if absent, and return NLTK's reader."""
    try:
        import nltk
        from nltk.corpus import wordnet
    except ImportError as exc:
        raise SystemExit("Install extraction dependencies: pip install -r requirements-eval.txt") from exc

    if not CORPUS_PATH.exists():
        CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(CORPUS_URL, timeout=60) as response:
            corpus_bytes = response.read()
        if hashlib.sha256(corpus_bytes).hexdigest() != CORPUS_SHA256:
            raise ValueError("Downloaded WordNet archive has an unexpected SHA-256 digest")
        CORPUS_PATH.write_bytes(corpus_bytes)
    if hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest() != CORPUS_SHA256:
        raise ValueError("Cached WordNet archive has an unexpected SHA-256 digest")
    nltk.data.path.insert(0, str(CORPUS_PATH.parents[1]))
    if wordnet.get_version() != "3.0":
        raise ValueError("Expected Princeton WordNet 3.0")
    return wordnet


def build_manifest() -> dict:
    """Extract *all* direct hypernym links among each case's selected senses."""
    wordnet = load_corpus()
    cases = []
    for case_id, synset_ids in CASE_SYNSETS.items():
        senses = [wordnet.synset(synset_id) for synset_id in synset_ids]
        labels = {
            sense.name(): LABELS.get(sense.name(), sense.name().rsplit(".", 2)[0].replace("_", " "))
            for sense in senses
        }
        for sense in senses:
            if labels[sense.name()].replace(" ", "_") not in sense.lemma_names():
                raise ValueError(f"Display label is not a WordNet lemma: {sense.name()}")
        if len(set(labels.values())) != len(senses):
            raise ValueError(f"Duplicate display labels in {case_id}")
        edges = [
            {"source": labels[sense.name()], "target": labels[parent.name()], "predicate": "subClassOf"}
            for sense in senses
            for parent in sorted(sense.hypernyms(), key=lambda item: item.name())
            if parent.name() in labels
        ]
        cases.append({
            "id": case_id,
            "terms": [labels[sense.name()] for sense in senses],
            "term_senses": {
                labels[sense.name()]: {
                    "synset": sense.name(),
                    "offset": sense.offset(),
                    "definition": sense.definition(),
                }
                for sense in senses
            },
            "ground_truth_edges": edges,
        })
    return {
        "benchmark_id": "wordnet-3.0-nouns-v1",
        "source": {
            "name": "Princeton WordNet",
            "version": "3.0",
            "url": "https://wordnet.princeton.edu/",
            "archive_url": CORPUS_URL,
            "archive_sha256": CORPUS_SHA256,
            "relationship": "direct hypernym",
            "license_file": "WORDNET-LICENSE.txt",
        },
        "selection": {
            "method": "Preselected familiar noun senses and intermediate categories in eight subject areas; no model-output-based selection.",
            "gold_policy": "Every WordNet hypernyms() link with both endpoints in the case; no transitive closure, instance links, or added commonsense relations.",
            "model_input": "Only case.terms; no synset identifiers, definitions, or ground-truth edges.",
            "sense_limitation": "Gold senses are explicit, but the model receives ambiguous surface words; sense disagreement is a benchmark error even when another reading is plausible.",
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the committed fixture against the checksum-pinned source")
    args = parser.parse_args()
    manifest = build_manifest()
    if args.check:
        if json.loads(MANIFEST.read_text(encoding="utf-8")) != manifest:
            raise SystemExit("Committed benchmark differs from the pinned WordNet extraction")
        print("WordNet fixture matches the pinned corpus")
    else:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        with zipfile.ZipFile(CORPUS_PATH) as archive:
            (MANIFEST.parent / "WORDNET-LICENSE.txt").write_bytes(archive.read("wordnet/LICENSE"))
        print(f"Wrote {len(manifest['cases'])} cases, {sum(len(case['terms']) for case in manifest['cases'])} terms, {sum(len(case['ground_truth_edges']) for case in manifest['cases'])} direct hypernym edges")


if __name__ == "__main__":
    main()
