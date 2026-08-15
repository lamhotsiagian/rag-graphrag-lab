#!/usr/bin/env python3
"""Regenerate the relevant_chunk_ids in the golden dataset.

Chapter 6 warns that relevance judgements key on chunk identifiers, and that
identifiers change whenever the chunking configuration changes. This script
makes that maintenance one command instead of a silent source of zeros.

It re-chunks the source document with the configuration recorded in the
dataset, then labels a chunk relevant when it contains the expected answer's
distinctive terms. That heuristic is a starting point, not a substitute for
human judgement: review the output before trusting the metrics.

    python scripts/rebuild_golden_ids.py [--write]
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chapter_02.chunkers import recursive_chunking          # noqa: E402
from chapter_02.loaders import clean_text                   # noqa: E402
from shared.db_utils import content_hash                    # noqa: E402

DATASET = PROJECT_ROOT / "chapter_06" / "golden_dataset.json"
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "its",
    "such", "using", "into", "each", "which", "they", "their", "them", "have",
    "learning", "data", "main", "types", "three",
}


def distinctive_terms(text: str, minimum: int = 4) -> set:
    words = re.findall(r"[a-z]{%d,}" % minimum, text.lower())
    return {w for w in words if w not in STOPWORDS}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="write the dataset back instead of printing a preview")
    args = parser.parse_args()

    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    config = dataset["chunking"]
    source = PROJECT_ROOT / config["source"]

    text = clean_text(source.read_text(encoding="utf-8"))
    chunks = recursive_chunking(text, config["size"], config["overlap"])
    ids = [content_hash(chunk, config["source"], i) for i, chunk in enumerate(chunks)]
    print(f"{len(chunks)} chunks from {config['source']}")

    for case in dataset["cases"]:
        if not case.get("is_answerable", True):
            case["relevant_chunk_ids"] = []
            case["relevance_grades"] = {}
            continue

        wanted = distinctive_terms(
            f"{case['question']} {case.get('expected_answer', '')}"
        )
        scored = []
        for chunk, chunk_id in zip(chunks, ids):
            overlap = len(wanted & distinctive_terms(chunk))
            if overlap:
                scored.append((overlap, chunk_id))

        scored.sort(reverse=True)
        top = scored[:3]
        case["relevant_chunk_ids"] = [chunk_id for _, chunk_id in top]
        # Grade 2 for the strongest match, 1 for supporting context.
        case["relevance_grades"] = {
            chunk_id: (2 if rank == 0 else 1)
            for rank, (_, chunk_id) in enumerate(top)
        }
        print(f"  {case['case_id']:<16} {len(top)} relevant chunk(s)")

    if args.write:
        DATASET.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {DATASET.relative_to(PROJECT_ROOT)}")
    else:
        print("\npreview only, pass --write to save")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
