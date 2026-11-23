"""Build the removal rule's table removal-ratings.json (spire.removal, notes/solo/removal-spec.md section 4).

    PYTHONPATH=<tree> python -B scripts/build_removal_ratings.py --vocab <dist>/vocab/vocab.json \
        --catalog <dist>/vocab/catalog.json --out removal-ratings.json alt/solo/study/ratings-v2/*.tsv \
        alt/solo/removal/ratings/EXTRA.tsv

Every vocab card that can be in a deck (catalog type not Status) must be rated (the build asserts it); the card
classes (curses, each character's starter Strike / Defend) come from the catalog. Prints a one-line summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from spire.removal import build_table, read_tsv, load_rule
from spire.vocab import Catalog, Vocab


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("tsv", type=Path, nargs="+")
    a = ap.parse_args()
    vocab = Vocab.load(a.vocab)
    catalog = Catalog.load(a.catalog, vocab)
    table = build_table(vocab, catalog, read_tsv((str(p), p.read_text(encoding="utf-8")) for p in a.tsv))
    a.out.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    rule = load_rule(vocab, a.out)
    print(json.dumps({"out": str(a.out), "game_version": table["game_version"], "rated": len(table["ratings"]),
                      "curses": len(table["curses"]), "characters": sorted(table["strike"]),
                      "dropped": table["dropped"], "loaded_rated": int((rule.rating >= 0).sum())}))


if __name__ == "__main__":
    main()
