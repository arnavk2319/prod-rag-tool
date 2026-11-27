"""Regenerate tests/contract/openapi.snapshot.json from the live FastAPI app."""

import json
from pathlib import Path

from rag_platform.api.app import create_app

SNAPSHOT = Path(__file__).resolve().parents[1] / "tests" / "contract" / "openapi.snapshot.json"


def main() -> None:
    doc = create_app().openapi()
    SNAPSHOT.write_text(json.dumps(doc, sort_keys=True, indent=2) + "\n")
    print(f"wrote {SNAPSHOT.relative_to(SNAPSHOT.parents[2])}")


if __name__ == "__main__":
    main()
