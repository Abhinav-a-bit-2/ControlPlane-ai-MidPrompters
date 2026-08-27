"""
Layer 2: Fast Heuristic & Deterministic Detection

Cheap regex firewall for known, obvious attack strings. This is NOT meant
to catch everything (that's Layer 3's job) — it's meant to reject the
common, unsophisticated cases in <1ms without spending an LLM call.

Patterns live in a versioned JSON file (not hardcoded) so the blocklist
can be updated without a code deploy, and so you can log which *version*
of the blocklist caught something during the demo.
"""
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("security.layer2")

_PATTERNS_FILE = Path(__file__).parent / "blocklist.json"


@dataclass
class HeuristicResult:
    passed: bool
    matched_pattern: str = ""
    blocklist_version: str = ""


def _load_blocklist() -> dict:
    with open(_PATTERNS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def check_heuristics(text: str) -> HeuristicResult:
    data = _load_blocklist()
    version = data.get("version", "unknown")
    compiled = [re.compile(p, re.IGNORECASE) for p in data.get("patterns", [])]

    for pattern, compiled_pattern in zip(data.get("patterns", []), compiled):
        if compiled_pattern.search(text):
            logger.warning(
                "heuristic_block", extra={"pattern": pattern, "blocklist_version": version}
            )
            return HeuristicResult(
                passed=False, matched_pattern=pattern, blocklist_version=version
            )

    return HeuristicResult(passed=True, blocklist_version=version)
