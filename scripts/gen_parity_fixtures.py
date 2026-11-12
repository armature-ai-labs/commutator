"""Generate parity fixtures: what solo-router (Python) decides for a grid of requests.

The Rust test crates/commutator-router/tests/parity.rs replays the same requests through
Commutator and requires identical results, field by field.

Usage:
    python scripts/gen_parity_fixtures.py /path/to/solo-router

Re-run when the default config or the request grid changes, and commit the output.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "crates/commutator-router/router.yaml"
OUT = ROOT / "tests/parity/fixtures.json"

# solo-router's own parity grid (tests/test_js_parity.py) ...
INSTRUCTIONS = [
    "pick up the red cup and place it on the tray", "stop!", "emergency, the arm is stuck",
    "hand me the scissors", "fold the towel", "patrol the warehouse aisle", "go to the kitchen",
    "clean the kitchen, then load the dishwasher and finally wipe the counters",
    "what do you see on the shelf", "why is joint 3 overheating", "hello, who are you", "",
]
EMBODIMENTS = [None, "arm", "so101", "humanoid", "quadruped", "drone"]
CONTEXTS = [
    {"observations": ["rgb", "proprio"], "control_hz": 20},
    {"observations": [], "network": "offline"},
    {"observations": ["wrist_rgb"], "control_hz": 100, "humans_nearby": True},
    {"observations": ["rgb"], "network": "degraded", "battery_pct": 10, "preference": "cost"},
    {"observations": ["rgb"], "latency_budget_ms": 400, "preference": "quality", "estop": False},
    {"observations": ["rgb"], "estop": True},
    {"observations": ["rgb"], "control_hz": 30, "overrides": {"intent": "perception", "horizon": "short"}},
    {"observations": ["rgb"], "control_hz": 30, "locations": ["browser"], "client_runtimes": ["webgpu", "wasm"]},
    {"observations": ["rgb"], "locations": ["browser", "edge"], "client_runtimes": []},
    {"observations": ["rgb"], "providers": ["openai"]},
    {"observations": [], "providers": ["nebius", "anthropic"], "network": "degraded"},
    {"observations": ["rgb"], "providers": [], "locations": ["cloud"]},
]

# ... plus extra cases that stress word boundaries, hazards, chunk budgets and preferences.
EXTRA_INSTRUCTIONS = [
    "Pick up the KNIFE carefully", "put the cupboard door back", "photo of the room please",
    "E-STOP", "stop moving now", "halt.", "please stop the music", "grab the hot pan, then place it",
    "turn left, then turn right, then go back to the dock", "calibrate the gripper encoder",
    "is there a person near the stove?", "count the blocks and stack them",
    "  Hand me   the glass  ", "what can you do", "make me a coffee and then clean the table",
]
EXTRA_CONTEXTS = [
    {},
    {"observations": ["rgb", "depth"], "control_hz": 50, "preference": "latency"},
    {"observations": ["rgb"], "control_hz": 10, "battery_pct": 15},
    {"observations": ["rgb", "proprio"], "control_hz": 200},
    {"observations": ["rgb"], "latency_budget_ms": 0, "preference": "balanced"},
    {"observations": ["rgb"], "humans_nearby": True, "embodiment": "lekiwi"},
    {"observations": ["rgb"], "network": "offline", "control_hz": 5},
    {"observations": ["audio"], "preference": "unknown-preference"},
]


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    solo = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(solo / "src"))
    from solo_router import Router, load_config  # noqa: PLC0415

    router = Router(load_config(CONFIG))
    routes = [{"instruction": i, "embodiment": e, **c}
              for i, e, c in itertools.product(INSTRUCTIONS, EMBODIMENTS, CONTEXTS)]
    routes += [{"instruction": i, "embodiment": "so101", **c}
               for i, c in itertools.product(EXTRA_INSTRUCTIONS, EXTRA_CONTEXTS)]
    tasks = [{"instruction": i, "embodiment": e, **c}
             for i, e, c in itertools.product(INSTRUCTIONS[:6], ["so101", "humanoid"], CONTEXTS)]

    commit = subprocess.run(["git", "-C", str(solo), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, check=False).stdout.strip()
    fixtures = {
        "source": {"repo": "https://github.com/ddiddi/solo-router", "commit": commit or "unknown"},
        "routes": [{"request": r, "expected": router.route(r).to_dict()} for r in routes],
        "tasks": [{"request": t, "expected": router.route_task(t).to_dict()} for t in tasks],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixtures, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {len(fixtures['routes'])} routes and {len(fixtures['tasks'])} tasks from solo-router {commit}")


if __name__ == "__main__":
    main()
