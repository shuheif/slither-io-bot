"""Pre-flight check for the live game: run this FIRST on a new machine.

``python -m slither_bot probe``

1. verifies every JS global the bot depends on (see js_bridge.py) and, if
   any are missing, scans ``window`` for plausible replacements to try;
2. joins a game and measures the real speed (wu/s vs ``.sp``) and reports
   the radius estimate, so the unit constants can be corrected;
3. times the READ_STATE round-trip to confirm the control rate is feasible;
4. dumps one raw game state as a JSON fixture for the offline parse tests.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from slither_bot.live import js_bridge
from slither_bot.live.backend import LiveBackend

CRITICAL = ("snake", "snakes", "foods", "playing", "xm", "ym")


def cmd_probe(args) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = LiveBackend(url=args.url)
    report: dict = {"url": args.url}
    try:
        driver = backend._ensure_driver()
        print(f"opened {args.url}; checking globals on the menu page...")
        time.sleep(3.0)  # let the page boot

        globals_found = driver.execute_script(js_bridge.PROBE)
        report["globals"] = globals_found
        missing = []
        for name, kind in globals_found.items():
            status = kind if kind is not None else "MISSING"
            print(f"  window.{name:<16} {status}")
            if name in CRITICAL and kind is None and name != "playing":
                missing.append(name)

        # xm/ym/snake often only appear once in-game; retry after joining.
        print("\njoining a game (clicking Play)...")
        percept = backend.reset()
        globals_found = driver.execute_script(js_bridge.PROBE)
        report["globals_in_game"] = globals_found
        missing = [n for n in CRITICAL if globals_found.get(n) is None]
        if missing:
            print(f"\nMISSING in-game globals: {missing}")
            print("scanning window for arrays of objects with .xx/.yy ...")
            candidates = driver.execute_script(js_bridge.SCAN)
            report["scan_candidates"] = candidates
            for c in candidates:
                print(f"  window.{c['name']} length={c['length']} keys={c['keys']}")
            print(
                "\nUpdate the names at the top of slither_bot/live/js_bridge.py "
                "to match one of the candidates above, then re-run the probe."
            )
            (out_dir / "probe_report.json").write_text(json.dumps(report, indent=2))
            return 1

        print("all critical globals present; measuring...")

        # --- speed measurement: hold a straight heading, track displacement
        raw0 = driver.execute_script(js_bridge.READ_STATE, backend.view_radius)
        heading = float(raw0["self"]["ang"])
        driver.execute_script(js_bridge.STEER, heading)
        samples = []
        for _ in range(30):
            raw = driver.execute_script(js_bridge.READ_STATE, backend.view_radius)
            if not raw.get("playing"):
                break
            samples.append((raw["t"], raw["self"]["x"], raw["self"]["y"], raw["self"]["sp"]))
            time.sleep(0.1)
        if len(samples) >= 10:
            t = np.array([s[0] for s in samples])
            xy = np.array([[s[1], s[2]] for s in samples])
            sp = float(np.mean([s[3] for s in samples]))
            dist = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
            speed = dist / max(t[-1] - t[0], 1e-6)
            scale = speed / max(sp, 1e-6)
            report["speed_wu_per_s"] = speed
            report["sp_mean"] = sp
            report["speed_scale_measured"] = scale
            print(f"  speed: {speed:.1f} wu/s at sp={sp:.2f} -> SPEED_SCALE ~ {scale:.1f} "
                  f"(js_bridge.py has {js_bridge.SPEED_SCALE})")
            sc = float(raw0["self"]["sc"])
            print(f"  scale sc={sc:.3f} -> radius estimate {js_bridge.RADIUS_PER_SC * sc:.1f} wu "
                  f"(RADIUS_PER_SC = {js_bridge.RADIUS_PER_SC})")
            report["sc"] = sc

        # --- read latency
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            raw = driver.execute_script(js_bridge.READ_STATE, backend.view_radius)
            times.append((time.perf_counter() - t0) * 1e3)
        times.sort()
        report["read_ms_median"] = times[len(times) // 2]
        report["read_ms_max"] = times[-1]
        print(f"  READ_STATE round-trip: median {times[len(times) // 2]:.1f} ms, "
              f"max {times[-1]:.1f} ms (15 Hz needs < ~66 ms)")

        # --- fixture dump for the offline parse tests
        if raw.get("playing"):
            fixture_path = Path(args.fixture)
            fixture_path.parent.mkdir(parents=True, exist_ok=True)
            fixture_path.write_text(json.dumps(raw, indent=2))
            report["fixture"] = str(fixture_path)
            n_pts = sum(len(e.get("pts", [])) for e in raw.get("enemies", []))
            print(f"  dumped state fixture -> {fixture_path} "
                  f"({len(raw.get('enemies', []))} enemies, {n_pts} body points, "
                  f"{len(raw.get('foods', []))} foods)")
            percept = js_bridge.parse_state(raw)
            print(f"  parse_state OK: head={percept.self_snake.head}, "
                  f"r={percept.self_snake.radius:.1f}, speed={percept.self_snake.speed:.1f}, "
                  f"arena_radius={percept.arena_radius}")

        (out_dir / "probe_report.json").write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out_dir / 'probe_report.json'} — probe PASSED")
        return 0
    finally:
        backend.close()
