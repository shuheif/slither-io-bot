"""Pre-flight check for the live game: run this FIRST on a new machine.

``python -m slither_bot probe``

1. verifies every JS global the bot depends on (see js_bridge.py);
2. joins a game — automatically if possible; if the game is joined but the
   snake state globals are invisible (a renamed client build), it scans
   ``window`` for the new names, live-verifies them, and continues; if the
   join fails outright it prints a full menu diagnosis and waits for you to
   click Play yourself;
3. measures the real speed (wu/s vs ``.sp``) and reports the radius estimate,
   so the unit constants can be corrected;
4. times the READ_STATE round-trip to confirm the control rate is feasible;
5. dumps one raw game state as a JSON fixture for the offline parse tests.

Every outcome (including failure and a closed browser window) writes
``probe_report.json`` and exits without a traceback.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from selenium.common.exceptions import WebDriverException

from slither_bot.live import js_bridge
from slither_bot.live.backend import LiveBackend

CRITICAL = ("snake", "snakes", "foods", "playing", "xm", "ym")


def _missing(globals_found: dict) -> list[str]:
    # None = undefined (truly missing); 'null' in-game means declared but
    # never populated — equally unusable, so both count.
    return [n for n in CRITICAL if globals_found.get(n) in (None, "null")]


def _rescue_renamed_globals(driver, backend: LiveBackend, report: dict) -> bool:
    """In-game but READ_STATE can't see the snake: find the renamed globals,
    apply them for this session, and verify end-to-end."""
    print("\nin-game but the snake state globals are missing — scanning window ...")
    scan = driver.execute_script(js_bridge.SCAN)
    report["scan"] = scan
    for a in scan.get("arrays", []):
        print(f"  array  window.{a['name']:<16} len={a['length']:<6} keys={a['keys']}")
    for o in scan.get("objects", []):
        print(f"  object window.{o['name']:<16} keys={o['keys']}")
    suggestion = scan.get("suggestion") or {}
    if not any(suggestion.get(k) for k in ("snake", "snakes", "foods")):
        print("scanner found no plausible candidates — full scan in probe_report.json")
        return False

    print(f"scanner suggestion: {suggestion}")
    js_bridge.set_global_names(
        snake=suggestion.get("snake"),
        snakes=suggestion.get("snakes"),
        foods=suggestion.get("foods"),
    )
    raw = backend._read_raw()
    if raw.get("playing") and not raw.get("snake_missing") and js_bridge.parse_state(raw).alive:
        names = dict(js_bridge.GLOBALS)
        report["renamed_globals"] = names
        print(f"\nVERIFIED renamed globals: {names}")
        print(
            "-> make this permanent: set these values in GLOBALS at the top of "
            "slither_bot/live/js_bridge.py"
        )
        return True
    print("suggested names did not verify — full scan in probe_report.json")
    return False


def cmd_probe(args) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = LiveBackend(
        url=args.url,
        play_timeout=getattr(args, "play_timeout", 30.0),
        force_server=getattr(args, "server", None),
    )
    report: dict = {"url": args.url}

    def write_report() -> None:
        (out_dir / "probe_report.json").write_text(json.dumps(report, indent=2))

    try:
        driver = backend._ensure_driver()
        print(f"opened {args.url}; checking globals on the menu page...")
        time.sleep(3.0)  # let the page boot

        globals_found = driver.execute_script(js_bridge.PROBE)
        report["globals"] = globals_found
        for name, kind in sorted(globals_found.items()):
            print(f"  window.{name:<16} {kind if kind is not None else 'MISSING'}")
        report["grd_menu"] = globals_found.get("grd")

        print("\njoining a game...")
        joined = False
        try:
            backend.reset()
            joined = True
        except TimeoutError as exc:
            report["join_error"] = str(exc)
            report["menu_diagnosis"] = backend.join_diagnosis
            report["join_attempts"] = backend.last_join_log
            print(f"\nauto-join FAILED: {exc}")
            if backend.join_diagnosis:
                print("\nfull menu diagnosis:")
                print(json.dumps(backend.join_diagnosis, indent=2))

            # Are we actually in a game whose state we just can't see?
            raw = backend._read_raw()
            if raw.get("playing"):
                if raw.get("snake_missing"):
                    joined = _rescue_renamed_globals(driver, backend, report)
                else:
                    joined = True  # joined in the race between timeout and now

            if not joined:
                headless = backend.headless or os.environ.get("SLITHER_BOT_HEADLESS") == "1"
                if headless:
                    print("\nheadless session — skipping the manual-join fallback")
                else:
                    wait = getattr(args, "manual_join_timeout", 90.0)
                    print(
                        f"\nclick Play manually in the opened Chrome window — "
                        f"waiting up to {wait:.0f}s ..."
                    )
                    deadline = time.monotonic() + wait
                    while time.monotonic() < deadline:
                        time.sleep(1.0)
                        try:
                            raw = backend._read_raw()
                        except WebDriverException as werr:
                            print(
                                f"browser window closed ({type(werr).__name__}) — "
                                "aborting the wait"
                            )
                            break
                        if not raw.get("playing"):
                            continue
                        if raw.get("snake_missing"):
                            joined = _rescue_renamed_globals(driver, backend, report)
                        else:
                            joined = True
                            report["joined_manually"] = True
                            print("joined manually — continuing the probe")
                        break

        if not joined:
            print("\nprobe could NOT join a game — diagnostics saved to probe_report.json")
            write_report()
            return 1
        report.setdefault("join_attempts", backend.last_join_log)

        globals_found = driver.execute_script(js_bridge.PROBE)
        report["globals_in_game"] = globals_found
        missing = _missing(globals_found)
        if missing:
            print(f"\nMISSING in-game globals: {missing}")
            print("scanning window for arrays of objects with .xx/.yy ...")
            scan = driver.execute_script(js_bridge.SCAN)
            report["scan"] = scan
            for c in scan.get("arrays", []):
                print(f"  window.{c['name']} length={c['length']} keys={c['keys']}")
            print(
                "\nUpdate the GLOBALS values at the top of slither_bot/live/js_bridge.py "
                "to match one of the candidates above, then re-run the probe."
            )
            write_report()
            return 1

        print("all critical globals present in-game; measuring...")

        # --- speed measurement: hold a straight heading, track displacement
        raw = driver.execute_script(js_bridge.READ_STATE, backend.view_radius)
        report["grd_in_game"] = raw.get("grd") if raw.get("playing") else None
        heading = float(raw["self"]["ang"]) if raw.get("playing") else 0.0
        driver.execute_script(js_bridge.STEER, heading)
        samples = []
        for _ in range(30):
            raw = driver.execute_script(js_bridge.READ_STATE, backend.view_radius)
            if not raw.get("playing") or raw.get("snake_missing"):
                break
            me = raw["self"]
            samples.append((raw["t"], me["x"], me["y"], me["sp"], me["sc"]))
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
            print(
                f"  speed: {speed:.1f} wu/s at sp={sp:.2f} -> SPEED_SCALE ~ {scale:.1f} "
                f"(js_bridge.py has {js_bridge.SPEED_SCALE})"
            )
            sc = float(np.mean([s[4] for s in samples]))
            report["sc"] = sc
            print(
                f"  scale sc={sc:.3f} -> radius estimate {js_bridge.RADIUS_PER_SC * sc:.1f} wu "
                f"(RADIUS_PER_SC = {js_bridge.RADIUS_PER_SC})"
            )
        if report.get("grd_in_game"):
            print(
                f"  map: grd={report['grd_in_game']} (menu default was {report.get('grd_menu')}) "
                f"-> arena center ({report['grd_in_game']}, {report['grd_in_game']}), "
                f"radius {report['grd_in_game']}"
            )

        # --- read latency
        times = []
        raw = None
        for _ in range(20):
            t0 = time.perf_counter()
            raw = driver.execute_script(js_bridge.READ_STATE, backend.view_radius)
            times.append((time.perf_counter() - t0) * 1e3)
        times.sort()
        report["read_ms_median"] = times[len(times) // 2]
        report["read_ms_max"] = times[-1]
        print(
            f"  READ_STATE round-trip: median {times[len(times) // 2]:.1f} ms, "
            f"max {times[-1]:.1f} ms (15 Hz needs < ~66 ms)"
        )

        # --- fixture dump for the offline parse tests
        if raw and raw.get("playing") and not raw.get("snake_missing"):
            fixture_path = Path(args.fixture)
            fixture_path.parent.mkdir(parents=True, exist_ok=True)
            fixture_path.write_text(json.dumps(raw, indent=2))
            report["fixture"] = str(fixture_path)
            n_pts = sum(len(e.get("pts", [])) for e in raw.get("enemies", []))
            print(
                f"  dumped state fixture -> {fixture_path} "
                f"({len(raw.get('enemies', []))} enemies, {n_pts} body points, "
                f"{len(raw.get('foods', []))} foods)"
            )
            percept = js_bridge.parse_state(raw)
            print(
                f"  parse_state OK: head={percept.self_snake.head}, "
                f"r={percept.self_snake.radius:.1f}, speed={percept.self_snake.speed:.1f}, "
                f"arena_radius={percept.arena_radius}"
            )

        write_report()
        print(f"\nwrote {out_dir / 'probe_report.json'} — probe PASSED")
        if report.get("renamed_globals"):
            print(
                f"\nREMINDER: this client renamed its state globals. Set GLOBALS in "
                f"slither_bot/live/js_bridge.py to {report['renamed_globals']} "
                f"so run/eval work without the probe."
            )
        return 0
    except WebDriverException as exc:
        first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        print(f"\nbrowser session ended unexpectedly ({type(exc).__name__}: {first_line})")
        report["error"] = f"{type(exc).__name__}: {first_line}"
        write_report()
        return 1
    finally:
        try:
            write_report()
        except Exception:
            pass
        backend.close()
