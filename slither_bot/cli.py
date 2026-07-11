"""Command-line interface: run | eval | probe."""

import argparse


def _add_common_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=["sim", "live", "vision"], default="sim")
    p.add_argument("--url", default="http://slither.io", help="game URL (live/vision backends)")
    p.add_argument("--play-timeout", type=float, default=30.0,
                   help="seconds to wait for auto-join before failing (live/vision)")
    p.add_argument("--server", default=None, metavar="IP:PORT",
                   help="pin a specific game server via window.forceServer (live/vision)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hz", type=float, default=15.0, help="control loop rate (live pacing / sim dt)")
    p.add_argument(
        "--max-time",
        type=float,
        default=300.0,
        help="episode time cap in seconds; 0 = no cap, play until death (default 300)",
    )
    p.add_argument("--render", action="store_true", help="save trajectory/barrier plots")
    p.add_argument(
        "--viz",
        action="store_true",
        help="open a live side window showing the percept and planner internals",
    )
    p.add_argument(
        "--viz-every",
        type=int,
        default=2,
        metavar="N",
        help="update the --viz window every N control ticks (default 2)",
    )
    p.add_argument("--out", default="results", help="output directory for artifacts")
    p.add_argument("--boost", action="store_true", help="allow the planner to boost")
    p.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="K=V",
        help="planner parameter override (e.g. --param gamma=2.0), repeatable",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slither_bot", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run episodes with one planner")
    _add_common_run_args(p_run)
    p_run.add_argument("--planner", choices=["random", "apf", "cbf"], default="cbf")
    p_run.add_argument("--episodes", type=int, default=1)

    p_eval = sub.add_parser("eval", help="compare planners over N episodes each")
    _add_common_run_args(p_eval)
    p_eval.add_argument("--planners", default="random,apf,cbf", help="comma-separated planner names")
    p_eval.add_argument("--episodes", type=int, default=20, help="episodes per planner")

    p_probe = sub.add_parser("probe", help="verify slither.io JS globals and measure game constants (live)")
    p_probe.add_argument("--url", default="http://slither.io")
    p_probe.add_argument("--out", default="results", help="output directory for the probe report")
    p_probe.add_argument(
        "--fixture",
        default="tests/fixtures/live_dump.json",
        help="where to write the captured game-state fixture",
    )
    p_probe.add_argument("--play-timeout", type=float, default=30.0,
                         help="seconds to attempt auto-join before falling back")
    p_probe.add_argument("--manual-join-timeout", type=float, default=90.0,
                         help="seconds to wait for a manual Play click after auto-join fails")
    p_probe.add_argument("--server", default=None, metavar="IP:PORT",
                         help="pin a specific game server via window.forceServer")

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        from slither_bot.eval.runner import cmd_run

        return cmd_run(args)
    if args.command == "eval":
        from slither_bot.eval.harness import cmd_eval

        return cmd_eval(args)
    if args.command == "probe":
        from slither_bot.live.probe import cmd_probe

        return cmd_probe(args)
    raise AssertionError(f"unhandled command {args.command}")
