"""Command-line interface.

    cofoldrank select PREDICTIONS_DIR

Ranks a pool of predicted antibody-antigen structures and reports the best pose. `PREDICTIONS_DIR`
holds one subdirectory per predictor.
"""
from __future__ import annotations

import argparse

from cofoldrank import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cofoldrank",
        description="Predictor-agnostic structural selection for antibody-antigen co-folding.")
    parser.add_argument("--version", action="version", version=f"cofoldrank {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="{select}")

    p = sub.add_parser("select", help="rank predicted structures and report the best pose",
                       description="Rank predicted antibody-antigen structures by cross-pose "
                                   "agreement. Expects one subdirectory per predictor.")
    p.add_argument("directory", help="directory containing one subdirectory per predictor")
    p.add_argument("--checkpoint", default=None,
                   help="selector checkpoint (default: bundled, or $COFOLDRANK_SELECTOR)")
    p.add_argument("--antibody-chains", default="H,L",
                   help="comma-separated antibody chain IDs; all other chains are the antigen")
    p.add_argument("--workers", type=int, default=1, help="processes used to parse structures")
    p.add_argument("--top", type=int, default=5, help="how many ranked poses to print")
    p.add_argument("--max-per-source", type=int, default=None,
                   help="use at most N poses from each predictor")
    p.add_argument("--csv", default=None, help="write the full ranking to this CSV")
    return parser


def _cmd_select(args) -> int:
    from cofoldrank.select import PoolSelector

    selector = PoolSelector(checkpoint=args.checkpoint,
                            antibody_chains=tuple(c for c in args.antibody_chains.split(",") if c),
                            workers=args.workers)
    result = selector.rank(args.directory, max_per_source=args.max_per_source)
    counts = result.pool.counts()
    print(f"pool: {len(result.pool)} poses from {result.pool.n_sources} sources "
          f"({', '.join(f'{k}={v}' for k, v in counts.items())})")
    print(f"\nbest: {result.best.path}")
    print(f"      source={result.best.source}  score={result.best.score:.4f}")
    print(f"\ntop {min(args.top, len(result.candidates))}:")
    for c in result.top(args.top):
        print(f"  {c.rank:>3}. {c.score:+.4f}  {c.source:<18} {c.path.name}")
    if args.csv:
        print(f"\nwrote {result.to_csv(args.csv)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "select":
        return _cmd_select(args)
    parser.error(f"unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
