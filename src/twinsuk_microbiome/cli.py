"""Command-line entry points for synthetic and explicit authorized inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .runtime import PACKAGE_ROOT
from .workflow import prepare_genus_inputs, prepare_inputs, run_workflow


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run preserved microbiome analyses on synthetic or explicit prepared inputs.")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run the bundled 600-person wholly synthetic fixture")
    run = commands.add_parser("run", help="Run an explicitly supplied prepared input directory")
    run.add_argument("--input-dir", type=Path, required=True)
    for command in [demo, run]:
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--community-permutations", type=int, default=19 if command is demo else 0)
        command.add_argument("--twin-bootstrap", type=int, default=0)
        command.add_argument("--allow-heavy", action="store_true", help="Explicitly enable production permutation counts or repeated twin-model fits")
    prepare = commands.add_parser("prepare", help="Prepare private local tables using the original raw-source layout")
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True, help="Empty private folder outside both the raw source and public release")
    genus = commands.add_parser("prepare-genus", help="Export separate private count and metadata inputs for the native R scripts")
    genus.add_argument("--input-dir", type=Path, required=True)
    genus.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_inputs(args.data_root, args.output_dir)
            print("PRIVATE PREPARED INPUTS: participant-level tables must be excluded from public releases.", file=sys.stderr)
        elif args.command == "prepare-genus":
            result = prepare_genus_inputs(args.input_dir, args.output_dir)
            print("PRIVATE GENUS INPUTS: count and metadata tables remain participant-level after pseudonymization.", file=sys.stderr)
        else:
            input_dir = PACKAGE_ROOT / "example_data" if args.command == "demo" else args.input_dir
            result = run_workflow(input_dir, args.output_dir, community_permutations=args.community_permutations,
                                  twin_bootstrap=args.twin_bootstrap, allow_heavy=args.allow_heavy)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        parser.exit(2, f"Input or execution error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
