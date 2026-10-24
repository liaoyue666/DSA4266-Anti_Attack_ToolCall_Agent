from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from tool_call_auditor.config import PilotConfig
from tool_call_auditor.data_export import export_dataset
from tool_call_auditor.workspace_runner import run, validate_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect and export AgentDojo tool-call and episode data."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the experiment matrix without creating files or calling an API.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export existing AgentDojo traces and audit logs without running an agent.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit existing output/export directories; existing exported files are replaced.",
    )
    args = parser.parse_args()

    config = PilotConfig.from_yaml(args.config)
    export_dir = config.export_dir or config.output_dir / "data"
    if args.dry_run:
        plan = validate_config(config, require_model=False)
        plan.update(
            {
                "output_dir": str(config.output_dir),
                "export_dir": str(export_dir),
                "api_calls": 0,
            }
        )
        print(json.dumps(plan, indent=2))
        return

    if args.export_only:
        paths = export_dataset(config.output_dir, export_dir, overwrite=args.overwrite)
        print(json.dumps({"exported_files": [str(path) for path in paths]}, indent=2))
        return

    load_dotenv(Path.cwd() / ".env")
    run_error: Exception | None = None
    summaries = None
    try:
        summaries = run(config, overwrite=args.overwrite)
    except Exception as error:
        run_error = error

    try:
        calls_path, episodes_path = export_dataset(
            config.output_dir,
            export_dir,
            overwrite=args.overwrite,
        )
    except Exception as export_error:
        if run_error is not None:
            raise export_error from run_error
        raise
    if run_error is not None:
        raise run_error

    print(
        json.dumps(
            {
                "summaries": summaries,
                "exported_files": [str(calls_path), str(episodes_path)],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
