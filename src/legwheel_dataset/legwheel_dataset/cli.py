"""Command-line entry points for the offline dataset pipeline."""

import argparse
from pathlib import Path

from legwheel_dataset.config import ProcessingConfig
from legwheel_dataset.progress import TerminalProgress, progress_range
from legwheel_dataset.processor import (
    extract_features,
    isolate_packets,
    validate_processing_configuration,
    validate_processing_configuration_with_preview,
)


def _stage1_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--processing-config", type=Path)
    parser.add_argument("--output-directory", type=Path)
    return parser


def isolate_packets_main(args=None) -> None:
    """Run only Stage 1 packet isolation."""
    parser = _stage1_parser("Isolate synchronized learning packets from MCAP.")
    options = parser.parse_args(args)
    config = ProcessingConfig.from_yaml(options.processing_config)
    output = isolate_packets(
        options.run_directory,
        config,
        options.experiment_config,
        options.output_directory,
    )
    print(f"Stage 1 dataset written to {output}")


def extract_features_main(args=None) -> None:
    """Run only Stage 2 feature extraction."""
    parser = argparse.ArgumentParser(
        description="Extract ROI, plane, and spectral features."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--processing-config", type=Path)
    options = parser.parse_args(args)
    config = (
        ProcessingConfig.from_yaml(options.processing_config)
        if options.processing_config
        else None
    )
    output = extract_features(options.dataset, config)
    print(f"Stage 2 features written to {output}")


def build_dataset_main(args=None) -> None:
    """Run Stage 1 followed immediately by Stage 2."""
    parser = _stage1_parser("Build the complete LegWheel ML dataset.")
    options = parser.parse_args(args)
    config = ProcessingConfig.from_yaml(options.processing_config)
    with TerminalProgress(options.run_directory.name) as progress:
        output = isolate_packets(
            options.run_directory,
            config,
            options.experiment_config,
            options.output_directory,
            progress=progress_range(progress.update, 0.0, 0.65),
        )
        extract_features(
            output, config,
            progress=progress_range(progress.update, 0.65, 0.99),
        )
    print(f"Complete dataset written to {output}")


def validate_config_main(args=None) -> None:
    """Explain a processing configuration without writing packet data."""
    import json

    parser = argparse.ArgumentParser(
        description=(
            "Validate LegWheel post-processing settings against one recorded run."
        )
    )
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--processing-config", type=Path)
    parser.add_argument(
        "--show-visualization",
        "--show-visualisation",
        action="store_true",
        help=(
            "Open a blocking midpoint RGB/depth track diagnostic after printing "
            "the JSON report."
        ),
    )
    options = parser.parse_args(args)
    config = ProcessingConfig.from_yaml(options.processing_config)
    if options.show_visualization:
        report, preview = validate_processing_configuration_with_preview(
            options.run_directory,
            config,
            options.experiment_config,
        )
    else:
        report = validate_processing_configuration(
            options.run_directory,
            config,
            options.experiment_config,
        )
        preview = None
    print(json.dumps(report, indent=2), flush=True)
    if options.show_visualization:
        if preview is None:
            reason = report["visualization_preview"]["reason"]
            _print_yellow_warning(
                f"Skipping visualization because {reason}."
            )
        else:
            from legwheel_dataset.validation_plot import show_validation_preview

            shown, reason = show_validation_preview(preview)
            if not shown:
                _print_yellow_warning(
                    f"Skipping visualization because {reason}."
                )


def _print_yellow_warning(message: str) -> None:
    """Write a conspicuous warning without contaminating JSON stdout."""
    import sys

    print(f"\033[33mWarning: {message}\033[0m", file=sys.stderr)
