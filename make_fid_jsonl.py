"""Convert TensorBoard validation FID scalars into validation_fid.jsonl."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Create validation_fid.jsonl from TensorBoard event files"
    )
    parser.add_argument("target_dir", type=Path, help="training run directory")
    args = parser.parse_args()

    if not args.target_dir.is_dir():
        parser.error(f"target directory does not exist: {args.target_dir}")

    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError as error:
        parser.error(
            "TensorBoard is required; install it in the training environment"
        )
        raise error

    event_files = sorted(args.target_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        parser.error(f"no TensorBoard event files found under {args.target_dir}")

    # Keep the most recently recorded value if multiple event files contain
    # the same step, which can happen after resuming a run.
    values_by_step = {}
    for event_file in event_files:
        accumulator = event_accumulator.EventAccumulator(
            str(event_file),
            size_guidance={event_accumulator.SCALARS: 0},
        )
        accumulator.Reload()
        if "Validation/FID" not in accumulator.Tags().get("scalars", []):
            continue

        for event in accumulator.Scalars("Validation/FID"):
            current = values_by_step.get(event.step)
            if current is None or event.wall_time >= current.wall_time:
                values_by_step[event.step] = event

    if not values_by_step:
        parser.error(
            f'no "Validation/FID" scalar found in TensorBoard files under {args.target_dir}'
        )

    output_path = args.target_dir / "validation_fid.jsonl"
    with output_path.open("w", encoding="utf-8") as output:
        for step in sorted(values_by_step):
            event = values_by_step[step]
            output.write(json.dumps({"iteration": event.step, "fid": event.value}) + "\n")

    print(f"wrote {len(values_by_step)} FID records to {output_path}")


if __name__ == "__main__":
    main()
