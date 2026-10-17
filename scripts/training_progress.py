"""Display resumable-training progress without deserializing model tensors."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from utils.training_checkpoint import read_training_checkpoint_metadata
from utils.training_sessions import session_plan


def progress(checkpoint, completed_epoch=None):
    checkpoint = Path(checkpoint).resolve()
    if completed_epoch is None:
        if checkpoint.exists():
            metadata = read_training_checkpoint_metadata(checkpoint)
            completed_epoch = int(metadata["epoch"])
            best_dice = float(metadata["best_dice"])
            total_train_time = float(metadata["total_train_time"])
            timestamp = datetime.fromtimestamp(checkpoint.stat().st_mtime).astimezone().isoformat()
            exists = True
            simulated = False
        else:
            metadata_path = checkpoint.with_name(checkpoint.name + ".metadata.json")
            if metadata_path.exists():
                raise FileNotFoundError(
                    f"Metadata exists but checkpoint is missing: {metadata_path}"
                )
            completed_epoch = 0
            best_dice = None
            total_train_time = 0.0
            timestamp = None
            exists = False
            simulated = False
    else:
        best_dice = None
        total_train_time = 0.0
        timestamp = None
        exists = checkpoint.exists()
        simulated = True

    result = session_plan(completed_epoch)
    result.update(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_exists": exists,
            "checkpoint_timestamp": timestamp,
            "best_dice": best_dice,
            "total_train_time": total_train_time,
            "simulated": simulated,
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--completed-epoch", type=int)
    parser.add_argument("--expect-completed-epoch", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = progress(args.checkpoint, args.completed_epoch)
    if (
        args.expect_completed_epoch is not None
        and result["completed_epoch"] != args.expect_completed_epoch
    ):
        raise RuntimeError(
            f"Expected completed epoch {args.expect_completed_epoch}, "
            f"found {result['completed_epoch']}"
        )

    if args.json:
        print(json.dumps(result, sort_keys=True))
        return

    mode = "SIMULATED" if result["simulated"] else "CHECKPOINT"
    print(f"Mode: {mode}")
    print(f"Checkpoint: {result['checkpoint']}")
    print(f"Last completed epoch: {result['completed_epoch']}")
    print(f"Best validation Dice: {result['best_dice']}")
    print(f"Current/last session: {result['current_session']}")
    print(f"Remaining epochs: {result['remaining_epochs']}")
    print(f"Checkpoint timestamp: {result['checkpoint_timestamp']}")
    print(f"Experiment completed: {result['completed']}")
    if not result["completed"]:
        print(
            "Next session: {session_number} (epochs {start_epoch}-{end_epoch}, "
            "--epoch {exclusive_epoch_limit})".format(**result)
        )


if __name__ == "__main__":
    main()
