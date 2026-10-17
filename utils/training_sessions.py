"""Pure scheduling logic for fixed five-epoch training sessions."""

TOTAL_EPOCHS = 199
SESSION_SIZE = 5
TOTAL_SESSIONS = 40


def session_plan(completed_epoch):
    if isinstance(completed_epoch, bool) or not isinstance(completed_epoch, int):
        raise TypeError("completed_epoch must be an integer")
    if completed_epoch < 0 or completed_epoch > TOTAL_EPOCHS:
        raise ValueError(f"completed_epoch must be between 0 and {TOTAL_EPOCHS}")

    remaining = TOTAL_EPOCHS - completed_epoch
    if remaining == 0:
        return {
            "completed": True,
            "completed_epoch": completed_epoch,
            "current_session": TOTAL_SESSIONS,
            "remaining_epochs": 0,
            "session_number": None,
            "start_epoch": None,
            "end_epoch": None,
            "exclusive_epoch_limit": None,
        }

    next_epoch = completed_epoch + 1
    session_number = (next_epoch - 1) // SESSION_SIZE + 1
    end_epoch = min(session_number * SESSION_SIZE, TOTAL_EPOCHS)
    current_session = 0 if completed_epoch == 0 else (completed_epoch - 1) // SESSION_SIZE + 1
    return {
        "completed": False,
        "completed_epoch": completed_epoch,
        "current_session": current_session,
        "remaining_epochs": remaining,
        "session_number": session_number,
        "start_epoch": next_epoch,
        "end_epoch": end_epoch,
        "exclusive_epoch_limit": end_epoch + 1,
    }


def all_session_bounds():
    return [
        {
            "session_number": number,
            "start_epoch": (number - 1) * SESSION_SIZE + 1,
            "end_epoch": min(number * SESSION_SIZE, TOTAL_EPOCHS),
            "exclusive_epoch_limit": min(number * SESSION_SIZE, TOTAL_EPOCHS) + 1,
        }
        for number in range(1, TOTAL_SESSIONS + 1)
    ]
