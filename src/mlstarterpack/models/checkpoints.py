import os

from typing import Pattern


def find_latest_checkpoint(model_dir, checkpoint_re: Pattern):
    checkpoint_numbers = []
    for filename in os.listdir(model_dir):
        match = checkpoint_re.match(filename)
        if not match:
            continue

        sorting_key = (int(match.group(1)), match.group(0))
        checkpoint_numbers.append(sorting_key)

    if not checkpoint_numbers:
        raise FileNotFoundError()

    checkpoint_number, filename = max(checkpoint_numbers)
    return checkpoint_number, filename
