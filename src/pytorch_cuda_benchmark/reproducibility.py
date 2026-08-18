"""Control random number generators used by benchmark runs."""

import random

import numpy as np
import torch

MAX_SEED = 2**32 - 1


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not 0 <= seed <= MAX_SEED:
        raise ValueError(f"'seed' must be between 0 and {MAX_SEED}.")


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""

    _validate_seed(seed)

    random.seed(seed)
    np.random.seed(seed)

    # PyTorch does not annotate the public function's seed parameter.
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]


def create_data_generator(seed: int) -> torch.Generator:
    """Create a seeded generator for deterministic DataLoader sampling."""

    _validate_seed(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return generator


def seed_data_worker(_worker_id: int) -> None:
    """Seed Python and NumPy within a DataLoader worker process."""

    worker_seed = torch.initial_seed() % (MAX_SEED + 1)

    random.seed(worker_seed)
    np.random.seed(worker_seed)
