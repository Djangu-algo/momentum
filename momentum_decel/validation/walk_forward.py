from __future__ import annotations

from collections.abc import Iterator


def blocked_time_series_splits(
    length: int,
    train_size: int,
    test_size: int,
    step: int | None = None,
) -> Iterator[tuple[range, range]]:
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive.")
    step = step or test_size
    start = 0
    while start + train_size + test_size <= length:
        train = range(start, start + train_size)
        test = range(start + train_size, start + train_size + test_size)
        yield train, test
        start += step

