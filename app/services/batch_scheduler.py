from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


@dataclass
class BatchScheduler:
    batch_size: int = 4

    def split(self, items: Iterable[T]) -> Iterator[list[T]]:
        batch: list[T] = []
        for item in items:
            batch.append(item)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
