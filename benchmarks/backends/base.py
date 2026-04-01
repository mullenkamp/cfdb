from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BackendBase(ABC):
    """Abstract interface for benchmark backends."""

    name: str

    @abstractmethod
    def setup(self, base_dir: Path, data_config: dict):
        """Store base_dir and data_config for use by other methods."""

    @abstractmethod
    def write_full(self) -> Path:
        """Create dataset with lat/lon/time coords and temperature variable, write all data. Return file path."""

    @abstractmethod
    def write_append(self, file_path: Path):
        """Open existing dataset, append extra time steps and write corresponding data."""

    @abstractmethod
    def read_full(self, file_path: Path) -> np.ndarray:
        """Read entire temperature variable into memory."""

    @abstractmethod
    def read_slice_aligned(self, file_path: Path) -> np.ndarray:
        """Read a chunk-aligned contiguous slice."""

    @abstractmethod
    def read_slice_unaligned(self, file_path: Path) -> np.ndarray:
        """Read a slice crossing chunk boundaries."""

    @abstractmethod
    def iterate_chunks(self, file_path: Path) -> int:
        """Iterate over all storage chunks, return total elements processed."""

    @abstractmethod
    def iterate_rechunked(self, file_path: Path) -> int:
        """Iterate with rechunked shape (full spatial, 7 time steps), return total elements."""

    @abstractmethod
    def groupby_7day(self, file_path: Path) -> int:
        """Group by 7-day periods along time, compute mean per group. Return group count."""

    @abstractmethod
    def get_file_size(self, file_path: Path) -> int:
        """Return total bytes on disk."""

    @abstractmethod
    def cleanup(self, file_path: Path):
        """Remove the dataset file/directory."""
