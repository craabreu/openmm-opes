"""Multi-walker bias sharing through a shared directory.

Pure NumPy. This module must not import openmm, and deliberately knows
nothing about kernels: it moves flat dicts of arrays so it stays testable
without a simulation.

The format is ``np.savez``, not pickle. A walker reads files written by other
processes out of a shared directory, and ``pickle.load`` on data from another
process would be a remote-code-execution primitive. OpenMM's own Metadynamics
writes .npy for the same reason.
"""

from __future__ import annotations

import os
import re
import warnings

import numpy as np

FILENAME_PATTERN = re.compile(r"kde_(\d+)_(\d+)\.npz")


class _LoadedBias:
    """A peer's most recently seen state. Mirrors Metadynamics' own namedtuple."""

    __slots__ = ("index", "state", "walkerId")

    def __init__(self, walkerId: int, index: int, state: dict):
        self.walkerId = walkerId
        self.index = index
        self.state = state


class BiasSharer:
    """Reads and writes walker bias states in a shared directory.

    Parameters
    ----------
    biasDir
        Directory shared by every walker.
    walkerId
        This walker's identifier. Drawn at random when omitted, from a fresh
        generator so the global NumPy random state is left undisturbed. Pass a
        stable value across restarts so a resumed walker reclaims its own file
        slot: with a fresh random id every run, the previous run's file is
        never overwritten and is read back forever as a phantom extra peer,
        double-counting that walker's pre-restart kernels.
    """

    def __init__(self, biasDir: str, walkerId: int | None = None):
        self.biasDir = biasDir
        self.walkerId = (
            int(np.random.default_rng().integers(0x7FFFFFFF))
            if walkerId is None
            else walkerId
        )
        self._loaded: dict[int, _LoadedBias] = {}
        # Resume past whatever this walker left behind, so save() overwrites
        # its own previous file instead of orphaning it.
        self._saveIndex = self._highestOwnIndex()

    def _highestOwnIndex(self) -> int:
        """Highest save index this walker already has on disk, or 0."""
        if not os.path.isdir(self.biasDir):
            return 0
        indices = [
            int(match.group(2))
            for match in map(FILENAME_PATTERN.match, os.listdir(self.biasDir))
            if match is not None and int(match.group(1)) == self.walkerId
        ]
        return max(indices, default=0)

    def _path(self, prefix: str, index: int) -> str:
        return os.path.join(self.biasDir, f"{prefix}_{self.walkerId}_{index}.npz")

    def save(self, state: dict) -> None:
        """Write this walker's state atomically and drop the previous index."""
        oldName = self._path("kde", self._saveIndex)
        self._saveIndex += 1
        tempName = self._path("temp", self._saveIndex)
        fileName = self._path("kde", self._saveIndex)
        np.savez(tempName, **state)
        os.rename(tempName, fileName)
        if os.path.exists(oldName):
            os.remove(oldName)

    def load(self) -> dict[int, dict]:
        """Return peer states that are new or have advanced since the last call."""
        updated: dict[int, dict] = {}
        for filename in os.listdir(self.biasDir):
            match = FILENAME_PATTERN.match(filename)
            if match is None:
                continue
            walkerId, index = int(match.group(1)), int(match.group(2))
            if walkerId == self.walkerId:
                continue
            known = self._loaded.get(walkerId)
            if known is not None and index <= known.index:
                continue
            try:
                with np.load(
                    os.path.join(self.biasDir, filename), allow_pickle=False
                ) as data:
                    state = {key: data[key] for key in data.files}
            except (OSError, ValueError):
                warnings.warn(
                    f"The file {filename} seems to have been deleted. Using the "
                    "latest loaded data from the same walker.",
                    stacklevel=2,
                )
                continue
            self._loaded[walkerId] = _LoadedBias(walkerId, index, state)
            updated[walkerId] = state
        return updated

    def getLoadedStates(self) -> dict[int, dict]:
        """Every peer state seen so far, whether or not it changed recently."""
        return {walkerId: bias.state for walkerId, bias in self._loaded.items()}
