# openmm-opes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the OPES research script in `opes-simulations` into a packaged, tested, documented OpenMM add-on library called `openmm-opes`.

**Architecture:** Three modules with a strict dependency gradient. `kde.py` is pure NumPy/SciPy (CV space, kernels, online KDE) and never imports OpenMM, which is what lets the bulk of the test suite run anywhere. `io.py` is pure NumPy and owns only the multi-walker file protocol, trading in flat dicts of arrays so it knows nothing about kernels. `opes.py` sits on top and is the only module that touches OpenMM, integrating the bias as a `CustomCVForce` with a tabulated function, closely mirroring `openmm.app.Metadynamics`.

**Tech Stack:** Python 3.11+, NumPy, SciPy, OpenMM 8.1+ (conda-forge), setuptools, pytest, ruff, ty, mkdocs-material + mkdocstrings.

**Spec:** `docs/superpowers/specs/2026-09-22-openmm-opes-design.md` — read it alongside this plan. Every task below argues from it; section references like §7.4 point into it.

## Global Constraints

- Python floor is **3.11**. `target-version = "py311"`.
- Public API uses **camelCase** (`addKernel`, `getFreeEnergy`, `varianceFrequency`) to match OpenMM's conventions. Ruff's `N` rules are deliberately not enabled.
- Distribution name `openmm-opes`; import name `openmm_opes`. `src/` layout.
- Ruff: `line-length = 88`, lint select `["E", "F", "I", "UP", "B", "C4", "RUF"]`.
- Runtime deps exactly: `numpy>=1.24`, `scipy>=1.10`, `openmm>=8.1`.
- `kde.py` and `io.py` **must not import openmm**, directly or transitively. A test enforces this.
- Default behavior must be numerically identical to the source implementation. Task 6's parity test is the guard. Never loosen its tolerance to make it pass.
- CVs are `openmm.app.BiasVariable` instances. No new CV abstraction.
- Every kernel-shape / compression option defaults to the source's current value.
- License MIT, already present. Version `0.1.0`.

## Plan-level decisions that refine the spec

Two points the spec left implicit. Both follow from §6's switch away from pickle.

1. **`RunningAverage` lives in `opes.py`** (as §5 says), and `io.py` does *not* import it. `io.py` deals only in `dict[str, np.ndarray | float]`, so it stays OpenMM-free and generic. Consequence: the §7.4.1 aliasing regression test lives in `tests/test_opes.py`, not `tests/test_io.py` as §8.1 suggested.
2. **State is exchanged via explicit flat `getState()` / `setState()`**, not `__getstate__`/`__setstate__`. The spec listed the pickle-protocol methods because the format was pickle; §6 supersedes that. `CVSpace` is never serialized — all walkers share identical CV definitions, so a loading walker constructs `OnlineKDE(self._cvSpace)` and calls `setState()`. `__copy__` remains explicit.

---

### Task 1: Package skeleton, packaging metadata, and lint CI

**Files:**
- Create: `pyproject.toml`
- Create: `src/openmm_opes/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_packaging.py`
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: nothing.
- Produces: an installable `openmm_opes` package exposing `__version__: str`; a CI `lint` job later tasks extend with a `test` job.

- [ ] **Step 1: Write the failing test**

`tests/test_packaging.py`:

```python
import importlib
import subprocess
import sys


def test_package_imports_and_has_version():
    mod = importlib.import_module("openmm_opes")
    assert isinstance(mod.__version__, str)
    assert mod.__version__ == "0.1.0"


def test_kde_and_io_do_not_import_openmm():
    """kde.py and io.py must stay OpenMM-free so most tests run anywhere.

    Runs in a subprocess because openmm may already be imported in-process.
    """
    code = (
        "import sys;"
        "import openmm_opes.kde, openmm_opes.io;"
        "assert 'openmm' not in sys.modules, sorted("
        "m for m in sys.modules if m.startswith('openmm') "
        "and not m.startswith('openmm_opes'))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_packaging.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'openmm_opes'`

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[project]
name = "openmm-opes"
version = "0.1.0"
description = "OpenMM implementation of On-the-fly Probability Enhanced Sampling (OPES)"
readme = "README.md"
license = { file = "LICENSE" }
authors = [{ name = "Charlles Abreu" }]
requires-python = ">=3.11"
dependencies = ["numpy>=1.24", "scipy>=1.10", "openmm>=8.1"]

[project.optional-dependencies]
dev = ["pytest>=7.4", "ruff>=0.16", "ty==0.0.72"]
# mkdocs >= 1.6 is required for the exclude_docs setting used by mkdocs.yml
docs = ["mkdocs>=1.6", "mkdocs-material>=9.5", "mkdocstrings[python]>=0.24"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 88
target-version = "py311"
include = ["src/**/*.py", "tests/**/*.py"]

[tool.ruff.lint]
# Deliberately no "N" rules: the public API is camelCase to match OpenMM.
select = ["E", "F", "I", "UP", "B", "C4", "RUF"]

[tool.ty.analysis]
# OpenMM ships no type stubs, and the lint CI job installs it without conda.
allowed-unresolved-imports = ["openmm", "openmm.**"]
```

- [ ] **Step 4: Write `src/openmm_opes/__init__.py`**

```python
"""OpenMM implementation of On-the-fly Probability Enhanced Sampling (OPES)."""

__version__ = "0.1.0"

__all__ = ["__version__"]
```

Create empty `tests/__init__.py`. Create placeholder modules so the import test can run — they are filled in by later tasks:

```bash
: > src/openmm_opes/kde.py
: > src/openmm_opes/io.py
```

- [ ] **Step 5: Write the lint CI workflow**

`.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      # Deliberately does NOT install the package: openmm's canonical channel
      # is conda-forge. ruff needs nothing installed, and ty needs only numpy
      # and scipy resolvable, with openmm covered by allowed-unresolved-imports.
      - run: pip install ruff ty numpy scipy
      - run: ruff check src tests
      - run: ruff format --check src tests
      - run: ty check src tests
```

- [ ] **Step 6: Install and run the tests**

Run:
```bash
pip install -e ".[dev]"
pytest tests/test_packaging.py -v
```
Expected: PASS (2 tests)

- [ ] **Step 7: Verify lint passes**

Run: `ruff check src tests && ruff format --check src tests && ty check src tests`
Expected: all clean. Run `ruff format src tests` first if the format check complains.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src tests .github
git commit -m "feat: package skeleton, metadata, and lint CI"
```

---

### Task 2: `CVSpace`

**Files:**
- Modify: `src/openmm_opes/kde.py`
- Create: `tests/helpers.py`
- Create: `tests/test_cvspace.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `CVSpace(variables, bounded: bool = False)` where `variables` is any sequence of objects with `minValue: float`, `maxValue: float`, `gridWidth: int`, `periodic: bool`.
  - Properties `gridShape -> tuple[int, ...]` (reversed widths), `numDimensions -> int`.
  - Methods `displacement(position, endpoint) -> np.ndarray`, `endpoint(position, displacement) -> np.ndarray`, `gridDistances(position) -> list[np.ndarray]`, `closestNode(position) -> tuple[int, ...]`, `foldedGrid(values) -> np.ndarray`.
  - `CVSpace.CV` — a namedtuple `(minValue, maxValue, gridWidth, periodic)` used by tests to build spaces without OpenMM.

- [ ] **Step 1: Write `tests/helpers.py`**

```python
"""Test helpers that build CV spaces without requiring OpenMM."""

from openmm_opes.kde import CVSpace


def makeSpace(*specs, bounded=False):
    """Build a CVSpace from (minValue, maxValue, gridWidth, periodic) tuples."""
    return CVSpace([CVSpace.CV(*spec) for spec in specs], bounded=bounded)
```

- [ ] **Step 2: Write the failing tests**

`tests/test_cvspace.py`:

```python
import numpy as np
import pytest

from tests.helpers import makeSpace


def test_grid_shape_is_reversed_widths():
    space = makeSpace((-1.0, 1.0, 11, False), (-2.0, 2.0, 21, False))
    assert space.numDimensions == 2
    assert space.gridShape == (21, 11)


def test_displacement_is_plain_difference_when_not_periodic():
    space = makeSpace((-5.0, 5.0, 11, False))
    disp = space.displacement(np.array([1.0]), np.array([4.0]))
    assert disp == pytest.approx([3.0])


def test_displacement_uses_minimum_image_across_the_periodic_seam():
    # domain length 2*pi; going from 3.0 to -3.0 is +0.2832, not -6.0
    space = makeSpace((-np.pi, np.pi, 11, True))
    disp = space.displacement(np.array([3.0]), np.array([-3.0]))
    assert disp == pytest.approx([2 * np.pi - 6.0])


def test_endpoint_wraps_back_into_the_periodic_domain():
    space = makeSpace((-np.pi, np.pi, 11, True))
    end = space.endpoint(np.array([3.0]), np.array([1.0]))
    assert -np.pi <= end[0] <= np.pi
    assert end == pytest.approx([4.0 - 2 * np.pi])


def test_displacement_and_endpoint_round_trip():
    space = makeSpace((-np.pi, np.pi, 11, True), (-1.0, 1.0, 11, False))
    a = np.array([2.0, -0.3])
    b = np.array([-2.5, 0.8])
    assert space.endpoint(a, space.displacement(a, b)) == pytest.approx(b)


def test_closest_node_clamps_outside_the_domain():
    space = makeSpace((0.0, 1.0, 11, False))
    assert space.closestNode(np.array([0.5])) == (5,)
    assert space.closestNode(np.array([-3.0])) == (0,)
    assert space.closestNode(np.array([9.0])) == (10,)


def test_closest_node_wraps_for_periodic_cvs():
    space = makeSpace((0.0, 1.0, 11, True))
    # 1.0 is identified with 0.0 for a periodic CV
    assert space.closestNode(np.array([1.0])) == (0,)


def test_grid_distances_identify_the_periodic_endpoints():
    space = makeSpace((0.0, 1.0, 11, True))
    dist = space.gridDistances(np.array([0.25]))[0]
    assert dist[-1] == pytest.approx(dist[0])
    assert np.all(np.abs(dist) <= 0.5 + 1e-12)


def test_bounded_grid_triples_then_folds_back():
    space = makeSpace((0.0, 1.0, 11, False), bounded=True)
    values = np.zeros(33)
    folded = space.foldedGrid(values)
    assert folded.shape == (11,)
    # three contributions of log(1) each fold to log(3)
    assert folded == pytest.approx(np.full(11, np.log(3.0)))


def test_folded_grid_is_identity_when_not_bounded():
    space = makeSpace((0.0, 1.0, 11, False))
    values = np.arange(11.0)
    assert space.foldedGrid(values) == pytest.approx(values)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_cvspace.py -v`
Expected: FAIL — `ImportError: cannot import name 'CVSpace'`

- [ ] **Step 4: Implement `CVSpace` in `src/openmm_opes/kde.py`**

```python
"""Online kernel density estimation for OPES.

Pure NumPy/SciPy. This module must not import openmm.
"""

from __future__ import annotations

import functools
from collections import namedtuple
from copy import copy

import numpy as np


class CVSpace:
    """The domain of the collective variables: grid, periodicity, boundaries.

    Parameters
    ----------
    variables
        Sequence of collective variable descriptions. Each must expose
        ``minValue``, ``maxValue``, ``gridWidth`` and ``periodic``; an
        ``openmm.app.BiasVariable`` qualifies.
    bounded
        Whether non-periodic variables have reflective boundaries. When true the
        internal grid is tripled with mirrored copies on each side, and
        :meth:`foldedGrid` folds them back.
    """

    CV = namedtuple("CV", ["minValue", "maxValue", "gridWidth", "periodic"])

    def __init__(self, variables, bounded: bool = False):
        self.variables = [
            self.CV(cv.minValue, cv.maxValue, cv.gridWidth, cv.periodic)
            for cv in variables
        ]
        self.bounded = bounded
        self._periodic = any(cv.periodic for cv in self.variables)
        self._grid = []
        for cv in self.variables:
            a, b, n = cv.minValue, cv.maxValue, cv.gridWidth
            points = np.linspace(a, b, n)
            if bounded and not cv.periodic:
                left = np.linspace(2 * a - b, a, n)
                right = np.linspace(b, 2 * b - a, n)
                left[-1] = right[0] = np.inf
                points = np.concatenate((points, np.flip(left), np.flip(right)))
            self._grid.append(points)
        self._widths = np.array([cv.gridWidth for cv in self.variables])
        self._lbounds = np.array([cv.minValue for cv in self.variables])
        self._lengths = np.array([cv.maxValue for cv in self.variables]) - self._lbounds
        if self._periodic:
            self._pdims = tuple(
                i for i, cv in enumerate(self.variables) if cv.periodic
            )
            self._plbounds = self._lbounds[list(self._pdims)]
            self._plengths = self._lengths[list(self._pdims)]

    @property
    def gridShape(self) -> tuple[int, ...]:
        """Shape of the CV-space grid, slowest axis first."""
        return tuple(int(width) for width in reversed(self._widths))

    @property
    def numDimensions(self) -> int:
        """Number of collective variables."""
        return len(self.variables)

    def displacement(self, position, endpoint):
        """Displacement from ``position`` to ``endpoint``, minimum-image if periodic."""
        disp = endpoint - position
        if self._periodic:
            disp[..., self._pdims] -= self._plengths * np.rint(
                disp[..., self._pdims] / self._plengths
            )
        return disp

    def endpoint(self, position, displacement):
        """Endpoint reached from ``position``, wrapped into the periodic domain."""
        end = position + displacement
        if self._periodic:
            end[..., self._pdims] = (
                self._plbounds
                + (end[..., self._pdims] - self._plbounds) % self._plengths
            )
        return end

    def gridDistances(self, position):
        """Per-axis distances from ``position`` to every grid node."""
        distances = [centers - x for centers, x in zip(self._grid, position)]
        if self._periodic:
            for dim, length in zip(self._pdims, self._plengths):
                distances[dim] -= length * np.rint(distances[dim] / length)
                # a periodic grid's last node is the same point as its first
                distances[dim][-1] = distances[dim][0]
        return distances

    def closestNode(self, position) -> tuple[int, ...]:
        """Index of the grid node nearest to ``position``, in gridShape order."""
        indices = np.rint(
            (self._widths - 1) * (position - self._lbounds) / self._lengths
        ).astype(int)
        if self._periodic:
            indices[list(self._pdims)] %= self._widths[list(self._pdims)]
        indices = np.clip(indices, 0, self._widths - 1)
        return tuple(reversed(indices))

    def foldedGrid(self, values):
        """Fold a tripled bounded grid back onto the physical domain, in log space."""
        if self.bounded:
            for i, cv in enumerate(reversed(self.variables)):
                if not cv.periodic:
                    values, left, right = np.array_split(values, 3, axis=i)
                    values = np.logaddexp(left, np.logaddexp(values, right))
        return values
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_cvspace.py -v`
Expected: PASS (10 tests)

- [ ] **Step 6: Lint and commit**

```bash
ruff format src tests && ruff check src tests && ty check src tests
git add src/openmm_opes/kde.py tests/helpers.py tests/test_cvspace.py
git commit -m "feat: add CVSpace with periodic and reflective boundary handling"
```

---

### Task 3: Kernel shapes and `Kernel`

**Files:**
- Modify: `src/openmm_opes/kde.py`
- Create: `tests/test_kernel.py`

**Interfaces:**
- Consumes: `CVSpace` from Task 2.
- Produces:
  - `GAUSSIAN` and `COMPACT` module-level shape objects, each with `name: str`, `logNorm: float`, `exponents(x) -> np.ndarray`.
  - `KERNEL_SHAPES: dict[str, KernelShape]` mapping `"gaussian"`/`"compact"`.
  - `Kernel(cvSpace, position, bandwidth, logWeight, numSamples=1, shape=GAUSSIAN)` with attributes `position`, `bandwidth`, `logWeight`, `numSamples`, `shape`, `logHeight`; methods `findNearest(centers, bandwidths, ignore=()) -> tuple[int, float]`, `merge(other) -> None`, `evaluate(points) -> np.ndarray`, `evaluateOnGrid() -> np.ndarray`, `__copy__`.

- [ ] **Step 1: Write the failing tests**

`tests/test_kernel.py`:

```python
import numpy as np
import pytest
from scipy import integrate

from openmm_opes.kde import COMPACT, GAUSSIAN, Kernel
from tests.helpers import makeSpace


def test_compact_shape_normalization_constant_is_exact():
    value, _ = integrate.quad(lambda x: (9 - x * x) ** 4, -3, 3)
    assert np.exp(COMPACT.logNorm) == pytest.approx(value)
    assert np.exp(COMPACT.logNorm) == pytest.approx(559872 / 35)


def test_gaussian_kernel_integrates_to_its_weight():
    space = makeSpace((-10.0, 10.0, 2001, False))
    kernel = Kernel(space, [0.3], [0.7], np.log(2.5))
    xs = np.linspace(-10, 10, 200001)
    values = np.exp(kernel.evaluate(xs.reshape(-1, 1)))
    # Riemann sum rather than np.trapezoid, which needs numpy >= 2.0
    mass = values.sum() * (xs[1] - xs[0])
    assert mass == pytest.approx(2.5, rel=1e-5)


def test_compact_kernel_integrates_to_its_weight():
    space = makeSpace((-10.0, 10.0, 2001, False))
    kernel = Kernel(space, [0.0], [0.7], np.log(2.5), shape=COMPACT)
    xs = np.linspace(-10, 10, 200001)
    values = np.exp(kernel.evaluate(xs.reshape(-1, 1)))
    mass = values.sum() * (xs[1] - xs[0])
    assert mass == pytest.approx(2.5, rel=1e-5)


def test_compact_kernel_has_support_of_three_bandwidths():
    space = makeSpace((-10.0, 10.0, 201, False))
    kernel = Kernel(space, [0.0], [0.5], 0.0, shape=COMPACT)
    assert np.isfinite(kernel.evaluate(np.array([[1.49]])))[0]
    assert np.isneginf(kernel.evaluate(np.array([[1.51]])))[0]


def test_zero_bandwidth_gives_a_null_kernel():
    space = makeSpace((-1.0, 1.0, 11, False))
    assert np.isneginf(Kernel(space, [0.0], [0.0], 0.0).logHeight)


def test_find_nearest_returns_index_and_squared_mahalanobis_distance():
    space = makeSpace((-10.0, 10.0, 101, False))
    kernel = Kernel(space, [0.0], [2.0], 0.0)
    centers = np.array([[6.0], [1.0], [-8.0]])
    bandwidths = np.array([[2.0], [2.0], [2.0]])
    index, sqDist = kernel.findNearest(centers, bandwidths)
    assert index == 1
    assert sqDist == pytest.approx(0.25)


def test_find_nearest_honours_ignored_indices():
    space = makeSpace((-10.0, 10.0, 101, False))
    kernel = Kernel(space, [0.0], [2.0], 0.0)
    centers = np.array([[6.0], [1.0], [-8.0]])
    bandwidths = np.full((3, 1), 2.0)
    index, _ = kernel.findNearest(centers, bandwidths, ignore=[1])
    assert index == 0


def test_find_nearest_on_empty_center_list():
    space = makeSpace((-10.0, 10.0, 101, False))
    index, sqDist = Kernel(space, [0.0], [1.0], 0.0).findNearest(
        np.empty((0, 1)), np.empty((0, 1))
    )
    assert index == -1
    assert np.isinf(sqDist)


def test_merge_preserves_weight_mean_and_second_moment():
    space = makeSpace((-10.0, 10.0, 501, False))
    merged = Kernel(space, [0.0], [1.0], np.log(3.0))
    other = Kernel(space, [2.0], [0.5], np.log(1.0))
    merged.merge(other)

    w1, w2 = 3.0, 1.0
    total = w1 + w2
    meanRef = (w1 * 0.0 + w2 * 2.0) / total
    varRef = (w1 * 1.0**2 + w2 * 0.5**2) / total + (w1 * w2 / total**2) * 2.0**2

    assert np.exp(merged.logWeight) == pytest.approx(total)
    assert merged.position[0] == pytest.approx(meanRef)
    assert merged.bandwidth[0] ** 2 == pytest.approx(varRef)
    assert merged.numSamples == 2


def test_evaluate_on_grid_matches_direct_evaluation():
    space = makeSpace((-4.0, 4.0, 41, False), (-4.0, 4.0, 31, False))
    kernel = Kernel(space, [0.5, -1.0], [0.8, 1.2], np.log(2.0))
    onGrid = kernel.evaluateOnGrid()
    assert onGrid.shape == space.gridShape

    xs = np.linspace(-4, 4, 41)
    ys = np.linspace(-4, 4, 31)
    points = np.stack(np.meshgrid(xs, ys), axis=-1)
    assert onGrid == pytest.approx(kernel.evaluate(points))


def test_copy_does_not_share_arrays():
    from copy import copy

    space = makeSpace((-4.0, 4.0, 41, False))
    kernel = Kernel(space, [0.5], [0.8], 0.0)
    duplicate = copy(kernel)
    duplicate.position[0] = 99.0
    assert kernel.position[0] == pytest.approx(0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_kernel.py -v`
Expected: FAIL — `ImportError: cannot import name 'COMPACT'`

- [ ] **Step 3: Implement shapes and `Kernel`, appended to `src/openmm_opes/kde.py`**

```python
class KernelShape:
    """A kernel profile: its per-dimension log normalization and its exponent."""

    def __init__(self, name: str, logNorm: float, exponents):
        self.name = name
        self.logNorm = logNorm
        self._exponents = exponents

    def exponents(self, x):
        """Log of the kernel profile at scaled distances ``x``."""
        return self._exponents(x)


def _gaussianExponents(x):
    return -0.5 * x**2


def _compactExponents(x):
    values = 9 - x**2
    mask = values > 0
    result = np.empty_like(values)
    result[mask] = 4 * np.log(values[mask])
    result[~mask] = -np.inf
    return result


#: Unbounded Gaussian profile; the default and the one both papers use.
GAUSSIAN = KernelShape("gaussian", np.log(2 * np.pi) / 2, _gaussianExponents)

#: Compact quartic profile with support of +/- 3 bandwidths. Its normalization
#: constant is exactly the integral of (9 - x**2)**4 over [-3, 3].
COMPACT = KernelShape("compact", np.log(559872 / 35), _compactExponents)

KERNEL_SHAPES = {shape.name: shape for shape in (GAUSSIAN, COMPACT)}


class Kernel:
    """A multivariate kernel with diagonal covariance."""

    def __init__(
        self, cvSpace, position, bandwidth, logWeight, numSamples=1, shape=GAUSSIAN
    ):
        self.cvSpace = cvSpace
        self.position = np.array(position, dtype=float)
        self.bandwidth = np.array(bandwidth, dtype=float)
        self.logWeight = logWeight
        self.numSamples = numSamples
        self.shape = shape
        self.logHeight = self._computeLogHeight()

    def __copy__(self):
        return Kernel(
            self.cvSpace,
            self.position,
            self.bandwidth,
            self.logWeight,
            self.numSamples,
            self.shape,
        )

    def _computeLogHeight(self):
        if np.any(self.bandwidth == 0):
            return -np.inf
        d = self.cvSpace.numDimensions
        return self.logWeight - d * self.shape.logNorm - np.sum(np.log(self.bandwidth))

    def _scaledDistances(self, points, bandwidths):
        return self.cvSpace.displacement(self.position, points) / bandwidths

    def findNearest(self, centers, bandwidths, ignore=()):
        """Index of and squared Mahalanobis distance to the nearest center."""
        if centers.size == 0:
            return -1, np.inf
        sqDistances = np.sum(self._scaledDistances(centers, bandwidths) ** 2, axis=-1)
        if len(ignore):
            sqDistances[list(ignore)] = np.inf
        index = int(np.argmin(sqDistances))
        return index, sqDistances[index]

    def merge(self, other) -> None:
        """Absorb ``other``, preserving total weight, mean and second moment."""
        logSumWeights = np.logaddexp(self.logWeight, other.logWeight)
        w1 = np.exp(self.logWeight - logSumWeights)
        w2 = np.exp(other.logWeight - logSumWeights)
        disp = self.cvSpace.displacement(self.position, other.position)
        self.position = self.cvSpace.endpoint(self.position, w2 * disp)
        self.bandwidth = np.sqrt(
            w1 * self.bandwidth**2 + w2 * other.bandwidth**2 + w1 * w2 * disp**2
        )
        self.logWeight = logSumWeights
        self.numSamples += other.numSamples
        self.logHeight = self._computeLogHeight()

    def evaluate(self, points):
        """Log of the kernel at the given point or points."""
        return self.logHeight + np.sum(
            self.shape.exponents(self._scaledDistances(points, self.bandwidth)), axis=-1
        )

    def evaluateOnGrid(self):
        """Log of the kernel on the CV-space grid, in gridShape order."""
        distances = self.cvSpace.gridDistances(self.position)
        exponents = [
            self.shape.exponents(dist / sigma)
            for dist, sigma in zip(distances, self.bandwidth)
        ]
        return self.cvSpace.foldedGrid(
            self.logHeight + functools.reduce(np.add.outer, reversed(exponents))
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_kernel.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Lint and commit**

```bash
ruff format src tests && ruff check src tests && ty check src tests
git add src/openmm_opes/kde.py tests/test_kernel.py
git commit -m "feat: add kernel shapes and moment-preserving Kernel merge"
```

---

### Task 4: `OnlineKDE`

**Files:**
- Modify: `src/openmm_opes/kde.py`
- Create: `tests/test_kde.py`

**Interfaces:**
- Consumes: `CVSpace`, `Kernel`, `KERNEL_SHAPES` from Tasks 2–3.
- Produces:
  - `OnlineKDE(cvSpace, compressionThreshold=1.0, useExistingBandwidths=True, kernelShape="gaussian")`.
  - `update(position, logWeight, variance) -> None`, `evaluate(point) -> float`, `getLogPDF() -> np.ndarray`, `getLogMeanDensity() -> float`, `getNumKernels() -> int`, `__iadd__`, `__copy__`, `__bool__`.
  - `getState() -> dict[str, np.ndarray | float]` and `setState(state) -> None` arrive in Task 5.

Fixes §7.4.2 (`__bool__` referencing a nonexistent attribute). Note `_removeKernels` carries the ordering comment required by §12.4.

- [ ] **Step 1: Write the failing tests**

`tests/test_kde.py`:

```python
import numpy as np
import pytest

from openmm_opes.kde import OnlineKDE
from tests.helpers import makeSpace


def directLogPDF(kde, grid):
    """Mixture density on a 1-D grid, computed straight from the kernel list."""
    total = np.zeros_like(grid)
    for kernel in kde._kernels:
        total += np.exp(kernel.evaluate(grid.reshape(-1, 1)))
    return np.log(total) - kde._logSumW


def test_empty_kde_is_falsy_and_has_no_kernels():
    kde = OnlineKDE(makeSpace((-4.0, 4.0, 41, False)))
    assert not kde
    assert kde.getNumKernels() == 0


def test_kde_is_truthy_after_a_deposit():
    kde = OnlineKDE(makeSpace((-4.0, 4.0, 41, False)))
    kde.update(np.array([0.0]), 0.0, np.array([0.25]))
    assert kde
    assert kde.getNumKernels() == 1


def test_grid_pdf_matches_a_direct_mixture_sum():
    space = makeSpace((-4.0, 4.0, 41, False))
    kde = OnlineKDE(space, compressionThreshold=0.0)
    rng = np.random.default_rng(0)
    for _ in range(12):
        kde.update(np.array([rng.uniform(-2, 2)]), rng.normal(), np.array([0.09]))
    grid = np.linspace(-4, 4, 41)
    assert kde.getLogPDF() == pytest.approx(directLogPDF(kde, grid), abs=1e-12)


def test_neff_matches_the_kish_definition():
    kde = OnlineKDE(makeSpace((-4.0, 4.0, 41, False)))
    rng = np.random.default_rng(1)
    logWeights = np.log(rng.uniform(0.5, 2.0, 25))
    for logWeight in logWeights:
        kde.update(np.array([rng.normal()]), logWeight, np.array([1.0]))
    weights = np.exp(logWeights)
    neff = np.exp(2 * kde._logSumW - kde._logSumWSq)
    assert neff == pytest.approx(weights.sum() ** 2 / (weights**2).sum())


def test_log_mean_density_is_the_mean_pdf_over_kernel_centers():
    kde = OnlineKDE(makeSpace((-4.0, 4.0, 41, False)))
    rng = np.random.default_rng(2)
    for _ in range(20):
        kde.update(np.array([rng.normal()]), rng.normal(), np.array([0.09]))
    centers = np.stack([k.position for k in kde._kernels])
    direct = np.mean([np.exp(kde.evaluate(c)) for c in centers])
    assert np.exp(kde.getLogMeanDensity()) == pytest.approx(direct, rel=1e-8)


def test_repeated_deposits_at_one_spot_are_compressed():
    kde = OnlineKDE(makeSpace((-4.0, 4.0, 41, False)))
    for _ in range(50):
        kde.update(np.array([0.0]), 0.0, np.array([0.25]))
    assert kde.getNumKernels() == 1


def test_well_separated_deposits_are_not_compressed():
    kde = OnlineKDE(makeSpace((-40.0, 40.0, 81, False)))
    for x in (-30.0, 0.0, 30.0):
        kde.update(np.array([x]), 0.0, np.array([0.01]))
    assert kde.getNumKernels() == 3


def test_zero_threshold_disables_merging():
    kde = OnlineKDE(makeSpace((-4.0, 4.0, 41, False)), compressionThreshold=0.0)
    for _ in range(25):
        kde.update(np.array([0.0]), 0.0, np.array([0.25]))
    assert kde.getNumKernels() == 25


def test_iadd_combines_weights_and_kernels():
    """__iadd__ transfers kernels verbatim (adjustBandwidth=False), because a
    peer's bandwidths already carry that peer's own Silverman factor."""
    from copy import copy

    space = makeSpace((-4.0, 4.0, 41, False))
    rng = np.random.default_rng(3)
    first = OnlineKDE(space, compressionThreshold=0.0)
    second = OnlineKDE(space, compressionThreshold=0.0)
    for _ in range(8):
        first.update(np.array([rng.normal()]), rng.normal(), np.array([0.09]))
        second.update(np.array([rng.normal()]), rng.normal(), np.array([0.09]))

    expectedLogSumW = np.logaddexp(first._logSumW, second._logSumW)
    expectedKernels = first.getNumKernels() + second.getNumKernels()
    union = [copy(k) for k in first._kernels] + [copy(k) for k in second._kernels]

    first += second
    assert first.getNumKernels() == expectedKernels
    assert first._logSumW == pytest.approx(expectedLogSumW)

    grid = np.linspace(-4, 4, 41)
    total = np.zeros_like(grid)
    for kernel in union:
        total += np.exp(kernel.evaluate(grid.reshape(-1, 1)))
    assert first.getLogPDF() == pytest.approx(
        np.log(total) - expectedLogSumW, abs=1e-12
    )


def test_copy_is_independent_of_the_original():
    from copy import copy

    space = makeSpace((-4.0, 4.0, 41, False))
    kde = OnlineKDE(space)
    kde.update(np.array([0.0]), 0.0, np.array([0.25]))
    duplicate = copy(kde)
    duplicate.update(np.array([2.0]), 0.0, np.array([0.25]))
    assert kde.getNumKernels() == 1
    assert duplicate.getNumKernels() == 2


def test_removal_cancellation_margin_stays_bounded_away_from_zero():
    """Pins the invariant behind spec section 12.4.

    _logsubexp(x, y) loses precision as x - y approaches zero. _pushKernel
    always adds the merged replacement kernel BEFORE removing what it absorbed,
    which bounds that margin. If a removal path without a compensating addition
    is ever introduced, this fails.
    """
    space = makeSpace((-4.0, 4.0, 21, False))
    kde = OnlineKDE(space)
    margins = []
    original = OnlineKDE._removeKernels

    def spy(self, centers, toRemove):
        for index in toRemove:
            margin = self._logPK - self._kernels[index].evaluate(centers)
            finite = np.isfinite(margin)
            if finite.any():
                margins.append(float(np.min(margin[finite])))
        return original(self, centers, toRemove)

    OnlineKDE._removeKernels = spy
    try:
        rng = np.random.default_rng(5)
        for _ in range(3000):
            kde.update(np.array([rng.normal()]), rng.normal(0, 4.0), np.array([0.01]))
    finally:
        OnlineKDE._removeKernels = original

    assert len(margins) > 1000
    assert min(margins) > 1e-3

    centers = np.stack([k.position for k in kde._kernels])
    recomputed = np.logaddexp.reduce(
        np.stack([k.evaluate(centers) for k in kde._kernels]), axis=0
    )
    assert kde._logPK == pytest.approx(recomputed, abs=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_kde.py -v`
Expected: FAIL — `ImportError: cannot import name 'OnlineKDE'`

- [ ] **Step 3: Implement `OnlineKDE`, appended to `src/openmm_opes/kde.py`**

```python
class OnlineKDE:
    """Online kernel density estimate with on-the-fly kernel compression.

    Parameters
    ----------
    cvSpace
        The :class:`CVSpace` the estimate lives on.
    compressionThreshold
        Merge a new kernel into any existing kernel closer than this, measured
        as a Mahalanobis distance. Zero disables merging.
    useExistingBandwidths
        Whether the nearest-neighbour search scales distances by the existing
        kernels' bandwidths rather than the incoming kernel's.
    kernelShape
        Either ``"gaussian"`` or ``"compact"``.
    """

    def __init__(
        self,
        cvSpace,
        compressionThreshold: float = 1.0,
        useExistingBandwidths: bool = True,
        kernelShape: str = "gaussian",
    ):
        if kernelShape not in KERNEL_SHAPES:
            raise ValueError(
                f"Unknown kernelShape {kernelShape!r}; "
                f"expected one of {sorted(KERNEL_SHAPES)}"
            )
        self._cvSpace = cvSpace
        self._compressionThreshold = compressionThreshold
        self._useExistingBandwidths = useExistingBandwidths
        self._shape = KERNEL_SHAPES[kernelShape]
        self._kernels: list[Kernel] = []
        self._logSumW = -np.inf
        self._logSumWSq = -np.inf
        self._logPK = np.empty(0)
        self._logPG = np.full(cvSpace.gridShape, -np.inf)
        self._d = cvSpace.numDimensions

    def __bool__(self) -> bool:
        return bool(self._kernels)

    def __copy__(self):
        new = self.__class__(
            self._cvSpace,
            self._compressionThreshold,
            self._useExistingBandwidths,
            self._shape.name,
        )
        new._kernels = list(map(copy, self._kernels))
        new._logSumW = self._logSumW
        new._logSumWSq = self._logSumWSq
        new._logPK = self._logPK.copy()
        new._logPG = self._logPG.copy()
        return new

    def __iadd__(self, other):
        for kernel in other._kernels:
            self._addKernel(
                kernel.position,
                kernel.bandwidth,
                kernel.logWeight,
                kernel.numSamples,
                adjustBandwidth=False,
            )
        return self

    @staticmethod
    def _logsubexp(x, y):
        """log(exp(x) - exp(y)), elementwise and numerically stable."""
        result = np.full_like(x, -np.inf)
        valid = y < x
        inner = -np.exp(y[valid] - x[valid])
        representable = inner > -1.0
        inner[representable] = np.log1p(inner[representable])
        inner[~representable] = -np.inf
        result[valid] = inner + x[valid]
        return result

    def _removeKernels(self, centers, toRemove):
        """Remove kernels by index, subtracting their contributions in log space.

        Callers MUST add any replacement kernel's contribution to _logPK and
        _logPG *before* calling this. _logsubexp loses precision as its two
        arguments converge, and the replacement is what keeps the removed
        kernel from dominating the density at its own center. See spec 12.4;
        tests/test_kde.py pins the resulting margin.
        """
        toRemove = sorted(toRemove, reverse=True)
        removed = []
        for index in toRemove:
            kernel = self._kernels.pop(index)
            self._logPK = self._logsubexp(self._logPK, kernel.evaluate(centers))
            self._logPG = self._logsubexp(self._logPG, kernel.evaluateOnGrid())
            removed.append(kernel)
        self._logPK = np.delete(self._logPK, toRemove)
        return removed

    def _pushKernel(self, newKernel):
        centers = np.stack([k.position for k in self._kernels])
        bandwidths = (
            np.stack([k.bandwidth for k in self._kernels])
            if self._useExistingBandwidths
            else newKernel.bandwidth
        )
        threshold = self._compressionThreshold
        index, minSqDist = newKernel.findNearest(centers, bandwidths)
        toRemove = []
        # threshold > 0 guard: without it, coincident kernels satisfy 0 <= 0
        # and merge even when compression is meant to be disabled.
        while threshold > 0 and index >= 0 and minSqDist <= threshold**2:
            toRemove.append(index)
            newKernel.merge(self._kernels[index])
            index, minSqDist = newKernel.findNearest(centers, bandwidths, toRemove)
        self._logPK = np.logaddexp(self._logPK, newKernel.evaluate(centers))
        self._logPG = np.logaddexp(self._logPG, newKernel.evaluateOnGrid())
        if toRemove:
            self._removeKernels(centers, toRemove)
        self._kernels.append(newKernel)
        self._logPK = np.append(
            self._logPK,
            np.logaddexp.reduce([k.evaluate(newKernel.position) for k in self._kernels]),
        )

    def _addKernel(
        self, position, bandwidth, logWeight, numSamples=1, adjustBandwidth=True
    ):
        self._logSumW = np.logaddexp(self._logSumW, logWeight)
        self._logSumWSq = np.logaddexp(self._logSumWSq, 2 * logWeight)
        if adjustBandwidth:
            neff = np.exp(2 * self._logSumW - self._logSumWSq)
            silverman = (neff * (self._d + 2) / 4) ** (-1 / (self._d + 4))
            bandwidth = bandwidth * silverman
        newKernel = Kernel(
            self._cvSpace, position, bandwidth, logWeight, numSamples, self._shape
        )
        if self._kernels:
            self._pushKernel(newKernel)
        else:
            self._kernels = [newKernel]
            self._logPG = newKernel.evaluateOnGrid()
            self._logPK = np.array([newKernel.logHeight])

    def update(self, position, logWeight, variance) -> None:
        """Deposit a kernel of the given log weight and per-CV variance."""
        self._addKernel(position, np.sqrt(variance), logWeight)

    def getNumKernels(self) -> int:
        """Number of compressed kernels currently stored."""
        return len(self._kernels)

    def getLogPDF(self):
        """Log of the normalized probability density on the grid."""
        return self._logPG - self._logSumW

    def getLogMeanDensity(self) -> float:
        """Log of Z_n, the mean density over the compressed kernel centers."""
        return (
            np.logaddexp.reduce(self._logPK)
            - np.log(len(self._kernels))
            - self._logSumW
        )

    def evaluate(self, point) -> float:
        """Log of the normalized density at a single point."""
        return (
            np.logaddexp.reduce([k.evaluate(point) for k in self._kernels])
            - self._logSumW
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_kde.py -v`
Expected: PASS (11 tests). The margin test takes a few seconds.

- [ ] **Step 5: Lint and commit**

```bash
ruff format src tests && ruff check src tests && ty check src tests
git add src/openmm_opes/kde.py tests/test_kde.py
git commit -m "feat: add OnlineKDE with kernel compression

Fixes the __bool__ AttributeError from the source implementation and
documents the _removeKernels ordering invariant that keeps the
incremental _logPK bookkeeping numerically safe."
```

---

### Task 5: KDE state save and restore

**Files:**
- Modify: `src/openmm_opes/kde.py`
- Modify: `tests/test_kde.py`

**Interfaces:**
- Consumes: `OnlineKDE` from Task 4.
- Produces: `OnlineKDE.getState() -> dict[str, np.ndarray | float]` with keys `positions` (n×d), `bandwidths` (n×d), `logWeights` (n), `numSamples` (n), `logSumW`, `logSumWSq`, `logPK`; and `OnlineKDE.setState(state) -> None`. `logPG` is rebuilt on restore, never stored. `CVSpace` is never serialized.

Fixes §7.4.3 (restoring an empty KDE raised `TypeError`).

- [ ] **Step 1: Write the failing tests, appended to `tests/test_kde.py`**

```python
def test_state_round_trip_preserves_the_density():
    space = makeSpace((-4.0, 4.0, 41, False))
    kde = OnlineKDE(space)
    rng = np.random.default_rng(7)
    for _ in range(40):
        kde.update(np.array([rng.normal()]), rng.normal(), np.array([0.09]))

    restored = OnlineKDE(space)
    restored.setState(kde.getState())

    assert restored.getNumKernels() == kde.getNumKernels()
    assert restored.getLogPDF() == pytest.approx(kde.getLogPDF(), abs=1e-12)
    assert restored.getLogMeanDensity() == pytest.approx(kde.getLogMeanDensity())


def test_empty_kde_state_round_trips():
    """Regression for spec 7.4.3: restoring an empty KDE raised TypeError.

    Routine on the warm-up path, where a walker syncs before its first deposit.
    """
    space = makeSpace((-4.0, 4.0, 41, False))
    restored = OnlineKDE(space)
    restored.setState(OnlineKDE(space).getState())
    assert restored.getNumKernels() == 0
    assert not restored
    assert np.all(np.isneginf(restored.getLogPDF()))


def test_state_contains_only_arrays_and_scalars():
    """The state must be np.savez-able, so no objects allowed."""
    space = makeSpace((-4.0, 4.0, 41, False))
    kde = OnlineKDE(space)
    kde.update(np.array([0.0]), 0.0, np.array([0.25]))
    for key, value in kde.getState().items():
        assert isinstance(value, (np.ndarray, float, int)), key
        if isinstance(value, np.ndarray):
            assert value.dtype != object, key
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_kde.py -k state -v`
Expected: FAIL — `AttributeError: 'OnlineKDE' object has no attribute 'getState'`

- [ ] **Step 3: Implement `getState`/`setState` on `OnlineKDE`**

```python
    def getState(self) -> dict:
        """Flat, npz-writable snapshot of this estimate.

        The CVSpace is deliberately excluded: all walkers share identical CV
        definitions, so a reader constructs its own and calls setState.
        """
        d = self._cvSpace.numDimensions
        if self._kernels:
            positions = np.stack([k.position for k in self._kernels])
            bandwidths = np.stack([k.bandwidth for k in self._kernels])
            logWeights = np.array([k.logWeight for k in self._kernels])
            numSamples = np.array([k.numSamples for k in self._kernels])
        else:
            positions = np.empty((0, d))
            bandwidths = np.empty((0, d))
            logWeights = np.empty(0)
            numSamples = np.empty(0, dtype=int)
        return {
            "positions": positions,
            "bandwidths": bandwidths,
            "logWeights": logWeights,
            "numSamples": numSamples,
            "logSumW": float(self._logSumW),
            "logSumWSq": float(self._logSumWSq),
            "logPK": self._logPK.copy(),
        }

    def setState(self, state) -> None:
        """Restore from a :meth:`getState` snapshot, discarding current contents."""
        self._kernels = [
            Kernel(self._cvSpace, position, bandwidth, logWeight, int(n), self._shape)
            for position, bandwidth, logWeight, n in zip(
                state["positions"],
                state["bandwidths"],
                state["logWeights"],
                state["numSamples"],
            )
        ]
        self._logSumW = float(state["logSumW"])
        self._logSumWSq = float(state["logSumWSq"])
        self._logPK = np.asarray(state["logPK"]).copy()
        # Seeded with -inf so an empty kernel list restores cleanly; the source
        # used a bare reduce here and raised TypeError on an empty sequence.
        self._logPG = functools.reduce(
            np.logaddexp,
            (k.evaluateOnGrid() for k in self._kernels),
            np.full(self._cvSpace.gridShape, -np.inf),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_kde.py -v`
Expected: PASS (14 tests)

- [ ] **Step 5: Lint and commit**

```bash
ruff format src tests && ruff check src tests && ty check src tests
git add src/openmm_opes/kde.py tests/test_kde.py
git commit -m "feat: add flat npz-writable KDE state save and restore

Seeds the logPG reduction so an empty KDE restores cleanly, which the
source could not do."
```

---

### Task 6: Parity against the source implementation

**Files:**
- Create: `tests/data/make_reference.py`
- Create: `tests/data/reference_kde.npz` (generated, then committed)
- Create: `tests/test_parity.py`

**Interfaces:**
- Consumes: `OnlineKDE` from Tasks 4–5.
- Produces: a committed reference fixture pinning default-path numerics.

This is the most important test in the suite: it proves the port did not silently change the science. The generator drives the **original** `online_kde.py`; the test replays the identical deposits through the new code.

- [ ] **Step 1: Write the reference generator**

`tests/data/make_reference.py`:

```python
"""Generate the parity fixture from the ORIGINAL online_kde.py.

Run once, commit the .npz, and do not run again unless the source
implementation itself changes:

    python tests/data/make_reference.py /path/to/opes-simulations
"""

import sys
from pathlib import Path

import numpy as np


def depositSequence(d, count, seed):
    """Deterministic deposits shared by the generator and the test."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-2.0, 2.0, (count, d))
    logWeights = rng.normal(0.0, 1.5, count)
    variances = np.full((count, d), 0.04)
    return positions, logWeights, variances


CASES = {
    "1d": dict(cvs=[(-4.0, 4.0, 61, False)], count=400, seed=11),
    "1d_periodic": dict(cvs=[(-np.pi, np.pi, 61, True)], count=400, seed=12),
    "2d": dict(cvs=[(-4.0, 4.0, 31, False), (-4.0, 4.0, 31, False)],
               count=300, seed=13),
}


def main(sourceDir):
    sys.path.insert(0, str(sourceDir))
    from online_kde import CVSpace, OnlineKDE  # the ORIGINAL implementation

    assert __import__("online_kde").COMPRESSION_THRESHOLD == 1.0
    assert __import__("online_kde").BOUNDED_KERNELS is False
    assert __import__("online_kde").UNCOMPRESSED_KDE is False
    assert __import__("online_kde").USE_EXISTING_BANDWIDTHS is True

    arrays = {}
    for name, case in CASES.items():
        space = CVSpace([CVSpace._CV(*cv) for cv in case["cvs"]])
        kde = OnlineKDE(space)
        positions, logWeights, variances = depositSequence(
            len(case["cvs"]), case["count"], case["seed"]
        )
        for position, logWeight, variance in zip(positions, logWeights, variances):
            kde.update(position, logWeight, variance)
        arrays[f"{name}_logPDF"] = kde.getLogPDF()
        arrays[f"{name}_logMeanDensity"] = np.array(kde.getLogMeanDensity())
        arrays[f"{name}_numKernels"] = np.array(kde.getNumKernels())
        arrays[f"{name}_centers"] = np.stack([k.position for k in kde._kernels])
        arrays[f"{name}_bandwidths"] = np.stack([k.bandwidth for k in kde._kernels])

    out = Path(__file__).parent / "reference_kde.npz"
    np.savez(out, **arrays)
    print(f"wrote {out} with {len(arrays)} arrays")


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 2: Generate the fixture**

Run:
```bash
/home/charlles/miniforge3/envs/openmm/bin/python tests/data/make_reference.py \
    /home/charlles/opes/opes-simulations
```
Expected: `wrote .../reference_kde.npz with 15 arrays`. If the asserts trip, the source's globals have been changed from the values this port assumes — stop and re-read spec §7.3 before continuing.

- [ ] **Step 3: Write the parity test**

`tests/test_parity.py`:

```python
"""Pins the port against the original implementation's numerics.

The port reorganizes code but changes no arithmetic on the default path, so
these must agree to floating-point tolerance. If this test fails, the port
changed the science. Do NOT loosen the tolerance to make it pass.
"""

from pathlib import Path

import numpy as np
import pytest

from openmm_opes.kde import OnlineKDE
from tests.data.make_reference import CASES, depositSequence
from tests.helpers import makeSpace

REFERENCE = Path(__file__).parent / "data" / "reference_kde.npz"


@pytest.fixture(scope="module")
def reference():
    return np.load(REFERENCE)


@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_the_original_implementation(reference, name):
    case = CASES[name]
    kde = OnlineKDE(makeSpace(*case["cvs"]))
    positions, logWeights, variances = depositSequence(
        len(case["cvs"]), case["count"], case["seed"]
    )
    for position, logWeight, variance in zip(positions, logWeights, variances):
        kde.update(position, logWeight, variance)

    assert kde.getNumKernels() == int(reference[f"{name}_numKernels"])
    assert kde.getLogPDF() == pytest.approx(reference[f"{name}_logPDF"], rel=1e-12)
    assert kde.getLogMeanDensity() == pytest.approx(
        float(reference[f"{name}_logMeanDensity"]), rel=1e-12
    )
    centers = np.stack([k.position for k in kde._kernels])
    bandwidths = np.stack([k.bandwidth for k in kde._kernels])
    assert centers == pytest.approx(reference[f"{name}_centers"], rel=1e-12)
    assert bandwidths == pytest.approx(reference[f"{name}_bandwidths"], rel=1e-12)
```

Add `tests/data/__init__.py` (empty) so `tests.data.make_reference` is importable.

- [ ] **Step 4: Run the parity test**

Run: `pytest tests/test_parity.py -v`
Expected: PASS (3 tests). A failure here means real numerical divergence — debug the port, never the tolerance.

- [ ] **Step 5: Commit**

```bash
ruff format src tests && ruff check src tests
git add tests/data tests/test_parity.py
git commit -m "test: pin KDE numerics against the original implementation"
```

---

### Task 7: `io.py` — multi-walker bias sharing over npz

**Files:**
- Modify: `src/openmm_opes/io.py`
- Create: `tests/test_io.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (deliberately generic).
- Produces:
  - `BiasSharer(biasDir: str, walkerId: int | None = None)` with attribute `walkerId: int`.
  - `save(state: dict[str, np.ndarray | float]) -> None` — atomic write, previous index removed.
  - `load() -> dict[int, dict]` — peer states by walker id, own files skipped, only advanced indices re-read.
  - Module constant `FILENAME_PATTERN`.

Per the plan-level decision, this module knows nothing about kernels or OpenMM — it moves flat dicts of arrays.

- [ ] **Step 1: Write the failing tests**

`tests/test_io.py`:

```python
import numpy as np
import pytest

from openmm_opes.io import BiasSharer


def makeState(value):
    return {"positions": np.full((2, 1), value), "logSumW": float(value)}


def test_save_writes_one_file_named_for_the_walker(tmp_path):
    sharer = BiasSharer(str(tmp_path), walkerId=7)
    sharer.save(makeState(1.0))
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["kde_7_1.npz"]


def test_save_removes_the_previous_index(tmp_path):
    sharer = BiasSharer(str(tmp_path), walkerId=7)
    sharer.save(makeState(1.0))
    sharer.save(makeState(2.0))
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["kde_7_2.npz"]


def test_save_leaves_no_temporary_files(tmp_path):
    sharer = BiasSharer(str(tmp_path), walkerId=7)
    sharer.save(makeState(1.0))
    assert not any(p.name.startswith("temp_") for p in tmp_path.iterdir())


def test_load_ignores_the_walkers_own_files(tmp_path):
    sharer = BiasSharer(str(tmp_path), walkerId=7)
    sharer.save(makeState(1.0))
    assert sharer.load() == {}


def test_load_picks_up_a_peer(tmp_path):
    mine = BiasSharer(str(tmp_path), walkerId=1)
    peer = BiasSharer(str(tmp_path), walkerId=2)
    peer.save(makeState(5.0))
    loaded = mine.load()
    assert set(loaded) == {2}
    assert loaded[2]["logSumW"] == pytest.approx(5.0)
    assert loaded[2]["positions"] == pytest.approx(np.full((2, 1), 5.0))


def test_load_returns_nothing_when_no_peer_advanced(tmp_path):
    mine = BiasSharer(str(tmp_path), walkerId=1)
    peer = BiasSharer(str(tmp_path), walkerId=2)
    peer.save(makeState(5.0))
    assert set(mine.load()) == {2}
    assert mine.load() == {}


def test_load_rereads_a_peer_that_advanced(tmp_path):
    mine = BiasSharer(str(tmp_path), walkerId=1)
    peer = BiasSharer(str(tmp_path), walkerId=2)
    peer.save(makeState(5.0))
    mine.load()
    peer.save(makeState(6.0))
    loaded = mine.load()
    assert loaded[2]["logSumW"] == pytest.approx(6.0)


def test_load_tolerates_a_peer_file_vanishing_mid_scan(tmp_path):
    mine = BiasSharer(str(tmp_path), walkerId=1)
    peer = BiasSharer(str(tmp_path), walkerId=2)
    peer.save(makeState(5.0))
    for path in tmp_path.glob("kde_2_*.npz"):
        path.unlink()
    with pytest.warns(UserWarning, match="seems to have been deleted"):
        assert mine.load() == {}


def test_walker_ids_are_distinct_when_not_supplied(tmp_path):
    ids = {BiasSharer(str(tmp_path)).walkerId for _ in range(20)}
    assert len(ids) == 20


def test_saved_files_contain_no_pickled_objects(tmp_path):
    """The format must load with allow_pickle=False (spec section 6)."""
    sharer = BiasSharer(str(tmp_path), walkerId=7)
    sharer.save(makeState(1.0))
    path = next(tmp_path.glob("kde_7_*.npz"))
    with np.load(path, allow_pickle=False) as data:
        assert set(data) == {"positions", "logSumW"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_io.py -v`
Expected: FAIL — `ImportError: cannot import name 'BiasSharer'`

- [ ] **Step 3: Implement `src/openmm_opes/io.py`**

```python
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

    __slots__ = ("walkerId", "index", "state")

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
        generator so the global NumPy random state is left undisturbed.
    """

    def __init__(self, biasDir: str, walkerId: int | None = None):
        self.biasDir = biasDir
        self.walkerId = (
            int(np.random.default_rng().integers(0x7FFFFFFF))
            if walkerId is None
            else walkerId
        )
        self._saveIndex = 0
        self._loaded: dict[int, _LoadedBias] = {}

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
```

Note: scalars written by `np.savez` come back as 0-d arrays. `float(state["logSumW"])` in `OnlineKDE.setState` handles that, and the test above compares with `pytest.approx`, which also handles it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_io.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Lint and commit**

```bash
ruff format src tests && ruff check src tests && ty check src tests
git add src/openmm_opes/io.py tests/test_io.py
git commit -m "feat: add npz-based multi-walker bias sharing

Replaces the source's pickle format, which executes arbitrary code on
load and embeds module paths that this port's rename would break."
```

---

### Task 8: `RunningAverage` and the `OPES` constructor

**Files:**
- Create: `src/openmm_opes/opes.py`
- Create: `tests/test_opes.py`

**Interfaces:**
- Consumes: `CVSpace`, `OnlineKDE` (Tasks 2–5); `BiasSharer` (Task 7).
- Produces:
  - `RunningAverage(numDimensions)` with `update(sample)`, `get() -> np.ndarray`, `copy()`, `__iadd__`, `getState()`, `setState(state)`.
  - `OPES(...)` constructed per spec §5, exposing `variables`, `temperature`, `barrier`, `frequency`, `varianceFrequency`, `biasFactor`, `exploreMode`, `bounded`, `saveFrequency`, `biasDir`, `warmupSteps`, and the force added to the system.
  - `getNumKernels()`, `getVariance()`.

Fixes §7.4.1 (aliasing) and the three §7.6 validation gaps, and implements the §7.5 `biasWidth` decision.

- [ ] **Step 1: Write the failing tests**

`tests/test_opes.py`:

```python
import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import app, unit  # noqa: E402

from openmm_opes.opes import OPES, RunningAverage  # noqa: E402


def makeSystemAndVariable(sigma=0.1, gridWidth=51):
    system = openmm.System()
    system.addParticle(1.0)
    force = openmm.CustomExternalForce("x")
    force.addParticle(0, [])
    variable = app.BiasVariable(force, -2.0, 2.0, sigma, False, gridWidth)
    return system, variable


def makeOPES(**kwargs):
    system, variable = makeSystemAndVariable(kwargs.pop("sigma", 0.1))
    kwargs.setdefault("varianceFrequency", 10)
    return OPES(system, [variable], 300.0, 20.0, 100, **kwargs)


def test_running_average_copies_do_not_share_their_accumulator():
    """Regression for spec 7.4.1.

    _syncWithDisk copies the 'self' accumulator to seed 'total', then adds
    peers into it. With a shared array that in-place add silently corrupted
    the walker's own variance, affecting every multi-walker run.
    """
    own = RunningAverage(1)
    own.update(np.array([5.0]))
    peer = RunningAverage(1)
    peer.update(np.array([1.0]))

    total = own.copy()
    before = own.get().copy()
    total += peer

    assert own.get() == pytest.approx(before)
    assert total.get() == pytest.approx([3.0])
    assert not np.shares_memory(own._total, total._total)


def test_running_average_state_round_trips():
    average = RunningAverage(2)
    average.update(np.array([1.0, 2.0]))
    average.update(np.array([3.0, 4.0]))
    restored = RunningAverage(2)
    restored.setState(average.getState())
    assert restored.get() == pytest.approx(average.get())


def test_constructor_adds_a_force_in_a_free_group():
    system, variable = makeSystemAndVariable()
    before = system.getNumForces()
    sampler = OPES(system, [variable], 300.0, 20.0, 100, 10)
    assert system.getNumForces() == before + 1
    assert sampler._force.getForceGroup() not in {
        system.getForce(i).getForceGroup()
        for i in range(system.getNumForces())
        if system.getForce(i) is not sampler._force
    }


def test_bias_factor_defaults_to_barrier_over_kT():
    sampler = makeOPES()
    kbt = (unit.MOLAR_GAS_CONSTANT_R * 300 * unit.kelvin).value_in_unit(
        unit.kilojoules_per_mole
    )
    assert sampler._biasFactor == pytest.approx(20.0 / kbt)


def test_variance_frequency_must_divide_frequency():
    with pytest.raises(ValueError, match="varianceFrequency must be a divisor"):
        makeOPES(varianceFrequency=30)


def test_zero_variance_frequency_is_rejected():
    """Regression for spec 7.6: this used to reach a ZeroDivisionError."""
    with pytest.raises(ValueError, match="varianceFrequency must be positive"):
        makeOPES(varianceFrequency=0)


def test_zero_save_frequency_is_rejected(tmp_path):
    """Regression for spec 7.6: this used to construct, then fail inside step."""
    with pytest.raises(ValueError, match="saveFrequency must be positive"):
        makeOPES(saveFrequency=0, biasDir=str(tmp_path))


def test_save_frequency_and_bias_dir_must_come_together(tmp_path):
    with pytest.raises(ValueError, match="Must specify both"):
        makeOPES(saveFrequency=100)
    with pytest.raises(ValueError, match="Must specify both"):
        makeOPES(biasDir=str(tmp_path))


def test_explicit_bias_factor_error_names_bias_factor():
    """Regression for spec 7.6: the message used to blame 'barrier'."""
    with pytest.raises(ValueError, match="biasFactor must be greater than 1"):
        makeOPES(biasFactor=0.5)


def test_low_barrier_error_names_the_barrier():
    system, variable = makeSystemAndVariable()
    with pytest.raises(ValueError, match="barrier must be greater than 1 kT"):
        OPES(system, [variable], 300.0, 0.5, 100, 10)


def test_mixed_periodicity_is_rejected():
    system = openmm.System()
    system.addParticle(1.0)
    variables = []
    for periodic in (True, False):
        force = openmm.CustomExternalForce("x")
        force.addParticle(0, [])
        variables.append(app.BiasVariable(force, -2.0, 2.0, 0.1, periodic, 31))
    with pytest.raises(ValueError, match="mixed periodic"):
        OPES(system, variables, 300.0, 20.0, 100, 10)


def test_more_than_three_cvs_is_rejected():
    system = openmm.System()
    system.addParticle(1.0)
    variables = []
    for _ in range(4):
        force = openmm.CustomExternalForce("x")
        force.addParticle(0, [])
        variables.append(app.BiasVariable(force, -2.0, 2.0, 0.1, False, 11))
    with pytest.raises(ValueError, match="1, 2, or 3 collective variables"):
        OPES(system, variables, 300.0, 20.0, 100, 10)


def test_fixed_bandwidth_stores_gamma_times_the_unbiased_variance():
    """Spec 7.5: biasWidth is the UNBIASED sigma^0, per both papers.

    _variance holds a sampled-distribution variance by convention, and the
    sampled distribution is wider than the unbiased one by sqrt(gamma).
    """
    sampler = makeOPES(varianceFrequency=None, sigma=0.1)
    expected = sampler._biasFactor * 0.1**2
    assert sampler.getVariance() == pytest.approx([expected])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_opes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'openmm_opes.opes'`

- [ ] **Step 3: Implement `src/openmm_opes/opes.py`**

```python
"""The OPES bias, integrated with OpenMM as a CustomCVForce.

Closely mirrors ``openmm.app.Metadynamics``: same constructor shape, same
unit-coercion idiom, same force-group selection, same ``step`` contract.
"""

from __future__ import annotations

import numpy as np
import openmm as mm
from openmm import unit

from .io import BiasSharer
from .kde import CVSpace, OnlineKDE


class RunningAverage:
    """Accumulates a running mean of per-CV squared deviations."""

    def __init__(self, numDimensions: int = 1):
        self._num = 0
        self._total = np.zeros(numDimensions)

    def __iadd__(self, other):
        self._total = self._total + other._total
        self._num += other._num
        return self

    def copy(self):
        """An independent copy. The array is copied, not aliased."""
        new = self.__class__(self._total.shape[0])
        new._num = self._num
        # The source aliased this array, so merging peers into a copy silently
        # corrupted the original. See spec 7.4.1.
        new._total = self._total.copy()
        return new

    def update(self, sample) -> None:
        """Fold one sample into the average."""
        self._num += 1
        self._total = self._total + sample

    def get(self):
        """The running average."""
        return self._total / max(1, self._num)

    def getState(self) -> dict:
        """Flat, npz-writable snapshot."""
        return {"num": float(self._num), "total": self._total.copy()}

    def setState(self, state) -> None:
        """Restore from a :meth:`getState` snapshot."""
        self._num = int(state["num"])
        self._total = np.asarray(state["total"]).copy()


class OPES:
    """On-the-fly Probability Enhanced Sampling.

    Implements OPES and its exploratory variant, following Invernizzi and
    Parrinello, https://doi.org/10.1021/acs.jpclett.0c00497 and
    https://doi.org/10.1021/acs.jctc.2c00152.

    Parameters
    ----------
    system: System
        The System to simulate. A CustomCVForce implementing the bias is
        created and added to it.
    variables: list of BiasVariable
        The collective variables whose sampling should be enhanced.
    temperature: temperature
        The temperature at which the simulation is run.
    barrier: energy
        The free energy barrier the simulation should overcome.
    frequency: int
        Interval in time steps at which to deposit a kernel.
    varianceFrequency: int or None
        Interval in time steps at which to update the CV variance. When None,
        the bandwidth is fixed and taken from each variable's ``biasWidth``,
        which is interpreted as the standard deviation of the UNBIASED
        distribution, exactly as both papers define sigma^(0).
    biasFactor: float, optional
        Defaults to ``barrier / kT``.
    exploreMode: bool
        Whether to apply the OPES-explore variant.
    bounded: bool
        Whether non-periodic CVs have reflective boundaries.
    saveFrequency: int, optional
        Interval in time steps at which to share the bias on disk. Must be a
        multiple of ``frequency``.
    biasDir: str, optional
        Directory shared with other walkers.
    warmupSteps: int, optional
        When given, run this many steps without depositing kernels while
        measuring the CV variance, then freeze it. See the spec, section 7.7.
    compressionThreshold: float
        Mahalanobis distance below which kernels are merged.
    useExistingBandwidths: bool
        Whether the merge search scales by existing kernels' bandwidths.
    kernelShape: str
        ``"gaussian"`` or ``"compact"``.
    statsWindowSize: int
        Window, in deposition strides, of the running CV-mean estimate.
    """

    def __init__(
        self,
        system,
        variables,
        temperature,
        barrier,
        frequency,
        varianceFrequency,
        biasFactor=None,
        exploreMode: bool = False,
        bounded: bool = False,
        saveFrequency=None,
        biasDir=None,
        warmupSteps=None,
        compressionThreshold: float = 1.0,
        useExistingBandwidths: bool = True,
        kernelShape: str = "gaussian",
        statsWindowSize: int = 10,
    ):
        if not unit.is_quantity(temperature):
            temperature = temperature * unit.kelvin
        if not unit.is_quantity(barrier):
            barrier = barrier * unit.kilojoules_per_mole

        self.variables = variables
        self.temperature = temperature
        self.barrier = barrier
        self.frequency = frequency
        self.varianceFrequency = varianceFrequency
        self.exploreMode = exploreMode
        self.bounded = bounded
        self.saveFrequency = saveFrequency
        self.biasDir = biasDir
        self.warmupSteps = warmupSteps
        self.statsWindowSize = statsWindowSize

        d = len(variables)
        kbt = unit.MOLAR_GAS_CONSTANT_R * temperature
        userSuppliedBiasFactor = biasFactor is not None
        if not userSuppliedBiasFactor:
            biasFactor = barrier / kbt
        self.biasFactor = biasFactor
        numPeriodics = sum(v.periodic for v in variables)
        freeGroups = set(range(32)) - {f.getForceGroup() for f in system.getForces()}
        self._validate(d, biasFactor, numPeriodics, freeGroups, userSuppliedBiasFactor)

        prefactor = (1 - 1 / biasFactor) * kbt
        if exploreMode:
            prefactor *= biasFactor
        self._kbt = kbt.in_units_of(unit.kilojoules_per_mole)
        self._biasFactor = biasFactor
        self._prefactor = prefactor
        self._logEpsilon = -barrier / prefactor

        self._cvSpace = CVSpace(variables, bounded)
        self._cases = ("total",) + ("self",) * bool(saveFrequency)
        kdeOptions = dict(
            compressionThreshold=compressionThreshold,
            useExistingBandwidths=useExistingBandwidths,
            kernelShape=kernelShape,
        )
        self._kde = {}
        for case in self._cases:
            self._kde[case] = OnlineKDE(self._cvSpace, **kdeOptions)
            self._kde[f"{case}.rw"] = OnlineKDE(self._cvSpace, **kdeOptions)

        self._adaptiveVariance = varianceFrequency is not None
        self._interval = varianceFrequency or frequency
        self._variance = {case: RunningAverage(d) for case in self._cases}
        self._warmupComplete = warmupSteps is None
        if self._adaptiveVariance:
            self._tau = statsWindowSize * frequency // varianceFrequency
            self._counter = 0
            self._sampleMean = np.zeros(d)
        else:
            # Spec 7.5: biasWidth is the unbiased sigma^(0), while _variance
            # holds a sampled-distribution variance, which is gamma times wider.
            self._setFixedVariance(
                biasFactor * np.array([v.biasWidth**2 for v in variables])
            )

        self._sharer = BiasSharer(biasDir) if saveFrequency else None

        gridWidths = [v.gridWidth for v in variables]
        self._widths = [] if d == 1 else gridWidths
        self._limits = sum(([v.minValue, v.maxValue] for v in variables), [])
        periodic = numPeriodics == d
        initial = np.full(
            int(np.prod(gridWidths)), -barrier / unit.kilojoules_per_mole
        )

        energyFunction = "table(" + ",".join(f"cv{i}" for i in range(d)) + ")"
        self._force = mm.CustomCVForce(energyFunction)
        for i, variable in enumerate(variables):
            self._force.addCollectiveVariable(f"cv{i}", variable.force)
        # Explicit branch rather than a dynamic getattr, so ty can check it.
        if d == 1:
            table = mm.Continuous1DFunction(initial, *self._limits, periodic)
        elif d == 2:
            table = mm.Continuous2DFunction(
                *self._widths, initial, *self._limits, periodic
            )
        else:
            table = mm.Continuous3DFunction(
                *self._widths, initial, *self._limits, periodic
            )
        self._force.addTabulatedFunction("table", table)
        self._force.setForceGroup(max(freeGroups))
        system.addForce(self._force)

    def _validate(
        self, d, biasFactor, numPeriodics, freeGroups, userSuppliedBiasFactor
    ):
        if self.varianceFrequency is not None and self.varianceFrequency <= 0:
            raise ValueError("varianceFrequency must be positive or None")
        if self.varianceFrequency and self.frequency % self.varianceFrequency != 0:
            raise ValueError("varianceFrequency must be a divisor of frequency")
        if (self.saveFrequency is None) != (self.biasDir is None):
            raise ValueError("Must specify both saveFrequency and biasDir")
        if self.saveFrequency is not None:
            if self.saveFrequency <= 0:
                raise ValueError("saveFrequency must be positive")
            if self.saveFrequency % self.frequency != 0:
                raise ValueError("saveFrequency must be a multiple of frequency")
        if self.warmupSteps is not None:
            if self.varianceFrequency is None:
                raise ValueError("warmupSteps requires varianceFrequency")
            if self.warmupSteps <= 0:
                raise ValueError("warmupSteps must be positive")
            if self.warmupSteps % self.varianceFrequency != 0:
                raise ValueError("warmupSteps must be a multiple of varianceFrequency")
            if self.warmupSteps // self.varianceFrequency < 2:
                raise ValueError(
                    "warmupSteps must span at least two varianceFrequency intervals"
                )
        if biasFactor <= 1.0:
            raise ValueError(
                "biasFactor must be greater than 1"
                if userSuppliedBiasFactor
                else "barrier must be greater than 1 kT"
            )
        if numPeriodics not in [0, d]:
            raise ValueError("OPES cannot handle mixed periodic/non-periodic variables")
        if not 1 <= d <= 3:
            raise ValueError("OPES requires 1, 2, or 3 collective variables")
        if not freeGroups:
            raise RuntimeError("OPES requires a free force group, but all are in use.")

    def _setFixedVariance(self, value) -> None:
        """Replace every accumulator with a frozen value (spec 7.5 and 7.7)."""
        for case in self._cases:
            average = RunningAverage(len(self.variables))
            average.update(np.asarray(value, dtype=float))
            self._variance[case] = average
        self._adaptiveVariance = False

    def getVariance(self):
        """Variance currently used to size newly deposited kernels."""
        return self._variance["total"].get()

    def getNumKernels(self) -> int:
        """Number of kernels in the estimate that defines the bias."""
        return self._kde["total" if self.exploreMode else "total.rw"].getNumKernels()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_opes.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Lint and commit**

```bash
ruff format src tests && ruff check src tests && ty check src tests
git add src/openmm_opes/opes.py tests/test_opes.py
git commit -m "feat: add RunningAverage and the OPES constructor

Fixes the RunningAverage aliasing bug that silently corrupted multi-walker
variance, rejects the varianceFrequency=0 and saveFrequency=0 configurations
that previously failed late, and names the right argument when biasFactor is
invalid."
```

---

### Task 9: Bias evaluation, deposition, and `step`

**Files:**
- Modify: `src/openmm_opes/opes.py`
- Modify: `src/openmm_opes/__init__.py`
- Modify: `tests/test_opes.py`

**Interfaces:**
- Consumes: Task 8's `OPES`.
- Produces: `getBias()`, `getFreeEnergy()`, `getAverageDensity()`, `getCollectiveVariables(simulation)`, `addKernel(values, biasEnergy, variance=None)`, `updateContext(context)`, `step(simulation, steps)`; `openmm_opes.OPES` re-exported.

Fixes §7.4.4 (the NaN bias).

- [ ] **Step 1: Write the failing tests, appended to `tests/test_opes.py`**

```python
def referenceBias(sampler):
    """Eq. iter_bias / iter_bias-explore, evaluated independently."""
    kde = sampler._kde["total" if sampler.exploreMode else "total.rw"]
    probability = np.exp(kde.getLogPDF())
    normalization = np.exp(kde.getLogMeanDensity())
    prefactor = sampler._prefactor.value_in_unit(unit.kilojoules_per_mole)
    epsilon = np.exp(sampler._logEpsilon)
    return prefactor * np.log(probability / normalization + epsilon)


@pytest.mark.parametrize("exploreMode", [False, True])
def test_bias_matches_the_published_equation(exploreMode):
    sampler = makeOPES(exploreMode=exploreMode)
    rng = np.random.default_rng(1)
    for _ in range(40):
        sampler.addKernel(
            np.array([rng.uniform(-1.5, 1.5)]),
            rng.uniform(0, 5) * unit.kilojoules_per_mole,
            variance=np.array([0.01]),
        )
    got = sampler.getBias().value_in_unit(unit.kilojoules_per_mole)
    assert got == pytest.approx(referenceBias(sampler), abs=1e-10)


def test_prefactor_follows_the_mode():
    kbt = (unit.MOLAR_GAS_CONSTANT_R * 300 * unit.kelvin).value_in_unit(
        unit.kilojoules_per_mole
    )
    plain = makeOPES()
    explore = makeOPES(exploreMode=True)
    gamma = plain._biasFactor
    assert plain._prefactor.value_in_unit(
        unit.kilojoules_per_mole
    ) == pytest.approx((1 - 1 / gamma) * kbt)
    assert explore._prefactor.value_in_unit(
        unit.kilojoules_per_mole
    ) == pytest.approx((gamma - 1) * kbt)


def test_epsilon_is_exp_minus_barrier_over_prefactor():
    sampler = makeOPES()
    prefactor = sampler._prefactor.value_in_unit(unit.kilojoules_per_mole)
    assert np.exp(sampler._logEpsilon) == pytest.approx(np.exp(-20.0 / prefactor))


def test_free_energy_always_uses_the_reweighted_estimate():
    sampler = makeOPES(exploreMode=True)
    sampler.addKernel(np.array([0.0]), 0.0 * unit.kilojoules_per_mole,
                      variance=np.array([0.01]))
    kbt = sampler._kbt.value_in_unit(unit.kilojoules_per_mole)
    expected = -kbt * sampler._kde["total.rw"].getLogPDF()
    got = sampler.getFreeEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert got == pytest.approx(expected)


def test_reweighted_kde_uses_variance_divided_by_the_bias_factor():
    sampler = makeOPES()
    sampler.addKernel(np.array([0.0]), 0.0 * unit.kilojoules_per_mole,
                      variance=np.array([0.04]))
    plain = sampler._kde["total"]._kernels[0].bandwidth[0]
    reweighted = sampler._kde["total.rw"]._kernels[0].bandwidth[0]
    assert plain / reweighted == pytest.approx(np.sqrt(sampler._biasFactor))


def test_non_positive_variance_skips_deposition_instead_of_producing_nan():
    """Regression for spec 7.4.4.

    A zero variance made a zero-bandwidth kernel whose logHeight is -inf,
    while its weight still entered logSumW. getLogPDF and getLogMeanDensity
    were then both -inf and their difference NaN, which reached the forces.
    """
    sampler = makeOPES()
    with pytest.warns(UserWarning, match="variance"):
        sampler.addKernel(
            np.array([0.0]),
            0.0 * unit.kilojoules_per_mole,
            variance=np.array([0.0]),
        )
    assert sampler.getNumKernels() == 0
    sampler.addKernel(np.array([0.0]), 0.0 * unit.kilojoules_per_mole,
                      variance=np.array([0.01]))
    assert np.all(np.isfinite(sampler.getBias().value_in_unit(
        unit.kilojoules_per_mole)))


def test_opes_is_exported_from_the_package():
    import openmm_opes

    assert openmm_opes.OPES is OPES
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_opes.py -k "bias or energy or export or variance_divided or nan" -v`
Expected: FAIL — `AttributeError: 'OPES' object has no attribute 'addKernel'`

- [ ] **Step 3: Implement the remaining `OPES` methods**

Add `import warnings` to the imports, then append to the class:

```python
    def getBias(self):
        """The OPES bias potential on the grid."""
        kde = self._kde["total" if self.exploreMode else "total.rw"]
        return self._prefactor * np.logaddexp(
            kde.getLogPDF() - kde.getLogMeanDensity(), self._logEpsilon
        )

    def getFreeEnergy(self):
        """Free energy as a function of the collective variables.

        Returned as an N-dimensional array in kJ/mole. The i'th position along
        an axis corresponds to ``minValue + i*(maxValue-minValue)/gridWidth``.
        Always the importance-sampling estimate, which converges better than
        the direct one in explore mode.
        """
        return -self._kbt * self._kde["total.rw"].getLogPDF()

    def getAverageDensity(self):
        """Z_n, the mean density over the explored CV space."""
        kde = self._kde["total" if self.exploreMode else "total.rw"]
        return np.exp(kde.getLogMeanDensity())

    def getCollectiveVariables(self, simulation):
        """Current values of all collective variables in a Simulation."""
        return self._force.getCollectiveVariableValues(simulation.context)

    def updateContext(self, context) -> None:
        """Push the current bias into a Context."""
        bias = self.getBias().value_in_unit(unit.kilojoules_per_mole)
        self._force.getTabulatedFunction(0).setFunctionParameters(
            *self._widths, bias.ravel(), *self._limits
        )
        self._force.updateParametersInContext(context)

    def addKernel(self, values, biasEnergy, variance=None) -> None:
        """Deposit a kernel into the probability estimates.

        This does not refresh any Context; call :meth:`updateContext`
        afterwards if a simulation is running.
        """
        if not unit.is_quantity(biasEnergy):
            biasEnergy = biasEnergy * unit.kilojoules_per_mole
        if variance is None:
            variance = self._variance["total"].get()
        if np.any(np.asarray(variance) <= 0):
            # Spec 7.4.4: a zero bandwidth poisons the estimate with NaN.
            warnings.warn(
                "Skipping kernel deposition: the CV variance estimate is not "
                "yet positive. Use a varianceFrequency smaller than frequency, "
                "or set warmupSteps.",
                stacklevel=2,
            )
            return
        for case in self._cases:
            self._kde[case].update(values, 0.0, variance)
            self._kde[f"{case}.rw"].update(
                values, biasEnergy / self._kbt, variance / self._biasFactor
            )

    def _updateSampleStats(self, values) -> None:
        self._counter += 1
        delta = self._cvSpace.displacement(self._sampleMean, values)
        x = 1 / min(self._tau, self._counter)
        self._sampleMean = self._cvSpace.endpoint(self._sampleMean, x * delta)
        sqdev = delta * self._cvSpace.displacement(self._sampleMean, values)
        for case in self._cases:
            self._variance[case].update(sqdev)

    def _syncWithDisk(self) -> None:
        self._sharer.save(self._getSharedState())
        if not self._sharer.load():
            return
        self._kde["total"] = copy(self._kde["self"])
        self._kde["total.rw"] = copy(self._kde["self.rw"])
        self._variance["total"] = self._variance["self"].copy()
        for state in self._sharer.getLoadedStates().values():
            self._kde["total"] += self._kdeFromState(state, "kde")
            self._kde["total.rw"] += self._kdeFromState(state, "kdeRW")
            peer = RunningAverage(len(self.variables))
            peer.setState(
                {"num": state["var_num"], "total": state["var_total"]}
            )
            self._variance["total"] += peer

    def _getSharedState(self) -> dict:
        state = {}
        for prefix, key in (("kde", "self"), ("kdeRW", "self.rw")):
            for name, value in self._kde[key].getState().items():
                state[f"{prefix}_{name}"] = value
        variance = self._variance["self"].getState()
        state["var_num"] = variance["num"]
        state["var_total"] = variance["total"]
        return state

    def _kdeFromState(self, state, prefix):
        kde = OnlineKDE(self._cvSpace)
        strip = len(prefix) + 1
        kde.setState(
            {
                key[strip:]: value
                for key, value in state.items()
                if key.startswith(f"{prefix}_")
            }
        )
        return kde

    def step(self, simulation, steps) -> None:
        """Advance the simulation by a number of time steps.

        Parameters
        ----------
        simulation: Simulation
            The Simulation to advance.
        steps: int
            The number of time steps to integrate.
        """
        stepsToGo = steps
        while stepsToGo > 0:
            nextSteps = min(
                stepsToGo,
                self._interval - simulation.currentStep % self._interval,
            )
            simulation.step(nextSteps)
            if simulation.currentStep % self._interval == 0:
                self._onInterval(simulation)
            stepsToGo -= nextSteps

    def _onInterval(self, simulation) -> None:
        position = self.getCollectiveVariables(simulation)
        if not self._warmupComplete:
            self._updateSampleStats(position)
            if simulation.currentStep >= self.warmupSteps:
                self._finishWarmup()
            return
        if self._adaptiveVariance:
            self._updateSampleStats(position)
        if simulation.currentStep % self.frequency == 0:
            groups = {self._force.getForceGroup()}
            energy = simulation.context.getState(
                getEnergy=True, groups=groups
            ).getPotentialEnergy()
            self.addKernel(position, energy)
            self.updateContext(simulation.context)
            if (
                self.saveFrequency is not None
                and simulation.currentStep % self.saveFrequency == 0
            ):
                self._syncWithDisk()
```

Add `from copy import copy` to the imports. `_finishWarmup` arrives in Task 10; add a stub that Task 10 replaces:

```python
    def _finishWarmup(self) -> None:
        raise NotImplementedError
```

Re-export from `src/openmm_opes/__init__.py`:

```python
"""OpenMM implementation of On-the-fly Probability Enhanced Sampling (OPES)."""

__version__ = "0.1.0"

__all__ = ["OPES", "__version__"]


def __getattr__(name):
    """Import OPES lazily (PEP 562).

    An eager ``from .opes import OPES`` would also run on
    ``import openmm_opes.kde``, dragging openmm in and breaking the guarantee
    that kde and io stay OpenMM-free. Task 1's test enforces that.
    """
    if name == "OPES":
        from .opes import OPES

        return OPES
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_opes.py -v`
Expected: PASS (21 tests)

- [ ] **Step 5: Lint and commit**

```bash
ruff format src tests && ruff check src tests && ty check src tests
git add src/openmm_opes tests/test_opes.py
git commit -m "feat: add OPES bias evaluation, deposition and stepping

Skips deposition when the variance estimate is not yet positive, which
previously produced a NaN bias that reached the forces."
```

---

### Task 10: Warm-up estimation of sigma^(0)

**Files:**
- Modify: `src/openmm_opes/opes.py`
- Modify: `tests/test_opes.py`

**Interfaces:**
- Consumes: Task 9's `OPES`.
- Produces: working `warmupSteps`; `_finishWarmup()`; `getState()`/`setState()` on `OPES` carrying `warmupComplete`.

Implements spec §7.7.

- [ ] **Step 1: Write the failing tests, appended to `tests/test_opes.py`**

```python
def runSteps(sampler, system, nsteps, seed=1234):
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 10.0 / unit.picosecond, 0.002 * unit.picoseconds
    )
    integrator.setRandomNumberSeed(seed)
    topology = app.Topology()
    topology.addAtom("A", None, topology.addResidue("M", topology.addChain()))
    simulation = app.Simulation(
        topology, system, integrator, openmm.Platform.getPlatformByName("Reference")
    )
    simulation.context.setPositions([openmm.Vec3(0.1, 0, 0)])
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, seed + 1)
    sampler.step(simulation, nsteps)
    return simulation


def makeHarmonicSystem():
    system = openmm.System()
    system.addParticle(12.0)
    potential = openmm.CustomExternalForce("500*x^2")
    potential.addParticle(0, [])
    system.addForce(potential)
    cv = openmm.CustomExternalForce("x")
    cv.addParticle(0, [])
    variable = app.BiasVariable(cv, -1.0, 1.0, 0.05, False, 51)
    return system, variable


def test_warmup_requires_variance_frequency():
    with pytest.raises(ValueError, match="warmupSteps requires varianceFrequency"):
        makeOPES(varianceFrequency=None, warmupSteps=100)


def test_warmup_must_span_at_least_two_intervals():
    with pytest.raises(ValueError, match="at least two varianceFrequency"):
        makeOPES(varianceFrequency=50, warmupSteps=50)


def test_no_kernels_are_deposited_during_warmup():
    system, variable = makeHarmonicSystem()
    sampler = OPES(system, [variable], 300.0, 20.0, 100, 10, warmupSteps=500)
    runSteps(sampler, system, 400)
    assert sampler.getNumKernels() == 0
    assert not sampler._warmupComplete


def test_deposition_starts_after_warmup_and_variance_freezes():
    system, variable = makeHarmonicSystem()
    sampler = OPES(system, [variable], 300.0, 20.0, 100, 10, warmupSteps=500)
    runSteps(sampler, system, 1500)
    assert sampler._warmupComplete
    assert sampler.getNumKernels() > 0
    frozen = sampler.getVariance().copy()
    runSteps(sampler, system, 500)
    assert sampler.getVariance() == pytest.approx(frozen)


def test_frozen_variance_is_gamma_times_the_measured_unbiased_variance():
    """Spec 7.7: warm-up measures an UNBIASED variance, while _variance holds
    a sampled one, so freezing stores gamma times the measurement."""
    system, variable = makeHarmonicSystem()
    sampler = OPES(system, [variable], 300.0, 20.0, 100, 10, warmupSteps=500)
    runSteps(sampler, system, 500)
    assert sampler._warmupComplete
    reweighted = sampler._kde["total.rw"]
    sampler.addKernel(np.array([0.0]), 0.0 * unit.kilojoules_per_mole)
    sigma0 = np.sqrt(sampler.getVariance() / sampler._biasFactor)
    # Silverman still applies to the very first kernel, where Neff == 1
    silverman = (1 * (1 + 2) / 4) ** (-1 / (1 + 4))
    assert reweighted._kernels[0].bandwidth[0] == pytest.approx(
        sigma0[0] * silverman
    )


def test_warmup_state_round_trips_without_rescaling_twice():
    system, variable = makeHarmonicSystem()
    sampler = OPES(system, [variable], 300.0, 20.0, 100, 10, warmupSteps=500)
    runSteps(sampler, system, 600)
    frozen = sampler.getVariance().copy()

    system2, variable2 = makeHarmonicSystem()
    restored = OPES(system2, [variable2], 300.0, 20.0, 100, 10, warmupSteps=500)
    restored.setState(sampler.getState())
    assert restored._warmupComplete
    assert restored.getVariance() == pytest.approx(frozen)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_opes.py -k warmup -v`
Expected: FAIL — `NotImplementedError` from the Task 9 stub

- [ ] **Step 3: Replace the `_finishWarmup` stub and add OPES state**

```python
    def _finishWarmup(self) -> None:
        """Freeze sigma^(0) measured from the unbiased warm-up segment.

        Nothing was deposited during warm-up, so the trajectory was unbiased
        and this measurement is the unbiased variance. _variance holds a
        sampled-distribution variance by convention, which is gamma times
        wider, so that is what gets stored. This is the same transformation
        the fixed-bandwidth path applies to biasWidth. See spec 7.5 and 7.7.
        """
        measured = self._variance["total"].get()
        if np.any(measured <= 0):
            raise ValueError(
                "Warm-up produced a non-positive CV variance. Increase "
                "warmupSteps or decrease varianceFrequency."
            )
        self._setFixedVariance(self._biasFactor * measured)
        self._warmupComplete = True

    def getState(self) -> dict:
        """Flat, npz-writable snapshot of this sampler's accumulated state."""
        state = self._getSharedState() if "self" in self._cases else {}
        for prefix, key in (("total", "total"), ("totalRW", "total.rw")):
            for name, value in self._kde[key].getState().items():
                state[f"{prefix}_{name}"] = value
        variance = self._variance["total"].getState()
        state["totalVar_num"] = variance["num"]
        state["totalVar_total"] = variance["total"]
        # Persisted explicitly rather than re-derived from the step counter:
        # re-deriving would risk rescaling an already-frozen variance by gamma
        # a second time on reload. See spec 7.7.
        state["warmupComplete"] = float(self._warmupComplete)
        return state

    def setState(self, state) -> None:
        """Restore from a :meth:`getState` snapshot."""
        for prefix, key in (("total", "total"), ("totalRW", "total.rw")):
            self._kde[key] = self._kdeFromState(state, prefix)
        average = RunningAverage(len(self.variables))
        average.setState(
            {"num": state["totalVar_num"], "total": state["totalVar_total"]}
        )
        self._variance["total"] = average
        self._warmupComplete = bool(float(state["warmupComplete"]))
        if self._warmupComplete and self.warmupSteps is not None:
            self._adaptiveVariance = False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_opes.py -v`
Expected: PASS (27 tests)

- [ ] **Step 5: Lint and commit**

```bash
ruff format src tests && ruff check src tests && ty check src tests
git add src/openmm_opes/opes.py tests/test_opes.py
git commit -m "feat: add warm-up estimation of the initial bandwidth

Automates the short unbiased run both papers prescribe for sigma^(0),
removes the sampled-versus-unbiased ambiguity, and guarantees adequate
statistics before the first kernel."
```

---

### Task 11: End-to-end integration tests

**Files:**
- Create: `tests/test_integration.py`

**Interfaces:**
- Consumes: the complete `OPES` from Tasks 8–10.
- Produces: nothing; this is the end-to-end correctness gate.

Implements spec §8.2. Tolerances come from the design review's own measured runs. **Do not re-tune them when a test fails** — that hides the regressions they exist to catch.

- [ ] **Step 1: Write the tests**

`tests/test_integration.py`:

```python
"""End-to-end runs on tiny analytic systems.

Reference numbers come from the design review (spec section 8.2): standard
OPES reached RMSE 0.78 kJ/mol at 1.2M steps with a barrier estimate of 24.43
against an analytic 25.00, and explore mode reached RMSE 3.00 with more
barrier crossings. Tolerances below are set from those runs and must not be
loosened to make a failing test pass.
"""

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import app, unit  # noqa: E402

from openmm_opes import OPES  # noqa: E402

BARRIER_HEIGHT = 25.0
TEMPERATURE = 300.0


def doubleWellSystem(gridWidth=151):
    """U(x) = 25*(1 - x^2)^2 kJ/mol: symmetric wells at x = +/-1."""
    system = openmm.System()
    system.addParticle(12.0)
    potential = openmm.CustomExternalForce(f"{BARRIER_HEIGHT}*(1-(x/1.0)^2)^2")
    potential.addParticle(0, [])
    system.addForce(potential)
    cv = openmm.CustomExternalForce("x")
    cv.addParticle(0, [])
    return system, app.BiasVariable(cv, -2.0, 2.0, 0.15, False, gridWidth)


def runSampler(sampler, system, chunks, chunkSize, seed=1234):
    integrator = openmm.LangevinMiddleIntegrator(
        TEMPERATURE * unit.kelvin, 10.0 / unit.picosecond, 0.002 * unit.picoseconds
    )
    integrator.setRandomNumberSeed(seed)
    topology = app.Topology()
    topology.addAtom("A", None, topology.addResidue("M", topology.addChain()))
    simulation = app.Simulation(
        topology, system, integrator, openmm.Platform.getPlatformByName("Reference")
    )
    simulation.context.setPositions([openmm.Vec3(-1.0, 0, 0)])
    simulation.context.setVelocitiesToTemperature(TEMPERATURE * unit.kelvin, seed + 1)
    trajectory = []
    for _ in range(chunks):
        sampler.step(simulation, chunkSize)
        trajectory.append(sampler.getCollectiveVariables(simulation)[0])
    return np.array(trajectory)


def fesError(sampler, gridWidth=151):
    grid = np.linspace(-2, 2, gridWidth)
    analytic = BARRIER_HEIGHT * (1 - grid**2) ** 2
    mask = analytic < 30.0
    fes = sampler.getFreeEnergy().value_in_unit(unit.kilojoules_per_mole)
    shifted = fes[mask] - fes[mask].min()
    reference = analytic[mask] - analytic[mask].min()
    rmse = np.sqrt(np.mean((shifted - reference) ** 2))
    barrier = shifted[np.argmin(np.abs(grid[mask]))]
    return rmse, barrier


@pytest.mark.slow
def test_opes_recovers_the_analytic_double_well():
    system, variable = doubleWellSystem()
    sampler = OPES(system, [variable], TEMPERATURE, 30.0, 200, 20)
    trajectory = runSampler(sampler, system, 600, 2000)

    assert trajectory.min() < -0.5 and trajectory.max() > 0.5
    rmse, barrier = fesError(sampler)
    assert rmse < 1.5, f"RMSE {rmse:.3f} kJ/mol (review run reached 0.78)"
    assert barrier == pytest.approx(BARRIER_HEIGHT, abs=2.5)


@pytest.mark.slow
def test_explore_mode_explores_more_but_converges_more_slowly():
    """The tradeoff the OPES-explore paper exists to demonstrate.

    Asserts the ordering, not the absolute values: the method guarantees the
    tradeoff, not any particular number.
    """
    results = {}
    for exploreMode in (False, True):
        system, variable = doubleWellSystem()
        sampler = OPES(
            system, [variable], TEMPERATURE, 30.0, 200, 20, exploreMode=exploreMode
        )
        trajectory = runSampler(sampler, system, 600, 2000)
        crossings = int(np.sum(np.diff(np.sign(trajectory)) != 0))
        results[exploreMode] = (crossings, fesError(sampler)[0])

    assert results[True][0] > results[False][0], "explore should cross more often"
    assert results[True][1] > results[False][1], "explore should converge more slowly"


def test_harmonic_free_energy_is_quadratic_where_sampled():
    system = openmm.System()
    system.addParticle(12.0)
    potential = openmm.CustomExternalForce("200*x^2")
    potential.addParticle(0, [])
    system.addForce(potential)
    cv = openmm.CustomExternalForce("x")
    cv.addParticle(0, [])
    variable = app.BiasVariable(cv, -1.0, 1.0, 0.05, False, 101)
    sampler = OPES(system, [variable], TEMPERATURE, 20.0, 200, 20)
    runSampler(sampler, system, 150, 2000)

    grid = np.linspace(-1, 1, 101)
    fes = sampler.getFreeEnergy().value_in_unit(unit.kilojoules_per_mole)
    core = np.abs(grid) < 0.25
    shifted = fes[core] - fes[core].min()
    analytic = 200 * grid[core] ** 2
    assert shifted == pytest.approx(analytic - analytic.min(), abs=4.0)


def test_bounded_and_compact_options_run_end_to_end():
    for kwargs in ({"bounded": True}, {"kernelShape": "compact"}):
        system, variable = doubleWellSystem(gridWidth=61)
        sampler = OPES(system, [variable], TEMPERATURE, 30.0, 200, 20, **kwargs)
        runSampler(sampler, system, 30, 1000)
        bias = sampler.getBias().value_in_unit(unit.kilojoules_per_mole)
        assert np.all(np.isfinite(bias)), kwargs
        assert sampler.getNumKernels() > 0, kwargs


def test_warmup_runs_end_to_end_and_beats_no_warmup_on_nan_safety():
    system, variable = doubleWellSystem(gridWidth=61)
    sampler = OPES(
        system, [variable], TEMPERATURE, 30.0, 200, 200, warmupSteps=2000
    )
    runSampler(sampler, system, 30, 1000)
    assert sampler._warmupComplete
    assert sampler.getNumKernels() > 0
    assert np.all(np.isfinite(sampler.getBias().value_in_unit(
        unit.kilojoules_per_mole)))


def test_two_walkers_share_their_kernels(tmp_path):
    samplers, systems = [], []
    for _ in range(2):
        system, variable = doubleWellSystem(gridWidth=61)
        samplers.append(
            OPES(
                system, [variable], TEMPERATURE, 30.0, 200, 20,
                saveFrequency=200, biasDir=str(tmp_path),
            )
        )
        systems.append(system)

    for index, (sampler, system) in enumerate(zip(samplers, systems)):
        runSampler(sampler, system, 20, 500, seed=1000 * (index + 1))

    # the second walker synced after the first had already written kernels
    assert samplers[1]._kde["total"].getNumKernels() > samplers[1]._kde[
        "self"
    ].getNumKernels()


def test_multiwalker_sync_does_not_corrupt_the_walkers_own_variance(tmp_path):
    """Regression for spec 7.4.1 at the level it actually bit.

    A peer must exist: with no peer, _syncWithDisk returns early and never
    reaches the copy-then-merge that the aliasing bug corrupted.
    """
    peerSystem, peerVariable = doubleWellSystem(gridWidth=61)
    peer = OPES(
        peerSystem, [peerVariable], TEMPERATURE, 30.0, 200, 20,
        saveFrequency=200, biasDir=str(tmp_path),
    )
    runSampler(peer, peerSystem, 10, 500, seed=99)

    system, variable = doubleWellSystem(gridWidth=61)
    sampler = OPES(
        system, [variable], TEMPERATURE, 30.0, 200, 20,
        saveFrequency=200, biasDir=str(tmp_path),
    )
    runSampler(sampler, system, 10, 500)

    own = sampler._variance["self"].get().copy()
    sampler._syncWithDisk()
    assert sampler._variance["self"].get() == pytest.approx(own)
```

- [ ] **Step 2: Register the `slow` marker in `pyproject.toml`**

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: end-to-end simulations that take minutes"]
```

- [ ] **Step 3: Run the fast integration tests**

Run: `pytest tests/test_integration.py -v -m "not slow"`
Expected: PASS (5 tests), in well under a minute.

- [ ] **Step 4: Run the slow tests**

Run: `pytest tests/test_integration.py -v -m slow`
Expected: PASS (2 tests). Roughly 30–60 s combined on the Reference platform. If the RMSE assertion fails, the port has a real convergence regression — debug it; do not raise the threshold.

- [ ] **Step 5: Commit**

```bash
ruff format src tests && ruff check src tests
git add tests/test_integration.py pyproject.toml
git commit -m "test: add end-to-end convergence and multi-walker tests"
```

---

### Task 12: Conda-based test CI

**Files:**
- Modify: `.github/workflows/ci.yml`
- Create: `environment.yml`

**Interfaces:**
- Consumes: the full test suite.
- Produces: a `test` job running everything with OpenMM from conda-forge.

- [ ] **Step 1: Write `environment.yml`**

```yaml
name: openmm-opes
channels:
  - conda-forge
dependencies:
  - python
  - numpy>=1.24
  - scipy>=1.10
  - openmm>=8.1
  - pytest>=7.4
  - setuptools>=68
  - pip
```

- [ ] **Step 2: Add the `test` job to `.github/workflows/ci.yml`**

```yaml
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.13"]
    defaults:
      run:
        shell: bash -el {0}
    steps:
      - uses: actions/checkout@v4
      # OpenMM's canonical distribution channel is conda-forge, not PyPI.
      - uses: mamba-org/setup-micromamba@v1
        with:
          environment-file: environment.yml
          create-args: python=${{ matrix.python-version }}
          cache-environment: true
      - run: pip install --no-deps -e .
      - run: pytest -v
```

- [ ] **Step 3: Verify the whole suite passes locally**

Run: `pytest -v`
Expected: PASS, all tests including slow ones.

- [ ] **Step 4: Commit and confirm CI is green**

```bash
git add .github/workflows/ci.yml environment.yml
git commit -m "ci: run the full suite against conda-forge OpenMM"
git push
```

Check the Actions tab. Both `lint` and `test` (×2 Python versions) must pass before continuing.

---

### Task 13: Documentation site and README

**Files:**
- Create: `mkdocs.yml`, `docs/index.md`, `docs/quickstart.md`, `docs/theory.md`, `docs/api.md`
- Modify: `README.md`, `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: the finished public API.
- Produces: a published documentation site.

- [ ] **Step 1: Write `mkdocs.yml`**

```yaml
site_name: openmm-opes
site_description: OpenMM implementation of On-the-fly Probability Enhanced Sampling
repo_url: https://github.com/craabreu/openmm-opes

theme:
  name: material
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      toggle: { icon: material/weather-night, name: Dark mode }
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      toggle: { icon: material/weather-sunny, name: Light mode }

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: numpy
            show_source: false

markdown_extensions:
  - admonition
  - pymdownx.arithmatex: { generic: true }
  - pymdownx.highlight
  - pymdownx.superfences

extra_javascript:
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js

# Design documents live under docs/ but are not part of the published site.
exclude_docs: |
  superpowers/

nav:
  - Home: index.md
  - Quickstart: quickstart.md
  - Theory: theory.md
  - API reference: api.md
```

- [ ] **Step 2: Write `docs/index.md`**

```markdown
# openmm-opes

An [OpenMM](https://openmm.org) implementation of **On-the-fly Probability
Enhanced Sampling** (OPES) and its exploratory variant, OPES-explore.

OPES builds a bias potential from an on-the-fly estimate of the probability
distribution along a set of collective variables, rather than accumulating
repulsive hills the way metadynamics does. It needs only three parameters:
the deposition pace, the initial kernel bandwidth, and the approximate height
of the barrier to overcome.

## Installation

OpenMM is distributed through conda-forge, so that is the supported route:

```bash
mamba install -c conda-forge openmm
pip install openmm-opes
```

## At a glance

```python
from openmm import app
from openmm_opes import OPES

sampler = OPES(
    system, [phi, psi], 300 * unit.kelvin, 40 * unit.kilojoules_per_mole,
    frequency=500, varianceFrequency=50,
)
sampler.step(simulation, 1_000_000)
fes = sampler.getFreeEnergy()
```

See the [quickstart](quickstart.md) for a complete runnable example, and the
[theory](theory.md) page for how the code maps onto the published equations.

## References

1. Invernizzi & Parrinello, *Rethinking Metadynamics: From Bias Potentials to
   Probability Distributions*, J. Phys. Chem. Lett. 2020.
   [doi:10.1021/acs.jpclett.0c00497](https://doi.org/10.1021/acs.jpclett.0c00497)
2. Invernizzi & Parrinello, *Exploration vs Convergence Speed in Adaptive-Bias
   Enhanced Sampling*, J. Chem. Theory Comput. 2022.
   [doi:10.1021/acs.jctc.2c00152](https://doi.org/10.1021/acs.jctc.2c00152)
```

- [ ] **Step 3: Write `docs/quickstart.md`**

```markdown
# Quickstart

A single particle on a two-dimensional Müller-Brown potential, biased along
`x` and `y`, recovering the free energy surface.

```python
import numpy as np
import openmm as mm
from openmm import app, unit

from openmm_opes import OPES

KB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
    unit.kilojoules_per_mole / unit.kelvin
)

muller_brown = (
    "-200*exp(-(x-1)^2-10*y^2)"
    " -100*exp(-x^2-10*(y-0.5)^2)"
    " -170*exp(-6.5*(0.5+x)^2+11*(x+0.5)*(y-1.5)-6.5*(y-1.5)^2)"
    " +15*exp(0.7*(1+x)^2+0.6*(x+1)*(y-1)+0.7*(y-1)^2)"
)

system = mm.System()
system.addParticle(1.0)
potential = mm.CustomExternalForce(muller_brown)
potential.addParticle(0, [])
system.addForce(potential)

variables = []
for expression, lo, hi in (("x", -1.5, 1.2), ("y", -0.2, 2.0)):
    force = mm.CustomExternalForce(expression)
    force.addParticle(0, [])
    variables.append(app.BiasVariable(force, lo, hi, 0.1, False, 101))

temperature = 1.0 / KB * unit.kelvin
sampler = OPES(
    system,
    variables,
    temperature,
    barrier=20 * unit.kilojoules_per_mole,
    frequency=500,
    varianceFrequency=50,
)

topology = app.Topology()
topology.addAtom("P", None, topology.addResidue("MOL", topology.addChain()))
integrator = mm.LangevinMiddleIntegrator(
    temperature, 10.0 / unit.picosecond, 0.005 * unit.picoseconds
)
simulation = app.Simulation(
    topology, system, integrator, mm.Platform.getPlatformByName("Reference")
)
simulation.context.setPositions([mm.Vec3(-0.5, 1.4, 0.0)])
simulation.context.setVelocitiesToTemperature(temperature)

sampler.step(simulation, 2_000_000)

fes = sampler.getFreeEnergy().value_in_unit(unit.kilojoules_per_mole)
fes -= fes.min()
print(f"{sampler.getNumKernels()} kernels, barrier {fes.max():.1f} kJ/mol")
```

`getFreeEnergy()` returns an array whose shape is the CV grid with the **last**
variable varying fastest, so for two CVs it has shape `(ny, nx)` and can be
handed straight to `matplotlib.pyplot.contourf`.

## Choosing the bandwidth

Both papers prescribe measuring the initial bandwidth from a short unbiased
run. You can let the sampler do that for you:

```python
sampler = OPES(..., varianceFrequency=50, warmupSteps=50_000)
```

Nothing is deposited during the first 50,000 steps; the CV variance measured
there becomes the initial bandwidth and is then held fixed.
```

- [ ] **Step 4: Write `docs/theory.md`**

```markdown
# The method as implemented

Notation follows Invernizzi and Parrinello. $\beta = 1/k_BT$, $\gamma$ is the
bias factor, and $\mathbf{s}$ the collective variables.

## The probability estimate

OPES estimates the unbiased distribution by weighted kernel density estimation,

$$P_n(\mathbf{s}) = \frac{\sum_k^n w_k\, G(\mathbf{s}, \mathbf{s}_k)}{\sum_k^n w_k},
\qquad w_k = e^{\beta V_{k-1}(\mathbf{s}_k)}$$

with Gaussian kernels of fixed height $h = \prod_i (\sigma_i\sqrt{2\pi})^{-1}$.
This is `OnlineKDE.getLogPDF()`; the weight is the bias energy read from the
OPES force group *before* the new kernel is deposited, so it really is
$V_{k-1}$.

## Bandwidth

Bandwidths shrink as the effective sample size
$N_{\text{eff}} = (\sum_k w_k)^2 / \sum_k w_k^2$ grows, by Silverman's rule:

$$\sigma_i^{(n)} = \sigma_i^{(0)}\left[N_{\text{eff}}^{(n)}(d+2)/4\right]^{-1/(d+4)}$$

## Normalization

$Z_n$ normalizes over the CV space explored so far, and is approximated by a
sum over the compressed kernel centers — `OnlineKDE.getLogMeanDensity()`.

$$Z_n = \frac{1}{|\Omega_n|}\int_{\Omega_n} P_n(\mathbf{s})\, d\mathbf{s}$$

## The bias

$$V_n(\mathbf{s}) = \left(1 - \tfrac{1}{\gamma}\right)\frac{1}{\beta}
\log\left(\frac{P_n(\mathbf{s})}{Z_n} + \epsilon\right)$$

with $\epsilon = e^{-\beta\Delta E/(1-1/\gamma)}$ limiting the bias to the
barrier $\Delta E$. This is `OPES.getBias()`.

## OPES-explore

The explore variant estimates the *sampled* distribution instead, with uniform
weights,

$$p^{\text{WT}}_n(\mathbf{s}) = \frac{1}{n}\sum_k^n G(\mathbf{s}, \mathbf{s}_k),
\qquad
V_n(\mathbf{s}) = (\gamma-1)\frac{1}{\beta}
\log\left(\frac{p^{\text{WT}}_n(\mathbf{s})}{Z_n} + \epsilon\right)$$

It explores faster and converges more slowly. `getBias()` reads the unweighted
estimate in explore mode and the reweighted one otherwise.

## Free energy

`getFreeEnergy()` always uses the importance-sampling estimate,
$F_n = -\beta^{-1}\log P_n$. In standard OPES the direct and reweighted routes
are equivalent; in explore mode they differ until convergence, and the
reweighted one converges better.

## Kernel compression

Rather than storing a bias grid, kernels closer than `compressionThreshold` in
Mahalanobis distance are merged, preserving total weight, mean and second
moment. The number of compressed kernels is what makes $|\Omega_n|$ estimable.
```

- [ ] **Step 5: Write `docs/api.md`**

```markdown
# API reference

::: openmm_opes.opes.OPES

::: openmm_opes.kde.OnlineKDE

::: openmm_opes.kde.CVSpace

::: openmm_opes.kde.Kernel
```

- [ ] **Step 6: Rewrite `README.md`**

```markdown
# openmm-opes

[![CI](https://github.com/craabreu/openmm-opes/actions/workflows/ci.yml/badge.svg)](https://github.com/craabreu/openmm-opes/actions/workflows/ci.yml)

An [OpenMM](https://openmm.org) implementation of On-the-fly Probability
Enhanced Sampling (OPES) and OPES-explore.

## Installation

```bash
mamba install -c conda-forge openmm
pip install openmm-opes
```

## Usage

```python
from openmm_opes import OPES

sampler = OPES(system, variables, temperature, barrier, frequency=500,
               varianceFrequency=50)
sampler.step(simulation, 1_000_000)
fes = sampler.getFreeEnergy()
```

Full documentation: <https://craabreu.github.io/openmm-opes>

## References

- Invernizzi & Parrinello, J. Phys. Chem. Lett. 2020,
  [doi:10.1021/acs.jpclett.0c00497](https://doi.org/10.1021/acs.jpclett.0c00497)
- Invernizzi & Parrinello, J. Chem. Theory Comput. 2022,
  [doi:10.1021/acs.jctc.2c00152](https://doi.org/10.1021/acs.jctc.2c00152)

## License

MIT
```

- [ ] **Step 7: Add the docs job to `.github/workflows/ci.yml`**

```yaml
  docs:
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      # mkdocstrings reads the source statically via griffe, so the docs build
      # needs neither openmm nor an importable package at runtime.
      - run: pip install mkdocs-material "mkdocstrings[python]"
      - run: pip install --no-deps -e .
      - run: mkdocs gh-deploy --force
```

- [ ] **Step 8: Build the docs locally**

Run:
```bash
pip install mkdocs-material "mkdocstrings[python]"
mkdocs build --strict
```
Expected: builds with no warnings. Confirm `site/` contains no `superpowers/` directory — the spec and plan must not be published.

- [ ] **Step 9: Commit**

```bash
git add mkdocs.yml docs README.md .github/workflows/ci.yml
git commit -m "docs: add mkdocs-material site and rewrite the README"
git push
```

Then enable GitHub Pages for the repository, serving from the `gh-pages` branch.

---

## Self-Review

**Spec coverage.** §2 scope → Tasks 2–11. §3 packaging → Task 1. §4 `kde.py` → Tasks 2–5. §5 `opes.py` → Tasks 8–10. §6 `io.py` → Task 7. §7.1/§7.2 removals → not carried in by construction: no task ports labels, gradients, `getSmoothedCopy`, or the uncompressed mode. §7.3 options → Tasks 4 and 8. §7.4 bugs 1–4 → Tasks 8, 4, 5, 9 respectively, each with a named regression test. §7.5 `biasWidth` → Task 8. §7.6 OpenMM conformance → Task 8 (explicit table branch, three validation fixes, type hints throughout). §7.7 warm-up → Task 10. §8.1 unit tests → Tasks 2–5, 7. §8.2 integration → Task 11. §8.3 parity → Task 6. §9 CI → Tasks 1 and 12. §10 docs → Task 13. §11 deliverables → all tasks; item 1 was completed before planning. §12.4 invariant → Task 4's margin test plus the `_removeKernels` comment.

**Type consistency.** `getState`/`setState` are spelled identically on `OnlineKDE` (Task 5), `RunningAverage` (Task 8) and `OPES` (Task 10). `KERNEL_SHAPES` keys `"gaussian"`/`"compact"` match the `kernelShape` argument in Tasks 4, 8 and 11. `CVSpace.CV` is used by `tests/helpers.py` (Task 2) and `make_reference.py` (Task 6, against the source's `_CV`). `_finishWarmup` is stubbed in Task 9 and implemented in Task 10. `BiasSharer.load()` returns only *changed* peers while `getLoadedStates()` returns all, and `_syncWithDisk` (Task 9) uses `load()` as a change flag and `getLoadedStates()` to rebuild — consistent.

**Defects this review caught and fixed in place.** Recorded so they are not
reintroduced: `np.trapezoid` needs numpy 2.0 while the floor is 1.24 (replaced with a
Riemann sum); `compressionThreshold=0` still merged coincident kernels because
`0 <= 0` (added a `threshold > 0` guard); the `__iadd__` test assumed Silverman is
re-applied on transfer, which it deliberately is not (rewritten against the union of
kernels); the frozen-variance test forgot that Silverman still scales the very first
kernel at `N_eff == 1`; the multi-walker aliasing regression had no peer, so
`_syncWithDisk` returned early and never reached the buggy path; and an eager
`from .opes import OPES` in `__init__.py` would have dragged openmm into
`import openmm_opes.kde`, breaking Task 1's own guarantee (now a PEP 562 lazy import).

**Known ordering note for the executor.** Task 9's `_syncWithDisk` calls `_kdeFromState`, and Task 9's `getState` stub interplay with Task 10 means `pytest tests/test_opes.py -k warmup` fails with `NotImplementedError` until Task 10 lands. That is intended and is what Task 10's Step 2 expects.
