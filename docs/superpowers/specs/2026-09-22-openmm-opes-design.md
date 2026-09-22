# openmm-opes: design

Date: 2026-09-22
Status: approved, ready for implementation planning

## 1. Purpose

Turn the working research implementation of On-the-fly Probability Enhanced Sampling
(OPES) — currently two loose modules, `opes.py` and `online_kde.py`, in the
`opes-simulations` repository — into a properly packaged, tested, documented Python
library that extends OpenMM.

The method follows Invernizzi and Parrinello:

1. *Rethinking Metadynamics: From Bias Potentials to Probability Distributions*,
   <https://doi.org/10.1021/acs.jpclett.0c00497> (OPES)
2. *Exploration vs Convergence Speed in Adaptive-Bias Enhanced Sampling*,
   <https://doi.org/10.1021/acs.jctc.2c00152> (OPES-Explore)

The library is a faithful port, not a redesign. Sampling behavior under default
settings must be unchanged from the source implementation (see §8.3, parity test).
Packaging, linting, typing, CI, and documentation conventions follow the `sieve`
repository.

## 2. Scope

### In scope for v1

- OPES and OPES-Explore bias, integrated with OpenMM via `CustomCVForce` and a
  tabulated function, for 1–3 collective variables.
- Online kernel density estimation with kernel compression and merging.
- Adaptive bandwidth (Silverman) and adaptive CV variance estimation.
- An optional warm-up phase that measures `sigma^(0)` from an initial unbiased segment
  and then freezes it (§7.7) — the one feature added rather than ported.
- Periodic and non-periodic CVs; reflective ("bounded") grid folding.
- Multi-walker bias sharing through a shared directory.
- Experimental variants promoted from module-level globals to constructor options
  (§7.3).
- Two-tier test suite, conda-based CI, mkdocs-material documentation site.

### Out of scope for v1

Deferred deliberately; none of it is used by any current experiment script, and all of
it belongs to the separate `opes-cv-learning` research line, whose own notes describe
it as exploratory and untested:

- State labeling and the recollector estimator (`stateIDFuncs`, per-kernel label
  fractions, `getRecollectorVariable`, `getRecollectors`).
- Gradient calculations (`getLogPDFGradients`, `getRecollectorGradients`,
  `Kernel.evaluateDirectionsOnGrid`).
- KDE smoothing (`getSmoothedCopy`).
- The uncompressed-KDE storage mode.
- Any coupling to learned collective variables or uniform recollector sampling.

These are removed rather than carried as dead code. The public API is designed so they
can be layered back on later without breaking it: labels re-enter as an optional
`numLabels`/`fractions` extension to `Kernel` and `OnlineKDE`, and gradients as
additional methods on `OnlineKDE`.

## 3. Repository and packaging

The existing `opesmm` repository (currently containing only `README.md`, `LICENSE`,
`.gitignore`) becomes the home of the library and is renamed:

- GitHub: `craabreu/opesmm` → `craabreu/openmm-opes`. **Done**; the local remote has been
  updated to the new URL and verified.
- Local working copy: `/home/charlles/opes/opesmm` → `/home/charlles/opes/openmm-opes`.
  **Done.**
- PyPI distribution name: `openmm-opes`.
- Python import name: `openmm_opes`.

Build backend is setuptools with a `src/` layout, matching `sieve`. Python floor is
3.11. License is MIT, already present.

```
openmm-opes/
  pyproject.toml
  README.md
  LICENSE
  mkdocs.yml
  src/openmm_opes/
    __init__.py
    opes.py
    kde.py
    io.py
  tests/
    __init__.py
    helpers.py
    test_cvspace.py
    test_kernel.py
    test_kde.py
    test_opes.py
    test_io.py
    test_parity.py
    data/reference_kde.npz
    data/make_reference.py
  docs/
    index.md
    quickstart.md
    theory.md
    api.md
  .github/workflows/ci.yml
```

`pyproject.toml` essentials:

- `dependencies = ["numpy>=1.24", "scipy>=1.10", "openmm>=8.1"]`. OpenMM is declared as
  a real dependency even though it is normally installed from conda-forge; declaring it
  keeps the metadata honest, and the documented install path is conda/mamba.
- `[project.optional-dependencies] dev = ["pytest>=7.4", "ruff>=0.16", "ty==0.0.72"]`
  and `docs = ["mkdocs-material", "mkdocstrings[python]"]`.
- `[tool.ruff]` with `line-length = 88`, `target-version = "py311"`, and the same lint
  selection as `sieve`: `["E", "F", "I", "UP", "B", "C4", "RUF"]`.
- `[tool.pytest.ini_options] testpaths = ["tests"]`.
- `[tool.ty.analysis]` with `allowed-unresolved-imports` for `openmm`/`openmm.**`, since
  OpenMM ships no type stubs and the lint CI job installs it without conda.

## 4. Module: `kde.py`

Holds everything that is pure numerical machinery and has no OpenMM dependency. This
separation is what lets the bulk of the test suite run without OpenMM installed.

### 4.1 `CVSpace`

Ported essentially unchanged. Represents the CV domain: grid, periodicity, and the
optional reflective folding used when `bounded=True`.

```
CVSpace(variables, bounded=False)
  .gridShape            -> tuple
  .numDimensions        -> int
  .displacement(position, endpoint)
  .endpoint(position, displacement)
  .gridDistances(position)
  .closestNode(position)
  .foldedGrid(values)
```

`variables` are `openmm.app.BiasVariable` instances — the same CV description type
metadynamics uses. No new CV abstraction is introduced. `CVSpace` reads only
`minValue`, `maxValue`, `gridWidth`, `periodic` from them and stores a plain internal
namedtuple, so it remains picklable and OpenMM-free after construction.

### 4.2 Kernel shapes

The `BOUNDED_KERNELS` global becomes a selectable kernel shape. It is named
`compactKernels` rather than anything containing "bounded", because the existing
`bounded=` argument means reflective grid folding — an unrelated concept that would
otherwise be confusable.

Two shapes, implemented as small module-level strategy objects with two members each, a
per-dimension log normalization constant and an exponent function:

- Gaussian (default): `logNorm = log(2*pi)/2`, `exponents(x) = -x**2/2`.
- Compact: `logNorm = log(559872/35)`, `exponents(x) = 4*log(9 - x**2)` where
  `|x| < 3`, and `-inf` elsewhere. Support is +/- 3 bandwidths.

`Kernel` holds a reference to its shape; `OnlineKDE` selects it once and passes it down.
Pickled state stores the shape's name string, not the object, so the on-disk format
stays stable and readable.

### 4.3 `Kernel`

```
Kernel(cvSpace, position, bandwidth, logWeight, numSamples=1, shape=GAUSSIAN)
  .logHeight            (derived)
  .findNearest(centers, bandwidths, ignore=())
  .merge(other)
  .evaluate(points)
  .evaluateOnGrid()
```

Changes from source: the `fractions` argument and `_logFraction`, the `label` arguments
on `evaluate`/`evaluateOnGrid`, and `evaluateDirectionsOnGrid` are all removed with the
recollector feature. The `BOUNDED_KERNELS` branches in `_computeLogHeight` and
`_exponents` are replaced by the shape strategy.

`merge` keeps the source's weighted-moment formulas exactly: log-sum-exp of weights,
weighted displacement of the center, and

```
bandwidth = sqrt(w1*b1^2 + w2*b2^2 + w1*w2*disp^2)
```

with `numSamples` accumulating.

### 4.4 `OnlineKDE`

```
OnlineKDE(cvSpace, compressionThreshold=1.0, useExistingBandwidths=True,
          kernelShape="gaussian")
  .update(position, logWeight, variance)
  .evaluate(point)
  .getLogPDF()
  .getLogMeanDensity()
  .getNumKernels()
  __iadd__, __copy__, __bool__, __getstate__, __setstate__
```

Maintains the compressed kernel list plus two running log-sums (`logSumW`,
`logSumWSq`), the log PDF on the grid (`logPG`), and the log PDF evaluated at the
kernel centers (`logPK`, needed for the mean density). Deposition applies the Silverman
factor from the effective sample size `neff = exp(2*logSumW - logSumWSq)`:

```
silverman = (neff * (d + 2) / 4) ** (-1 / (d + 4))
```

then merges the new kernel into every existing kernel closer than
`compressionThreshold` in scaled (Mahalanobis) distance, repeating until no neighbor
qualifies.

All `UNCOMPRESSED_KDE` branches and the `_maskPG` grid mask are removed, which
simplifies `__iadd__`, `__getstate__`, `__setstate__`, `evaluate`, and
`getLogMeanDensity` to a single path each. `numLabels`, per-kernel `fractions`, and the
`label` parameters are removed with the recollector feature, as are `getSmoothedCopy`,
`getLogPDFGradients`, and `getRecollector*`.

`__getstate__` stores kernel data as stacked arrays (positions, bandwidths, log weights,
sample counts) plus the scalar log-sums and `logPK`; `logPG` is not stored and is
rebuilt on load.

## 5. Module: `opes.py`

The OpenMM-facing bias class. One public class.

```
OPES(system, variables, temperature, barrier, frequency, varianceFrequency,
     biasFactor=None, exploreMode=False, bounded=False,
     saveFrequency=None, biasDir=None, warmupSteps=None,
     compressionThreshold=1.0, useExistingBandwidths=True,
     kernelShape="gaussian", statsWindowSize=10)

  .step(simulation, steps)
  .getCollectiveVariables(simulation)
  .getFreeEnergy()
  .getBias()
  .getAverageDensity()
  .getVariance()
  .getNumKernels()
  .addKernel(values, biasEnergy, variance=None)
  .updateContext(context)
```

Behavior is carried over unchanged:

- Constructor computes `kbt`, resolves `biasFactor` (defaulting to `barrier / kbt`),
  and forms `prefactor = (1 - 1/biasFactor) * kbt`, multiplied by `biasFactor` again in
  explore mode, with `logEpsilon = -barrier / prefactor`.
- It builds a `CustomCVForce` whose energy is a `ContinuousNDFunction` table over the
  CV grid, registers each variable's force, assigns the highest free force group, and
  adds the force to the system.
- `step` advances the simulation in chunks aligned to `varianceFrequency` (or
  `frequency` when adaptive variance is off), updating sample statistics at each
  interval, depositing a kernel and refreshing the tabulated bias every `frequency`
  steps, and syncing with disk every `saveFrequency` steps.
- Two KDEs are maintained per case, the plain one and the reweighted (`.rw`) one. The
  reweighted KDE receives `biasEnergy / kbt` as log weight and `variance / biasFactor`
  as variance. `getFreeEnergy` always reads the reweighted KDE; `getBias` and
  `getAverageDensity` read the plain KDE in explore mode and the reweighted one
  otherwise.
- `_cases` is `("total",)`, plus `("self",)` when multi-walker saving is on.

Validation in the constructor keeps every existing check: `varianceFrequency` must
divide `frequency`, `saveFrequency` and `biasDir` must be given together,
`saveFrequency` must be a multiple of `frequency`, `biasFactor` must exceed 1, CVs must
be uniformly periodic or uniformly non-periodic, dimensionality must be 1–3, and a free
force group must exist. `stateIDFuncs` and its callability check are removed.

`STATS_WINDOW_SIZE` becomes the `statsWindowSize` constructor argument (default 10,
unchanged), which sets the running-average window
`tau = statsWindowSize * frequency // varianceFrequency`.

`RunningAverage` moves here (it is about CV statistics, not kernel density) with the
aliasing bug in §7.4 fixed.

## 6. Module: `io.py`

Multi-walker bias sharing, lifted out of `OPES._syncWithDisk` so the bias class does not
also own a file protocol.

The protocol is unchanged: each walker writes its own KDE, its own reweighted KDE, and
its own variance accumulator to `kde_<walkerId>_<index>`, written to a temporary name
and then renamed for atomicity, with the previous index deleted afterward. On each sync
a walker rescans the directory, loads any peer file whose index advanced, and rebuilds
its `total` state as its own state plus every peer's.

The **serialization format changes from pickle to `np.savez`**. The source pickles live
objects, which is wrong for a published library on three counts:

1. *It executes arbitrary code on load.* In multi-walker mode a walker unpickles files
   written by other processes out of a shared directory. `pickle.load` on data from
   another process is a remote-code-execution primitive. OpenMM's own `Metadynamics`
   writes `.npy` for exactly this reason.
2. *It embeds module paths.* Confirmed by disassembling a saved file: the pickle carries
   `online_kde` and `opes` as module names, so every bias file written by the current
   code raises `ModuleNotFoundError: No module named 'online_kde'` once the modules are
   renamed to `openmm_opes.kde` and `openmm_opes.opes`. A pickle format cannot survive
   this port regardless, so "keep the format unchanged" was never actually available.
3. *It is fragile against refactoring.* Any change to class layout can break old files
   silently or loudly.

`OnlineKDE.__getstate__` already produces exactly stacked arrays plus scalars, so the
change is mechanical: write those arrays and scalars with `np.savez`, read them back
with `np.load(..., allow_pickle=False)`. The result is safe, inspectable, and stable
across refactors. Old `.pkl` files are not migrated; the extension changes to `.npz` so
a stale directory cannot be half-read.

`io.py` also defines its own `_LoadedBias` equivalent namedtuple rather than importing
the private `openmm.app.metadynamics._LoadedBias`, removing a dependency on OpenMM
internals.

The module exposes a small `BiasSharer` (holding `biasDir`, the walker id, the save
index, and the loaded peer state) with `save(state)` and `load()`, so it can be tested
against a `tmp_path` with no simulation running.

## 7. Changes from the source implementation

### 7.1 Removed as dead code

`CORRECTED_OPES_EXPLORE` and `USE_PDF_OPES_EXPLORE` are assigned in `opes.py` and never
read anywhere. They are deleted.

### 7.2 Removed as deferred features

The uncompressed-KDE mode, the recollector/labeling machinery, the gradient methods, and
`getSmoothedCopy` — see §2.

### 7.3 Promoted from globals to constructor options

| Source global | New option | Default |
|---|---|---|
| `COMPRESSION_THRESHOLD` | `compressionThreshold` | `1.0` |
| `USE_EXISTING_BANDWIDTHS` | `useExistingBandwidths` | `True` |
| `BOUNDED_KERNELS` | `kernelShape` (`"gaussian"` / `"compact"`) | `"gaussian"` |
| `STATS_WINDOW_SIZE` | `statsWindowSize` | `10` |

Every default reproduces the source's current value, so default behavior is unchanged.
`OPES` accepts these and forwards them to the KDEs it owns.

### 7.4 Bugs found while reading, fixed here

Each gets a regression test named for the behavior it protects.

1. **`RunningAverage` copies share their accumulator array.** `__setstate__` assigns
   `self._total = state["total"]` without copying, so a `copy.copy` of a
   `RunningAverage` aliases the original's array. `_syncWithDisk` then does
   `self._variance["total"] = copy(self._variance["self"])` followed by `+=`, and
   `__iadd__`'s in-place `self._total += other._total` silently corrupts the walker's
   own variance accumulator. This affects every multi-walker run. Fixed by copying the
   array in `__setstate__`.
2. **`OnlineKDE.__bool__` references a non-existent attribute.** It returns
   `self._numSamples > 0`, but `_numSamples` is never assigned on `OnlineKDE`, so truth
   testing a KDE raises `AttributeError`. Fixed to report whether any kernel exists.
3. **`OnlineKDE.__setstate__` cannot restore an empty KDE.** It rebuilds `logPG` with
   `functools.reduce(np.logaddexp, ...)` over the kernels with no initial value, which
   raises `TypeError` on an empty sequence — reachable when a walker syncs before
   depositing its first kernel. Fixed by seeding the reduction with a `-inf` grid.
4. **`varianceFrequency == frequency` silently produces a NaN bias.** Validation accepts
   it (it only requires that `varianceFrequency` divide `frequency`), but it leaves
   exactly one sample-statistics update before the first deposition. The Welford update
   yields `sqdev == 0` for a single sample, so the first kernel is built with a zero
   bandwidth, giving `logHeight = -inf`. Its weight is still added to `logSumW`, so
   `getLogPDF()` and `getLogMeanDensity()` are both `-inf`, their difference in
   `getBias` is `NaN`, and the `NaN` is written straight into the tabulated function and
   thence into the forces. Confirmed by direct reproduction. Fixed by skipping
   deposition while the variance estimate is not strictly positive in every dimension,
   and warning — this changes behavior only in the configuration that is currently
   broken, so the §8.3 parity test is unaffected.

   **This fix was incomplete, and shipped broken in 0.1.0.** Skipping the deposition
   left `_onInterval` calling `updateContext` unconditionally straight afterwards, so
   the NaN reached the forces anyway by the other route. The accompanying test asserted
   only that no kernel had been deposited, then checked the bias *after* a healthy
   kernel had been added — it never looked at the bias during the broken window, so it
   passed against code that still destroyed the trajectory. Corrected in 0.1.1: see
   §7.8 finding 1.

### 7.5 What `biasWidth` means on the fixed-bandwidth path

Throughout the implementation, `self._variance` holds the variance of the **sampled**
(biased) distribution. `addKernel` therefore passes `variance` to the plain KDE, which
estimates the sampled distribution, and `variance / biasFactor` to the reweighted KDE,
which estimates the unbiased one — correct, because the well-tempered sampled
distribution `P^(1/gamma)` is wider than `P` by a factor `sqrt(gamma)`.

With adaptive variance this is self-consistent: the running variance really is measured
from the biased trajectory. With `varianceFrequency=None` it is not. That path seeds
the accumulator from the user's `BiasVariable.biasWidth`, which implicitly declares
`biasWidth` to be the width of the *sampled* distribution — whereas both papers define
`sigma^(0)` as "the initial standard deviation estimated from a short unbiased
simulation", and OpenMM's own metadynamics uses `biasWidth` in that same unbiased sense.

Confirmed by direct measurement: with `biasWidth = 0.1` and `gamma = 8.02`, the
reweighted KDE — the KDE that defines both the bias and the free energy in standard
OPES — builds its kernels with `sigma = 0.0374`, i.e. `biasWidth / sqrt(gamma)`. At the
`gamma = 20` used in the Müller-Brown runs the discrepancy is a factor of 4.5.

No current experiment script exercises this path (all of them pass a
`varianceFrequency`), so no existing result is affected.

**Decided (confirmed by the author): take the papers' convention.** `biasWidth` is the
unbiased `sigma^(0)`, so the fixed path seeds the accumulator with
`biasFactor * biasWidth**2`. The reweighted KDE then builds its kernels with `biasWidth`
and the plain KDE with `sqrt(biasFactor) * biasWidth`, each correct for the distribution
it estimates, and `biasWidth` now means the same thing here as it does in
`openmm.app.Metadynamics`. This is a deliberate behavior change on a path with no users;
it is called out in the `OPES` docstring and in the changelog, and covered by a test that
asserts the reweighted KDE's first kernel has bandwidth `biasWidth` when
`varianceFrequency is None`.

### 7.6 Conformance to OpenMM conventions

The source is already a close, deliberate mirror of `openmm.app.Metadynamics`, and the
port keeps it that way: same constructor shape, same unit-coercion idiom, same
`CustomCVForce` plus tabulated-function construction, same `_widths`/`_limits`, same
mixed-periodicity rejection, same `max(freeGroups)` force-group selection and
`RuntimeError`, same `system.addForce` side effect, same `step(simulation, steps)`
chunking, same `getFreeEnergy()` contract (an N-dimensional `Quantity` array in
kJ/mol), same `getCollectiveVariables(simulation)`, and the same atomic
temp-write-rename-delete disk protocol. Tabulated-function usage was checked against the
installed OpenMM: `Continuous1DFunction` takes no size arguments while 2D/3D take sizes
first, `periodic` is fixed at construction and correctly omitted from
`setFunctionParameters`, and the C-order `ravel` of a reversed-shape grid puts the first
CV fastest, which is what OpenMM expects.

Three deviations are deliberate and kept: storing compressed kernels rather than a bias
grid (the point of OPES); splitting `addKernel` from `updateContext`, which lets kernels
be deposited without a `Context` and is what makes the KDE testable in isolation (its
docstring must say that `addKernel` alone leaves the context stale); and drawing the
walker id from a fresh generator instead of the global NumPy RNG, which is better than
what `Metadynamics` does — modernized to `np.random.default_rng().integers`.

The following are corrected:

- `getattr(mm, f"Continuous{d}DFunction")` is replaced by an explicit branch on
  dimensionality, as in `Metadynamics`. The dynamic lookup is opaque and cannot be
  resolved by `ty`, which CI runs.
- `varianceFrequency=0` passes validation — the check is `if self.varianceFrequency and
  ...`, and `0` is falsy — then raises `ZeroDivisionError` while computing `tau`.
  Confirmed. Validation must reject it explicitly.
- `saveFrequency=0` with a `biasDir` constructs successfully but silently disables
  multi-walker setup (`_cases` stays `("total",)` and `_id` is never assigned), then
  raises `ZeroDivisionError` inside `step`. Confirmed. Validation must reject it.
- `biasFactor <= 1.0` raises `"OPES barrier must be greater than 1 kT"` even when the
  user passed `biasFactor` explicitly and never set a barrier. Confirmed. The message
  must name whichever argument the user actually supplied.
- The public API gains type hints, and docstring parameter lines gain types
  (`system: System`), matching `Metadynamics`' documented style. `Metadynamics` predates
  annotations, but this package's CI type-checks with `ty`.

A check that was suspected and found to be a non-issue: `Metadynamics` validates
`saveFrequency < frequency` where the source only checks divisibility, but
`saveFrequency % frequency != 0` already rejects every such case except `0`, which the
new explicit check above covers.

### 7.7 New in v1: warm-up estimation of `sigma^(0)`

The one feature added rather than ported. Opt-in via `warmupSteps`, default `None`,
which reproduces current behavior exactly and so leaves the §8.3 parity test untouched.

**What it does.** For the first `warmupSteps` steps the simulation runs with **no kernels
deposited**. The bias table keeps its initial constant value, which is flat and therefore
exerts no force, so the trajectory is genuinely unbiased. CV statistics are collected
every `varianceFrequency` steps using the existing estimator, unchanged. At
`warmupSteps` the measured variance is frozen as `sigma^(0)**2` and no longer updated;
deposition begins from that point.

**Why it is worth adding to an otherwise faithful port.** The original paper prescribes
exactly this procedure and leaves it to the user: *"The initial bandwidth is simply
chosen to be equal to the smaller standard deviation of the CVs in the minima, which can
be measured in a short unbiased run."* The warm-up automates that run instead of
requiring a separate simulation and a hand-copied number. It also earns its place three
other ways:

1. It removes the interpretive question of §7.5 entirely. Nothing is deposited during
   warm-up, so the trajectory is unbiased and the measured width *is* `sigma^(0)` — there
   is no sampled-versus-unbiased ambiguity left to resolve.
2. It makes the §7.4.4 NaN impossible by construction, since adequate statistics before
   the first kernel become a structural guarantee rather than a constraint on parameters.
3. It cleans up Silverman's rule. Today the bandwidth is
   `sigma^(0)(t) * silverman(N_eff(t))`, two independently drifting factors, while the
   rule assumes `sigma^(0)` is a fixed property of the distribution and that all shrinkage
   comes from `N_eff`. Freezing `sigma^(0)` makes the shrinkage clean and monotonic.

**Arithmetic, shared with §7.5.** `self._variance` holds the variance of the *sampled*
distribution by convention. The warm-up measures the *unbiased* variance. So on freezing,
the stored value is `biasFactor * measuredVariance` — precisely the transformation §7.5
applies to `biasWidth`. Both paths therefore reduce to "store `gamma` times an unbiased
variance", and share one code path: replace the accumulators with a fresh
`RunningAverage` seeded with that value and clear the adaptive flag. The reweighted KDE
then builds kernels at `sigma^(0)` and the plain KDE at `sqrt(gamma) * sigma^(0)`, each
correct for what it estimates.

**Restart.** Warm-up completion is keyed off an explicit `warmupComplete` flag persisted
in the walker's own saved state, not re-derived from the step counter. Deriving it would
risk rescaling the already-frozen variance by `gamma` a second time on reload.

**Multi-walker.** Each walker warms up independently and freezes its own `sigma^(0)`.
Sharing one value would require a synchronization barrier — every walker blocked until
the slowest finishes warm-up — which is a great deal of coordination machinery for a
quantity that varies little between walkers sampling the same basins. The per-walker
behavior is documented.

Note the interaction this creates: during warm-up a walker has no kernels, so it writes
and its peers read an **empty** KDE on every sync. That makes the §7.4.3 empty-KDE
restore bug a routine occurrence on this path rather than a rare one, so that fix is
load-bearing here rather than defensive.

**Validation.** `warmupSteps` requires `varianceFrequency` (there is otherwise no
sampling schedule), must be a positive multiple of it, and must yield at least two
samples; the docstring recommends substantially more. The §7.4.4 positive-variance guard
still applies as a backstop.

### 7.8 Post-release code review (fixed in 0.1.1)

A code review run against the merged `src/openmm_opes` after 0.1.0 shipped found seven
further defects. Each was independently reproduced before being accepted, and each fix
carries a regression test that was confirmed to fail against the unfixed code — the
discipline that was missing the first time, and the direct reason finding 1 escaped.

1. **NaN bias still reached the forces** (HIGH). `addKernel` skipped the deposition, but
   `_onInterval` then called `updateContext` regardless. Reproduced: with
   `varianceFrequency == frequency` the particle is at `Vec3(nan, nan, nan)` by step
   300. A freshly constructed sampler also returned NaN from `getBias()`. Fixed on both
   fronts: `getBias` now returns the well-defined empty-estimate limit (the flat
   `-barrier` floor the table is initialized to), and `addKernel` reports whether it
   deposited so `_onInterval` only refreshes the context when the estimate changed.
2. **`setState` silently reverted the KDE options** (HIGH). `_kdeFromState` built
   `OnlineKDE(cvSpace)` with no kwargs, and the options were never retained. Reproduced:
   a `kernelShape="compact", compressionThreshold=0.0` sampler restored as
   `gaussian`/`1.0`, an immediate 8.19 kJ/mol bias discrepancy. Fixed by retaining the
   options and funnelling every rebuild through one `_newKDE` factory.
3. **`setState` never restored the `"self"` accumulators** that `getState` writes
   (HIGH). Since `_syncWithDisk` rebuilds `total` from `self` plus peers, the restored
   history was discarded at the next sync, and — worse — the walker republished an
   *empty* state to `biasDir`, so its peers lost that history too. Verified by reading
   the republished `.npz` back: `kde kernels = 0`.
4. **`__iadd__` corrupted `_logSumWSq`** (MEDIUM). Re-deriving the sum of squared
   weights from merged kernels overstates it badly, because a compressed kernel carries
   the combined weight of everything it absorbed. Reproduced: merging two 2000-sample
   KDEs gave `neff = 92.9` instead of 4000, making every subsequently deposited kernel
   2.12x too wide. **This one is inherited, not introduced**: the source's compressed
   branch does the same, while its uncompressed branch (dropped here) combines the
   moments correctly. Fixing it is therefore a deliberate divergence from the source,
   affecting multi-walker merging only; §8.3's fixture exercises single-KDE deposition
   and is unaffected, which the passing parity test confirms.
5. **`frequency` and `statsWindowSize` lacked positivity checks** (LOW), the same class
   already guarded for `varianceFrequency`/`saveFrequency`. Both constructed fine and
   then raised `ZeroDivisionError` mid-run.
6. **`getFreeEnergy`'s docstring stated the wrong grid spacing** (LOW): it claimed
   `(maxValue-minValue)/gridWidth` where the axis is built by
   `np.linspace(minValue, maxValue, gridWidth)`, i.e. `/(gridWidth-1)`. Inherited from
   `openmm.app.Metadynamics`' own docstring, but wrong either way.
7. **A restarted walker could not reclaim its file slot** (LOW). `OPES` always built
   `BiasSharer(biasDir)` with a fresh random id and no way to pass one, so the previous
   run's file lingered and was read back forever as a phantom extra peer — compounding
   with finding 3. Fixed by plumbing `walkerId` through the `OPES` constructor and
   having `BiasSharer` resume past its own highest existing index.

### 7.9 Style

The camelCase API is kept throughout (`addKernel`, `getFreeEnergy`, `varianceFrequency`,
…) to match OpenMM's own conventions, since users of `openmm.app` and of the
metadynamics class already work in that idiom. Ruff's naming lint is not enabled (the
selected rule set contains no `N` rules), so this needs no per-file suppression.

## 8. Testing

### 8.1 Unit tests, no OpenMM required

The bulk of the suite. These import only `kde.py` and `io.py`, so they run in any
environment.

- `test_cvspace.py` — displacement and endpoint round-trips in periodic and
  non-periodic spaces, minimum-image correctness across the periodic seam, grid
  distances, `closestNode` at and beyond the domain edges, and `foldedGrid` reflection
  when `bounded=True`.
- `test_kernel.py` — log height normalization for both shapes (numerically integrate a
  single kernel on a fine grid and confirm it integrates to its weight), compact-shape
  support truncation at 3 bandwidths, `findNearest` with and without ignored indices,
  and `merge` conserving total weight and sample count while reproducing the weighted
  mean and second moment of the two parents.
- `test_kde.py` — a KDE built from deposits at known positions reproduces, on the grid,
  a directly computed mixture of the same kernels; `getLogPDF` normalizes; compression
  reduces kernel count when deposits repeat at one spot but leaves well-separated
  deposits alone; `compressionThreshold=0` disables merging; `getLogMeanDensity` matches
  a direct computation; with merging disabled, `__iadd__` of two KDEs equals one KDE fed
  both deposit streams; save/load round-trips preserve `getLogPDF` exactly, including
  for an empty KDE; and — per §12.4 — across a long run with aggressive merging the
  per-removal cancellation margin `min(x - y)` stays bounded away from zero, which is
  the invariant that makes the incremental `_logPK` bookkeeping safe.
- `test_io.py` — `BiasSharer` writes atomically, ignores its own files, picks up a
  peer's newer index, tolerates a peer file vanishing mid-scan, and does not mutate the
  walker's own state when building the total (the §7.4.1 regression).

### 8.2 Integration tests, OpenMM required

A handful of genuinely-running simulations on tiny systems, kept small enough for CI.
They are collected in `test_opes.py` and skipped with `pytest.importorskip("openmm")`
so the suite still passes without OpenMM.

- Constructor validation: every error in §5 raises with the expected message.
- A single particle on the 1-D double well `U(x) = 25*(1 - x**2)**2` kJ/mol, biased
  along `x`, with `barrier=30`, `frequency=200`, `varianceFrequency=20`, a seeded
  `LangevinMiddleIntegrator` and the Reference platform: the recovered
  `getFreeEnergy()` profile matches the analytic potential after subtracting its
  minimum, and the simulation crosses the barrier repeatedly. This configuration has
  been run during the design review and converges as follows (2.4M steps, ~13 s):

  | steps | RMSE vs analytic | estimated barrier | kernels |
  |---|---|---|---|
  | 400k | 5.97 | 31.64 | 71 |
  | 1.2M | 0.78 | 24.43 | 84 |
  | 2.4M | 0.43 | 24.39 | 89 |

  against an analytic barrier of 25.00 kJ/mol and `kT = 2.49` kJ/mol. The CI test runs
  the shorter 1.2M-step version and asserts RMSE below 1.5 kJ/mol. Tolerances are fixed
  once from a seeded run and then left alone — they are not to be re-tuned whenever a
  test fails, since that would hide the very regressions the test exists to catch.
- The same system run in explore mode reproduces the qualitative tradeoff the
  OPES-Explore paper exists to demonstrate: more barrier crossings (79 vs 67 over 2.4M
  steps) but slower FES convergence (RMSE 2.34 vs 0.43 kJ/mol). The test asserts the
  ordering of both quantities, not their absolute values, since it is the tradeoff and
  not the specific numbers that the method guarantees.
- A 2-D harmonic system: `getFreeEnergy()` recovers the known quadratic profile in the
  sampled region.
- Explore mode runs and produces a finite bias and a free energy of the right shape.
- Multi-walker: two `OPES` instances sharing one `biasDir`, stepped alternately, end up
  with each walker's `getNumKernels()` reflecting both walkers' deposits.
- `bounded=True` and `kernelShape="compact"` each run end to end.

### 8.3 Parity against the source implementation

The correctness guarantee that matters most for a scientific port: the library must
reproduce the existing implementation, not merely be self-consistent.

A one-off script (kept out of the package, under `tests/data/`) drives the *original*
`online_kde.py` with a fixed pseudo-random sequence of deposits — positions, log
weights, and variances — in 1-D and 2-D, periodic and non-periodic, and saves the
resulting `getLogPDF()` grid, `getLogMeanDensity()`, kernel count, and kernel centers
to `tests/data/reference_kde.npz`. That file is committed.

`test_parity.py` replays the identical deposit sequence through `openmm_opes.kde` and
asserts agreement with the stored reference at `rel=1e-5` (`pytest.approx`). A tighter
`rel=1e-12` was tried first and passed on the machine that generated the fixture, but
failed in CI on Python 3.13 (while 3.11 passed) with differences up to `rel=3e-7` —
confirmed to be cross-build floating-point drift in `exp`/`log` between separately
compiled conda-forge numpy builds, not a code difference, since the same commit's
`src/openmm_opes` passed against the identical fixture on 3.11. `rel=1e-5` sits two
orders of magnitude above that observed worst case, while a genuine formula error would
show up as an O(1) or many-percent difference, not a few ULPs compounded across ~400
chained deposits. This pins the port against silent numerical drift without vendoring
the old code into
the repository.

## 9. Continuous integration

`.github/workflows/ci.yml`, three jobs, on push to `main` and on pull requests.

- **lint** — Python 3.11 with a plain `pip install ruff ty numpy scipy`, then
  `ruff check src tests`, `ruff format --check src tests`, `ty check src tests`. It
  deliberately does *not* install the package itself, because `openmm` is a hard
  dependency whose canonical channel is conda-forge; ruff needs nothing installed, and
  `ty` needs only numpy and scipy resolvable, with `openmm` covered by the
  `allowed-unresolved-imports` entry in §3. This keeps the lint job fast and free of
  any conda setup.
- **test** — `mamba-org/setup-micromamba` with a conda-forge environment installing
  `openmm` plus the dev extras, matrixed over Python 3.11 and 3.13, then `pytest -v`.
  This job runs the whole suite including the integration tests.
- **docs** — on pushes to `main` only: installs the `docs` extra and runs
  `mkdocs gh-deploy --force` to publish to GitHub Pages.

The conda-based test job is the one deliberate divergence from `sieve`'s all-pip CI, and
it exists because OpenMM's canonical distribution channel is conda-forge.

## 10. Documentation

mkdocs-material with mkdocstrings generating the API reference from the NumPy-style
docstrings already present in the source (carried over and extended for the new
options).

- `index.md` — what OPES is, what the package provides, install instructions leading
  with conda/mamba.
- `quickstart.md` — a complete runnable example: a particle on a 2-D Müller-Brown
  potential, defining `BiasVariable`s, constructing `OPES`, stepping, and plotting the
  recovered free-energy surface. Derived from the existing `müller-brown/simulate.py`.
- `theory.md` — the method as implemented, tying the code to the two papers: the
  reweighted probability estimate, the bias expression
  `prefactor * logaddexp(logPDF - logMeanDensity, logEpsilon)`, the explore-mode
  variant, kernel compression, and the Silverman bandwidth rule.
- `api.md` — autodoc for `OPES`, `OnlineKDE`, `CVSpace`, `Kernel`.

Published to GitHub Pages by a `docs` job on pushes to `main`.

Because the spec directory lives under `docs/`, `mkdocs.yml` sets
`exclude_docs: superpowers/` so design documents do not end up in the published site.

## 11. Deliverables

1. Repository renamed to `openmm-opes` on GitHub and locally, remote updated. Done.
2. `pyproject.toml`, `src/openmm_opes/{__init__,opes,kde,io}.py`.
3. Test suite per §8, including the committed parity reference.
4. CI workflow per §9.
5. mkdocs site per §10 and a rewritten `README.md`.
6. All of §7.4's bug fixes, each with a regression test.
7. The warm-up mode of §7.7, with tests covering the frozen-`sigma^(0)` value, restart
   across the `warmupComplete` flag, and the empty-KDE sync it makes routine.

Version 0.1.0. Publishing to PyPI or conda-forge is not part of this work.

## 12. Appendix: verification of the method against the papers

Before freezing the port, the source implementation was checked equation by equation
against Invernizzi and Parrinello, and the checks were run rather than reasoned about.
Everything below was reproduced numerically in the local `openmm` environment. The
conclusion is that **the method implementation is correct**; the defects in §7.4 and
§7.5 are all in the machinery around it, not in the OPES equations.

### 12.1 Exact agreement with the published equations

| Quantity | Reference | Result |
|---|---|---|
| Bias, standard OPES | Explore paper Eq. 5 / original Eq. 9: `V = (1-1/gamma)/beta * log(P_n/Z_n + eps)` | matches to 3.6e-15 |
| Bias, OPES-explore | Explore paper Eq. 9: `V = (gamma-1)/beta * log(p^WT_n/Z_n + eps)` | matches to 5.3e-15 |
| Regularization `eps` | `eps = exp(-beta*dE/(1-1/gamma))` | matches to machine precision, both modes |
| Probability estimate | Eq. `iter_prob`, weights `w_k = exp(beta*V_(k-1)(s_k))` | bias energy is read from the OPES force group *before* the new kernel is deposited, so the weight is genuinely `V_(k-1)` |
| Explore estimate | Eq. `iter_prob-explore`, `p^WT_n = (1/n) sum_k G` | the plain KDE deposits with `logWeight = 0`, so weights are uniform and `logSumW = log n` |
| Normalization `Z_n` | Eq. `iter_zed`, integral approximated by a sum over compressed kernels | `getLogMeanDensity()` agrees with a direct mean of `P_n` over the kernel centers to 1e-8 |
| Bandwidth | Eq. `bandwidth`, Silverman with `N_eff = (sum w)^2 / sum w^2` | `exp(2*logSumW - logSumWSq)` agrees with the Kish definition to machine precision |
| Kernel normalization | `h = prod_i (sigma_i*sqrt(2*pi))^-1` | a single kernel integrates to its weight to 1e-6 |
| Compact kernel constant | `559872/35` | equals `integral of (9-x^2)^4 over [-3,3]` exactly |
| Grid evaluation | — | `getLogPDF()` on the grid equals a direct mixture sum to 4.4e-16 |

The correct mode-dependent choice of estimator is worth stating explicitly, because it
is the one place the two papers are easy to conflate. `getBias` reads the **plain**
(unweighted) KDE in explore mode and the **reweighted** KDE otherwise, which is exactly
the distinction in the explore paper's Sec. 3. `getFreeEnergy` always reads the
reweighted KDE, i.e. the importance-sampling estimator of Eq. `reweighting`; in explore
mode the paper also permits `F = -gamma/beta * log p^WT_n`, and notes the two agree only
at convergence. Always using the reweighted route is the better-converging choice.

### 12.2 The `variance / biasFactor` factor is correct, and why

The reweighted KDE receives `variance / biasFactor` rather than `variance`. This is not
an arbitrary fudge: the running variance is measured from the biased trajectory, whose
distribution is `P^(1/gamma)` and therefore wider than `P` by `sqrt(gamma)`. Dividing by
`gamma` converts a sampled variance into an unbiased one, which is what the reweighted
KDE needs. The plain KDE correctly keeps the full sampled variance, since it estimates
the sampled distribution. The factor is right on the adaptive path; §7.5 covers the one
path where the premise behind it does not hold.

### 12.3 End-to-end convergence

See the table in §8.2. Standard OPES recovers the analytic double-well barrier to
0.6 kJ/mol (0.24 kT) with an RMSE of 0.43 kJ/mol, converging monotonically. Explore mode
reproduces the published tradeoff — more barrier crossings, slower convergence.

### 12.4 `_logPK` drift: no safeguard, but pin the invariant that makes it safe

`_logPK` (the density at each kernel center) is maintained incrementally, and removed
kernels are subtracted in log space by `_logsubexp`, which computes
`log(exp(x) - exp(y))` and loses precision as `x - y` approaches zero. Here `x` is the
total log density at a center and `y` is the contribution of the kernel being removed,
so the quantity that decides whether a safeguard is needed is `min(x - y)` across every
removal that ever happens.

**Structurally, that margin cannot vanish in v1.** `_removeKernels` has exactly two call
sites. In `_pushKernel` it is always reached *after* the merged replacement kernel's
contribution has been added to `_logPK`; that replacement carries the combined weight of
everything being removed and sits among them, so it dominates the local density and
bounds the margin away from zero. The other call site is `getSmoothedCopy`, which
removes kernels with no compensating addition — and which is out of scope for v1 (§2).

Measured, instrumenting every removal across four regimes (~120,000 removals):

| regime | deposits | removals | `min(x-y)` | max `logPK` drift |
|---|---|---|---|---|
| 1-D, mild weights | 40,000 | 39,815 | 0.707 nats | 8.9e-15 |
| 1-D, wide weights (`exp ±12`) | 40,000 | 39,950 | 0.111 nats | 1.4e-14 |
| 1-D, wide weights, tight cluster | 40,000 | 39,992 | 0.410 nats | 9.9e-14 |
| 2-D, wide weights | 20,000 | 19,543 | 0.0155 nats | 6.4e-14 |

The worst margin observed was 0.0155 nats — the removed kernel never accounted for more
than about 98.5% of the density at any center, costing at most ~2 decimal digits in that
one subtraction. Error in `log Z_n` was at most 7.1e-15. By contrast, removing an
isolated kernel with no compensating addition gives a margin of exactly 0.000, i.e. full
catastrophic cancellation, confirming that the margin is indeed the operative quantity
and that the ordering in `_pushKernel` is what supplies it.

**Recommendation: do not add a periodic-recomputation safeguard.** Cost is not the
objection — an `O(n^2)` recompute with `n` in the hundreds is cheap beside the per-deposit
grid evaluation. The objection is that such a safeguard would *mask* rather than prevent:
if the invariant were ever violated, a periodic patch-up would hide a real correctness
bug and would still leave garbage between recomputations, while adding a tunable with no
principled value.

Instead, pin the invariant. `tests/test_kde.py` asserts that across a long run with
aggressive merging the per-removal margin stays bounded away from zero, and
`_removeKernels` carries a comment stating the ordering requirement. This fails loudly if
someone later reintroduces a removal path without a compensating addition — which is
exactly what re-adding `getSmoothedCopy` would do, so that deferred feature must revisit
this section before it lands.
