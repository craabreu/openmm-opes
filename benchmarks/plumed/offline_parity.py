"""
Deposition parity between openmm-opes and PLUMED's OPES, without dynamics.

Both implementations receive the same CV time series. PLUMED reads it with
READ in ``plumed driver --noatoms``. openmm-opes gets it through
:class:`TrajectoryReplay`, a stand-in for an OpenMM Simulation whose ``step``
moves a dummy particle, whose x (and y) coordinate is the CV, to the next
sample. Since OPES depositions depend only on the visited CV values and on the
bias itself, any difference in the resulting bias comes from the algorithms,
not from dynamics.

Usage:
    python offline_parity.py PLUMED_EXECUTABLE WORKDIR [--case NAME ...]
        [--plumed-merge]

With --plumed-merge, openmm-opes merges kernels with PLUMED's rule (see
plumed_merge.py) instead of its own, which isolates the merge rule as the
remaining difference between the two codes.

For each case and each bias acting in it (OPES and OPES-explore in the
combined cases), prints the kernel counts, the range of each final bias on a
grid over the sampled region, and the RMS and maximum difference between the
two codes' biases on that grid after shifting each to zero at its maximum. It
then prints the mean and RMS difference between the total biases felt along
the second half of the trajectory.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openmm
from common import KB
from openmm import app, unit

from openmm_opes import OPES, CombinedOPES


@dataclass
class Case:
    """One comparison: dimensionality, OPES variant and sampler settings.

    ``bandwidth`` fixes the initial kernel bandwidth in both codes; when None,
    both estimate it adaptively. With an ``exploreBarrier``, OPES and
    OPES-explore act together (OPES_COM): two actions in PLUMED and a
    CombinedOPES in openmm-opes, and ``explore`` is ignored.
    """

    dims: int
    explore: bool
    barrier: float = 30.0
    exploreBarrier: float | None = None
    pace: int = 10
    samples: int = 30_000
    temperature: float = 300.0
    seed: int = 7
    grid: tuple = ((-2.0, 2.0, 801), (-2.0, 2.0, 201))
    bandwidth: float | None = None


CASES = {
    "1d-opes": Case(dims=1, explore=False),
    "1d-explore": Case(dims=1, explore=True),
    "2d-opes": Case(dims=2, explore=False),
    "2d-explore": Case(dims=2, explore=True),
    "1d-opes-fixed": Case(dims=1, explore=False, bandwidth=0.05),
    "1d-explore-fixed": Case(dims=1, explore=True, bandwidth=0.05),
    "1d-combined": Case(dims=1, explore=False, exploreBarrier=15.0),
    "2d-combined": Case(dims=2, explore=False, exploreBarrier=15.0),
    "1d-combined-fixed": Case(
        dims=1, explore=False, exploreBarrier=15.0, bandwidth=0.05
    ),
}


def plumed_actions(case: Case) -> list[tuple[str, str, float]]:
    """Label, PLUMED action and barrier of each bias acting in a case."""
    if case.exploreBarrier is not None:
        return [
            ("o", "OPES_METAD", case.barrier),
            ("e", "OPES_METAD_EXPLORE", case.exploreBarrier),
        ]
    action = "OPES_METAD_EXPLORE" if case.explore else "OPES_METAD"
    return [("o", action, case.barrier)]


def trajectory(case: Case) -> np.ndarray:
    """
    A CV time series that crosses between wells: overdamped Langevin dynamics
    on U = 25 (1 - x^2)^2 (+ 10 y^2) kJ/mol at 900 K, so that both wells are
    visited without any bias.
    """
    rng = np.random.default_rng(case.seed)
    kT = KB * 900.0
    dt = 2e-4  # in units where the mobility is 1
    x = np.zeros(case.dims)
    x[0] = -1.0
    out = np.empty((case.samples, case.dims))
    for i in range(case.samples):
        for _ in range(5):
            force = np.empty(case.dims)
            force[0] = 100 * x[0] * (1 - x[0] ** 2)
            if case.dims == 2:
                force[1] = -20 * x[1]
            x = x + dt * force + np.sqrt(2 * kT * dt) * rng.standard_normal(case.dims)
        out[i] = x
    return out


def evaluation_grid(case: Case, traj: np.ndarray) -> np.ndarray:
    """Points inside the sampled region, on which both biases are compared."""
    if case.dims == 1:
        return np.linspace(traj[:, 0].min(), traj[:, 0].max(), 200)[:, None]
    xs = np.linspace(traj[:, 0].min(), traj[:, 0].max(), 40)
    ys = np.linspace(np.percentile(traj[:, 1], 1), np.percentile(traj[:, 1], 99), 20)
    return np.array([(x, y) for x in xs for y in ys])


def plumed_bias(plumed, workdir: Path, case: Case, traj, grid):
    """Run PLUMED on the trajectory, then evaluate its final bias on the grid.

    Returns the bias felt at each sample, the final bias on the grid, and the
    final number of compressed kernels.
    """
    names = ["x", "y"][: case.dims]
    actions = plumed_actions(case)
    sigma = "" if case.bandwidth is None else f" SIGMA={case.bandwidth}"

    def bias_actions(pace, state):
        return "".join(
            f"{label}: {action} ARG={','.join(names)} TEMP={case.temperature}"
            f" BARRIER={barrier}{sigma} PACE={pace} {state.format(label=label)}\n"
            for label, action, barrier in actions
        )

    labels = [label for label, _, _ in actions]
    total = "b: COMBINE ARG=" + ",".join(f"{la}.bias" for la in labels)
    total += " PERIODIC=NO\n"

    def write_series(path, values):
        with open(path, "w") as file:
            file.write("#! FIELDS time " + " ".join(names) + "\n")
            for i, row in enumerate(values):
                file.write(f"{i} " + " ".join(f"{v:.10f}" for v in row) + "\n")

    def read_actions(filename):
        return "".join(
            f"{name}: READ FILE={filename} VALUES={name} IGNORE_TIME IGNORE_FORCES\n"
            for name in names
        )

    run = [plumed, "driver", "--noatoms", "--plumed"]
    write_series(workdir / "traj.dat", traj)
    # PLUMED writes the state file when the step is a multiple of the stride,
    # and the driver's last step is samples - 1.
    (workdir / "deposit.dat").write_text(
        read_actions("traj.dat")
        + bias_actions(
            case.pace,
            f"STATE_WFILE=state_{{label}} STATE_WSTRIDE={case.samples - 1}",
        )
        + total
        + "PRINT ARG=b,"
        + ",".join(f"{la}.nker" for la in labels)
        + " FILE=colvar FMT=%.10g\n"
    )
    subprocess.run([*run, "deposit.dat"], cwd=workdir, check=True, capture_output=True)
    deposit = np.loadtxt(workdir / "colvar", comments="#")

    # Evaluate the final bias by restarting from the state with a deposition
    # pace longer than the grid, so that nothing is added.
    write_series(workdir / "grid.dat", grid)
    (workdir / "evaluate.dat").write_text(
        "RESTART\n"
        + read_actions("grid.dat")
        + bias_actions(10 * len(grid), "STATE_RFILE=state_{label}")
        + "PRINT ARG="
        + ",".join(f"{la}.bias" for la in labels)
        + " FILE=gridbias FMT=%.10g\n"
    )
    subprocess.run([*run, "evaluate.dat"], cwd=workdir, check=True, capture_output=True)
    components = np.loadtxt(workdir / "gridbias", comments="#", ndmin=2)[:, 1:]
    kernels = tuple(int(n) for n in deposit[-1, 2:])
    return deposit[:, 1], components, kernels


class TrajectoryReplay:
    """A stand-in for an OpenMM Simulation that replays a CV trajectory.

    It has the ``context``, ``currentStep`` and ``step`` that
    :meth:`OPES.stepTogether` uses. Each step moves the dummy particle to the
    next sample and records the bias felt there, before any deposition at that
    step, which is the order in which PLUMED's driver reports the bias.
    """

    def __init__(self, system, samples, samplers):
        self.samples = samples
        self.samplers = samplers
        self.context = openmm.Context(
            system,
            openmm.VerletIntegrator(0.001),
            openmm.Platform.getPlatformByName("Reference"),
        )
        self.currentStep = 0
        self.biases = np.empty(len(samples))
        self._move(0)

    def _move(self, index):
        xyz = np.zeros(3)
        xyz[: self.samples.shape[1]] = self.samples[index]
        self.context.setPositions([openmm.Vec3(*xyz)])
        self.biases[index] = self.bias()

    def bias(self):
        """The total bias at the particle's current position, in kJ/mol."""
        return sum(
            sampler.getBiasEnergy(self.context).value_in_unit(unit.kilojoules_per_mole)
            for sampler in self.samplers
        )

    def step(self, steps):
        for _ in range(steps):
            self.currentStep += 1
            self._move(self.currentStep)


def openmm_opes_bias(case: Case, traj, grid):
    """Replay the trajectory through openmm-opes.

    Returns the bias felt at each sample, the final bias of each component on
    the grid, and the final numbers of compressed kernels.
    """
    system = openmm.System()
    system.addParticle(1.0)
    variables = []
    for k, (low, high, size) in enumerate(case.grid[: case.dims]):
        force = openmm.CustomExternalForce("xyz"[k])
        force.addParticle(0, [])
        width = case.bandwidth or (high - low) / 100
        variables.append(app.BiasVariable(force, low, high, width, False, size))
    # PLUMED updates the adaptive variance at every step
    varianceFrequency = None if case.bandwidth else 1
    temperature = case.temperature * unit.kelvin
    barrier = case.barrier * unit.kilojoules_per_mole
    if case.exploreBarrier is None:
        sampler = OPES(
            system,
            variables,
            temperature,
            barrier,
            case.pace,
            varianceFrequency,
            exploreMode=case.explore,
        )
    else:
        sampler = CombinedOPES(
            system,
            variables,
            temperature,
            barrier,
            case.exploreBarrier * unit.kilojoules_per_mole,
            case.pace,
            varianceFrequency,
        )
    replay = TrajectoryReplay(system, traj, [sampler])
    OPES.stepTogether(replay, [sampler], len(traj) - 1)
    on_grid = []
    for point in grid:
        xyz = np.zeros(3)
        xyz[: case.dims] = point
        replay.context.setPositions([openmm.Vec3(*xyz)])
        if isinstance(sampler, CombinedOPES):
            energies = sampler.getComponentBiasEnergies(replay.context)
            on_grid.append(
                [e.value_in_unit(unit.kilojoules_per_mole) for e in energies]
            )
        else:
            on_grid.append([replay.bias()])
    kernels = sampler.getNumKernels()
    if isinstance(kernels, int):
        kernels = (kernels,)
    return replay.biases, np.array(on_grid), kernels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plumed")
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--case", action="append", choices=sorted(CASES))
    parser.add_argument("--plumed-merge", action="store_true")
    args = parser.parse_args()
    if args.plumed_merge:
        from plumed_merge import use_plumed_merge

        use_plumed_merge()
    for name in args.case or list(CASES):
        case = CASES[name]
        workdir = args.workdir / name
        if workdir.exists():
            shutil.rmtree(workdir)  # PLUMED appends to old output on RESTART
        workdir.mkdir(parents=True)
        traj = trajectory(case)
        grid = evaluation_grid(case, traj)
        p_along, p_grid, p_kernels = plumed_bias(args.plumed, workdir, case, traj, grid)
        o_along, o_grid, o_kernels = openmm_opes_bias(case, traj, grid)
        np.savez(
            workdir / "biases.npz",
            traj=traj,
            grid=grid,
            plumed_along=p_along,
            openmm_opes_along=o_along,
            plumed_grid=p_grid,
            openmm_opes_grid=o_grid,
        )
        late = slice(len(traj) // 2, None)
        along = o_along[late] - p_along[late]
        for k, (_, action, _) in enumerate(plumed_actions(case)):
            p, o = p_grid[:, k], o_grid[:, k]
            shape = (o - o.max()) - (p - p.max())
            print(
                f"{name:17s} {action:18s} kernels {p_kernels[k]:4d} (PLUMED)"
                f" {o_kernels[k]:4d} (openmm-opes)"
                f" | bias range {np.ptp(p):6.2f} / {np.ptp(o):6.2f}"
                f" | grid diff rms {np.sqrt(np.mean(shape**2)):5.2f}"
                f" max {np.abs(shape).max():5.2f} kJ/mol",
                flush=True,
            )
        print(
            f"{name:17s} {'total bias felt':18s} along diff mean {along.mean():+6.2f}"
            f" rms {np.sqrt(np.mean((along - along.mean()) ** 2)):5.2f} kJ/mol",
            flush=True,
        )


if __name__ == "__main__":
    main()
