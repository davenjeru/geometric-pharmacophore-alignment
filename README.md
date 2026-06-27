# Geometric Pharmacophore Alignment

This is a cross-docking tool. It places a small molecule into a protein pocket
that is described only by pharmacophore interaction sites and exclusion spheres,
with no explicit protein structure to work from.

For each target it generates 3D conformers from the SMILES, aligns them so that
the ligand's chemical features land on the matching interaction sites, throws
away any pose that bumps into an exclusion sphere, and keeps the single
best-scoring pose. All five best poses are written to one SDF file.

## Setup

The project targets Python 3.13 and uses RDKit for the cheminformatics work.

```bash
uv sync
```

If you are not using uv, install the dependencies listed in `pyproject.toml`
(`rdkit`, `ijson`, plus the notebook extras) into a Python 3.13 environment.

## Running it

```bash
uv run main.py
```

By default it reads `data/targets.json` (it looks for `/root/data/targets.json`
first, then falls back to the local copy) and writes `results/docked_poses.sdf`.
You can point it somewhere else:

```bash
uv run main.py --input data/targets.json --output results/docked_poses.sdf
```

Run `uv run main.py --help` to see every option with its default.

## How it works

The pipeline is five steps, one per stage of the problem:

1. Generate a conformer ensemble. The SMILES is embedded with ETKDG and each
   conformer is relaxed with the MMFF force field, so we sample realistic 3D
   shapes of the molecule.
2. Match features. Every ligand atom is tagged with the pharmacophore families
   it belongs to (Donor, Acceptor, Hydrophobe, Aromatic). Only atoms whose
   family matches a site are allowed to satisfy that site.
3. Align. For each conformer we guess which atom goes to which site, then solve
   for the rigid rotation and translation that best lines those atoms up with
   the sites. The math is a weighted Kabsch superposition, which RDKit provides
   directly, so we are not hand-rolling the linear algebra.
4. Score and reject. A pose is scored with the task's Gaussian function. Any
   pose where a heavy atom sits inside an exclusion sphere is rejected.
5. Select. We keep the highest-scoring pose, always preferring a clash-free pose
   over a clashing one regardless of score.

The hard part is step 3, because we do not know up front which atom belongs to
which site. We handle that with random restarts: try many random starting
correspondences per conformer, refine each one by snapping every site to its
nearest matching atom and refitting, and keep the best result. This turned out
to work as well as running the refinement to convergence, and it is simpler to
reason about.

## Project layout

```
main.py            Command line entry point, argument parsing, summary table
docking/
  config.py        Constants and default paths in one place
  models.py        Dataclasses for targets, sites, exclusion volumes, poses
  io.py            Streaming JSON reader and SDF writer
  features.py      Feature detection and the per-atom family map
  alignment.py     Point-set superposition and the correspondence search
  scoring.py       Gaussian score and the clash check
  pipeline.py      Ties it all together, one target at a time
experiment/        Exploratory notebook and an earlier loader, kept for reference
```

The notebook in `experiment/` is where the approach was worked out before it was
turned into the package. It is not part of the deliverable.

## Options

Most flags trade pose quality against runtime. A few are modeling choices that
are off by default so the results match a fixed baseline.

| Flag | Default | What it does |
| --- | --- | --- |
| `--conformers` | 200 | Conformers generated per ligand. More shapes, better coverage, slower. |
| `--restarts` | 20 | Random correspondence guesses tried per conformer. |
| `--refine-steps` | 1 | Reassign-and-refit cycles after the initial fit. 0 trusts the guess, higher runs full ICP. |
| `--seed` | 42 | Random seed, so a run can be repeated. |
| `--lump-hydrophobes` | off | Count RDKit's LumpedHydrophobe atoms as Hydrophobe too. |
| `--clash-include-hydrogens` | off | Include hydrogens in the clash check, not just heavy atoms. |

The scoring constants (the Gaussian width of 1.25 and the 0.1 clash tolerance)
are fixed by the task and live in `config.py` rather than as flags.

## Results

A default run produces clash-free poses for all five targets. Scores as a
percentage of the maximum possible:

| Target | Score | Max | Percent |
| --- | --- | --- | --- |
| target_1 | 4.80 | 5.40 | 88.9% |
| target_2 | 3.42 | 7.10 | 48.2% |
| target_3 | 4.70 | 8.30 | 56.6% |
| target_4 | 7.28 | 12.60 | 57.8% |
| target_5 | 5.35 | 10.75 | 49.8% |

The output SDF keeps the original SMILES atom count and ordering, follows the
input target order, and tags each record with its score, percentage of the
maximum, and whether it is clash-free.

## Tests

The test suite covers the deterministic parts of the code: the scoring formula,
the clash rule, the alignment helpers against transforms we build by hand, the
feature tagging, the model parsing, and the SDF round trip. There is one small
end to end test that runs the real pipeline on a tiny molecule and checks basic
invariants. It does not assert exact docked scores, because the search is random
and the conformer step is approximate, so pinning numbers there would be flaky.

```bash
uv run pytest
```

They run on every push and pull request through GitHub Actions, see
`.github/workflows/ci.yml`.

## A note on reproducibility

The conformer relaxation step runs across all CPU cores for speed. Multithreaded
force field minimization is not bit-for-bit reproducible, so scores can wobble by
a small amount from one run to the next even with a fixed seed. If you need
identical output every time, set `numThreads=1` in `pipeline.py`, at the cost of
a slower run.

## Possible improvements

- Refine the winning pose against the actual score with a continuous optimizer,
  instead of relying on the distance-based alignment.
- Enumerate correspondences exhaustively when a site has only a few candidate
  atoms, rather than sampling them.
- Fall back to the UFF force field when MMFF parameters are missing for an atom.
