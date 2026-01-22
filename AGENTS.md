# Petri Dish NCA - AGENTS

This repo implements Petri Dish Neural Cellular Automata (PD-NCA): multiple NCA
agents share a grid, compete/cooperate via attack/defense channels, and update
their parameters continuously during the simulation. The code follows the paper's
core loop of proposal -> competition -> normalization -> state update, with a
background "sun" competitor and per-agent learning. Use this file as your map
when adding features, fixing bugs, or asking questions.

Maintenance note: when you make meaningful code changes (new features, config
fields, workflows, or behavior changes), update this AGENTS.md so future agents
and users have an accurate map.

## Quickstart
- Install: `uv sync`
- Train: `uv run python src/train.py --n-ncas 3 --epochs 1000 --device cpu`
- Train from config: `uv run python src/train.py --config configs/example.json`
- Live metrics/plots: `uv run python src/train.py --n-ncas 3 --epochs 1000 --device cpu --live-viz`
- Visualize a trained run: `uv run python src/visualize_trained.py --model-path <run_dir>`
- 3D grid size via CLI: `uv run python src/train.py --n-ncas 3 --epochs 200 --grid-size 10 10 10`
- 3D view with grid override: `uv run python src/visualize_trained.py --model-path <run_dir> --grid-size 10 10 10`
- 3D plotly view with mouse controls + playback: `uv run python src/visualize_trained.py --model-path <run_dir> --plotly`
- Plotly playback speed: `uv run python src/visualize_trained.py --model-path <run_dir> --plotly --plotly-speed 0.5`

## Repo Layout
- `src/config.py`: Config dataclass, validation, device/seed setup, JSON load/save.
- `src/model.py`: Core PD-NCA model, competition, and optimization logic.
  - `MergedCAModel`: per-agent update proposal network (grouped conv).
  - `CAEntity`: model + optimizer + gradient normalization.
  - `CASunGroup`: competition, update resolution, sun background, save/load.
- `src/world.py`: World state, grid pool, feature hooks, step scheduling.
- `src/train.py`: CLI, training loop, wandb logging, optional live visualization.
- `src/viz.py`: Snapshots, color mapping, 3D slice stacks (matplotlib/plotly), entropy/compression metrics, video export.
- `src/visualize_trained.py`: Load a saved run and render a live simulation (3D slice stacks + plotly final view).
- `configs/*.json`: Example configs.
- `notebooks/interactive_viz.ipynb`: exploration/visualization notebook.

## Core Data Flow (per epoch)
1) `World.get_seed()` selects a batch from the pool.
2) `World.step()` runs:
   - optional burn-in steps without gradients
   - steps with gradients via `CASunGroup.__call__`
3) `CASunGroup.update_models()` computes per-agent losses (territory coverage),
   backprops, updates model + optional sun.
4) Metrics logged (wandb/terminal) and pool updated.

## Key Concepts Mapped to Code
- **Aliveness channels**: `CASunGroup.ali_idxs` and `World._init_pool`.
- **Attack/Defense channels**: `CASunGroup.att_idxs` / `def_idxs`.
- **Competition**: `_run_competition_parallel()` (cosine similarity).
- **Normalization**: `softmax_temp` and aliveness redistribution.
- **Background competitor ("sun")**: `CASunGroup._setup_sun_update()`.
- **Learning in-the-loop**: `update_models()` each epoch.

## Paper Alignment Cheatsheet
- **Four phases**: processing -> competition -> normalization -> state update
  corresponds to `CASunGroup._parallel_forward_step()` and
  `CASunGroup._run_competition_parallel()`.
- **Background environment**: paper uses a static environment tensor; this
  code implements it as a learnable "sun" update vector (optionally updated).
- **Aliveness threshold**: fixed at 0.4 in `CASunGroup` (not configurable).
- **Objective**: maximize territorial coverage; implemented via an asinh/log-like
  growth term and summed losses in `update_models()`.

## Configuration Notes
- `grid_size` is `(D, H, W)`; 2D `(H, W)` configs are still accepted and promoted to `(1, H, W)`.
- `cell_state_dim` must be even (split into attack/defense).
- `batch_size <= pool_size`.
- `batch_size` can be overridden via `--batch-size` on `src/train.py`.
- `visualize_trained.py` also supports `--batch-size` to override the saved config during visualization.
- `n_seeds * n_ncas <= grid_size[0] * grid_size[1] * grid_size[2]`.
- `alive_visible` controls whether aliveness channels are visible to models.
- `steps_before_update` + `steps_per_update` are the epoch schedule.
- `burn_in` uses `SimpleBurnInFeature` to ramp step counts.
- `sun_update_epoch_wait` controls when the sun updates.
- Visualization-only controls: `viz_slice_axis`, `viz_slice_stride`, `viz_slice_spacing`, `viz_slice_alpha`, `viz_max_slices`.
- Plotly visualization includes a slice-end slider to hide later slices.
- Plotly visualization shows NCA population lines (percent occupied) up to the current step.

## Adding Features (recommended pattern)
1) Add new config fields in `src/config.py` with validation if needed.
2) For training-time behavior, create a `Feature` subclass in `src/world.py`.
3) Wire it into `World._build_features()` based on config flags.
4) If the feature affects learning/competition, update `CASunGroup` accordingly.
5) Update `src/train.py` CLI + `README.md` and add an example in `configs/`.

## Debugging + Common Pitfalls
- **Device mismatch**: `Config.__post_init__()` falls back to CPU if CUDA/MPS
  unavailable. Ensure tensors are created on `config.device`.
- **Dtype issues**: `World` uses `bfloat16` on CUDA; keep ops compatible.
- **Sun/seed loading**: `CASunGroup.load()` strips `_orig_mod.` keys if needed.
- **Grid value range**: visualization metrics assume grid values in [-1, 1].

## Saved Artifacts
Each training run saves a folder named by timestamp (or wandb run name) containing:
- `config.json`
- `sun.npy`
- `model.pt`
- `seed.npy`

## Asking Questions / Filing Bugs
When asking for help or filing an issue, include:
- Exact command and config file used.
- Device + torch version.
- Error stack trace or unexpected behavior description.
- Whether `--live-viz` or `--wandb` was enabled.
- If reproducible, the saved run directory.

## Suggested Extensions
- Larger grids / more NCAs to probe open-endedness.
- New competition rules (alternate similarity functions, different thresholds).
- Additional metrics (entropy, compression, novelty scores).
- Multi-GPU or distributed training for large-scale experiments.
