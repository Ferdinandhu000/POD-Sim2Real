# POD-Sim2Real

Reusable Sim-to-Real flow forecasting baselines for RealPDEBench foil data.
The project reads Hugging Face Arrow trajectory shards, audits Real/Sim
settings, renders synchronized videos, and trains a common set of U-Net, FNO,
AFNO and POD models with Sim pretraining followed by Real fine-tuning.

## Data

Large Arrow shards are intentionally excluded from Git. For the current small
local smoke dataset, mount them as:

```text
data/data_real/*.arrow
data/data_sim/*.arrow
```

Each record is decoded from the Arrow stream using `sim_id`, `shape_*`, and
binary field columns. Matching is by normalized `(Re, AoA)` setting. Matching
does not imply identical physical trajectories: Real and Sim fields are
separate domain samples. Models train on `u/v`; pressure is ignored.

For a complete RealPDEBench Foil download on a server, preserve the official
Hugging Face directory rather than copying only the `.arrow` shards:

```text
data/
└── foil/
    ├── hf_dataset/
    │   ├── real/                       # dataset_info.json, state.json, *.arrow
    │   ├── numerical/                  # dataset_info.json, state.json, *.arrow
    │   ├── train_index_real.json
    │   ├── val_index_real.json
    │   ├── test_index_real.json
    │   ├── train_index_numerical.json
    │   ├── val_index_numerical.json
    │   └── test_index_numerical.json
    ├── remain_params_real.json
    ├── in_dist_test_params_real.json
    ├── out_dist_test_params_real.json
    ├── remain_params_numerical.json
    ├── in_dist_test_params_numerical.json
    └── out_dist_test_params_numerical.json
```

The full directory is required because the loader uses
`datasets.load_from_disk`; `state.json` and `dataset_info.json` describe the
Arrow data and must remain next to it. In this layout the trainer detects the
official index files automatically. Each index entry is one exact temporal
window `(sim_id, time_id)`, so the repository does not randomly re-split the
98 Real and 99 Numerical trajectories.

## Install and audit

Use an environment containing PyTorch and PyArrow (the existing `RealPDEBench`
environment already has PyArrow):

```bash
python -m pip install -e .
python -m pod_sim2real.visualization.audit_dataset --data-root .
python -m pod_sim2real.visualization.render_pairs --data-root . --max-frames 0
```

`--max-frames 0` renders the complete trajectory. The Arrow records currently
contain 3990 frames (about 20 seconds at the source `dt=0.005`); the MP4 is a
full-frame inspection video, so its display duration is controlled by `--fps`.
The `real_id`/`sim_id` values in `pairing.csv` are embedded logical identifiers
(often the original `.h5` basename), while `real_arrow`/`sim_arrow` are the
actual files used for reading.

For readable domain comparison, videos use a fixed full-trajectory color scale
for each `(domain, channel)` pair. Real and Sim therefore have independent
u/v/speed limits; the limits are shown in each panel and stored in the video
manifest. The scale never changes between frames.

Outputs are written under `artifacts/audit` and `artifacts/videos` and are
ignored by Git.

## Smoke training

```bash
python -m pod_sim2real.training.train --config configs/smoke.yaml --model unet
python -m pod_sim2real.training.train --config configs/smoke.yaml --model fno
python -m pod_sim2real.training.train --config configs/smoke.yaml --model afno
python -m pod_sim2real.training.train --config configs/smoke.yaml --model pod-fno
python -m pod_sim2real.training.train --config configs/smoke.yaml --model pod-afno
python -m pod_sim2real.training.train --config configs/smoke.yaml --model pod-unet
python -m pod_sim2real.training.train --config configs/smoke.yaml --model pod-itransformer
python -m pod_sim2real.training.train --config configs/smoke.yaml --model itransolver
python -m pod_sim2real.training.train --config configs/smoke.yaml --model pod-itransolver
```

Each run creates `artifacts/runs/<model>/` with `pretrain_sim` and
`finetune_real` stages, checkpoints, JSON metrics, and logs. tqdm progress is
shown in the terminal and epoch metrics are written to `logs/train.log` and
`metrics.jsonl`.

## Full runs

### Baseline: first third of every trajectory

`configs/baseline.yaml` is the reproducible baseline configuration. It uses
the official trajectory/parameter-level assignment from the six
`*_index_{real,numerical}.json` files, so a complete filename belongs to only
one of train, validation, or test. It then uses only the first third of every
assigned 3990-frame trajectory (1330 frames). The default 32x64 resolution is
a memory-conscious view of the original 128x256 fields; change it to
`[128, 256]` for a full-resolution run. If the official index files are not
available, the same mode falls back to a deterministic trajectory split using
the configured seed and fractions. A trajectory is never divided between
train, validation, and test; the prefix is taken only after assignment.

Prefix experiments use `split_mode: trajectory_prefix`. The former
`temporal_prefix` mode is unsupported because it could place different time
ranges of one trajectory in different splits.

Run the baseline U-Net on the server with:

```bash
python -m pod_sim2real.training.train \
  --config configs/baseline.yaml \
  --model unet \
  --data-root .
```

The split counts, exact trajectory IDs, and exact settings are saved in
`artifacts/baseline_first_third/unet/split_manifest.json`; final test MSE is
written to `artifacts/baseline_first_third/unet/results.json`. To run another baseline architecture,
reuse the same config and change only `--model` to `fno` or `afno`.

With the official directory above, use the supplied configuration:

```bash
python -m pod_sim2real.training.train \
  --config configs/foil_official.yaml \
  --model pod-fno \
  --data-root .
```

`--data-root data` is also accepted when the server mounts the data directory
directly. The program first pretrains on `numerical/train_index_numerical.json`,
then fine-tunes on `real/train_index_real.json`, validates using the matching
official validation index, and reports `real_test_mse` on the official real
test index in `results.json`.

The default `test_mode: all` evaluates every test window. To match a standard
Foil subset, set `data.test_mode` in a copied config or pass one of:

```bash
--test-mode seen
--test-mode in_dist
--test-mode out_dist
--test-mode unseen
```

`seen` selects IDs in `remain_params_*`; `in_dist` and `out_dist` select their
respective official parameter lists; `unseen` is their union. As in the
upstream loader, a selected test mode filters the official validation and test
windows, while training windows remain unchanged.

The complete Foil data is large. `OfficialArrowWindowDataset` loads metadata
and the requested trajectory row lazily, then decodes only the requested
`input_steps + output_steps` window at runtime. Do not call
`discover_trajectories` on the complete official dataset, since that fallback
is intended only for the small standalone Arrow smoke files.

Change resolution, epoch counts, `pod_rank`, batch size and paths in a copied
YAML configuration. No code uses machine-specific absolute paths, so the same
commands work on Linux servers.
