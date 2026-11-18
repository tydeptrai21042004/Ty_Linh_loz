# MR-SJPA paper experiments — three datasets, authors' baselines only

This is the **changes-only patch for `Ty_Linh_loz-main(2).zip`**, not an
independent project. The default research plan contains **111 real training
jobs**. No hyperparameter *confirmation* stage is scheduled. No synthetic
fusion controls are described or reported as independent paper baselines.

## DOI-traceable datasets

| key | Paper DOI | Download/converter | Geometry | Warning |
| --- | --- | --- | --- | --- |
| `vedai` | `10.1016/j.jvcir.2015.11.002` | GREYC official multipart download; RGB/NIR pairing, native OBB labels | native OBB | Official provider must remain reachable |
| `dronevehicle` | `10.1109/TCSVT.2022.3168279` | publicly named HF mirror of author release (`McCheng/DroneVehicle`); original paired XML polygons, crop handling | native OBB | Unofficial mirror; substantial disk space; RGB-side shared labels are a benchmark convention |
| `m3fd` | `10.1109/CVPR52688.2022.00571` | author Google Drive with public HF fallback; paired images and detection labels | HBB converted into axis-aligned quadrilaterals | **not** native oriented annotations; report separately |

The held-out view deliberately transcodes both modalities to common RGB JPEG
(quality=100, subsampling=0) so C²Former can reuse the exact source image bytes.
This increases dataset disk use and is not the original author-native test protocol.

All three are author-paper **DOIs**, not dataset-deposit DOIs. The download
adapter fails on missing pairs or annotations; there is no mock-data fallback.
Run acquisition separately if desired:

```bash
python layout_manager.py restore
pip install -r requirements-data.txt
python paper_three_datasets.py --accept-license --output datasets
```

Large downloads, third-party host access and original-author GPU compatibility
are **not verified** in this environment. Explicit license acknowledgement
is required. Default full-resolution DroneVehicle may exceed a single Kaggle
session's disk budget; prepare/persist one dataset at a time if needed.

## Training matrix (no confirmation stage)

| Component | Per dataset | All three datasets |
| --- | ---: | ---: |
| One-seed hyperparameter screening: default + 12 single-variable changes, 30 epochs | 13 | **39** |
| Final: MR-SJPA, official ETFNet, official C²Former × seeds 0/1/2, 100 epochs | 9 | **27** |
| Five MR-SJPA component ablations × seeds 0/1/2, 100 epochs | 15 | **45** |
| **Total** | **37** | **111** |

With an explicit `--without-c2former` flag, only the two source-supported
architectures are compared: **102 training jobs**. The five ablation IDs
(1–5) disable spatial correction, Procrustes correction, reliability gating,
trust projection, and RGB residual correction, respectively. The original
full proposal is shared with the main comparison; it is not retrained as a
sixth ablation. All five are components of the **proposed model**, never
independent literature baselines.

Hyperparameters: trust radius (0.10/0.25/0.50), coherence threshold
(0.20/0.30/0.40), groups (16/32/64), sample grid (8/16/24), spatial penalty
(0/.01/.03), shift margin (0/.01/.03). Run the default once and the other
two values for each dimension; select the best *validation* mAP50–95 after
30 epochs. Train the selected configuration **from scratch** for 100 epochs
and three seeds. Do not evaluate trial configs on held-out test images.

## GPU execution

One manifest command, safe to run on CPU:

```bash
python paper_research_pipeline.py --stage plan --work ./my_runs
```

Run everything in a working GPU environment with pinned original author
sources, accepted dataset license, and pre-existing compatible MMRotate
interpreter:

```bash
python layout_manager.py restore
pip install -r requirements-data.txt
pip install --no-deps -e .
python paper_research_pipeline.py --stage all --accept-license \
  --repo . --work /kaggle/working \
  --datasets vedai dronevehicle m3fd \
  --c2former-python /path/to/mmrotate03/python \
  --etfnet-sha FULL_OFFICIAL_COMMIT_SHA \
  --max-new-training-jobs 8 --max-new-evaluations 63
```

Or, **explicitly** omit C²Former if the required official-source environment
is unavailable by replacing `--c2former-python ...` with
`--without-c2former`. No fabricated replacement is used. For Kaggle, upload
the supplied full corrected ZIP and execute `Kaggle_Run_111_Experiments.py`
with Internet and GPU enabled. A dataset mirror may require Kaggle login
or manual provider authorization; an unattended transfer is not guaranteed.

Every training job records a settings hash and checks for a genuinely completed
checkpoint + validation CSV. Interrupted jobs resume from a saved checkpoint
when available; a change to configuration/data settings cannot silently reuse
an old run. Kaggle sessions do NOT preserve `/kaggle/working` automatically:
export its `research/` directory as a Kaggle output Dataset and mount it as
`PREVIOUS_RESEARCH_DATASET_DIR` in the next session. Preserve `datasets/` as
well to avoid expensive re-downloads. The script will not auto-overwrite
existing results.

## Robustness and evaluation

The final three-seed checkpoints are reused for **20 distortions + clean**:
Gaussian noise, blur, darkness, occlusion (RGB and IR separately, two
severities each), IR translation (two severities), and modality dropout (RGB
or IR). All transformations are deterministic and apply only to held-out test
images, with original labels unchanged. Each condition uses identical test
bytes across models and one common OBB polygon AP101 evaluator.

The complete plan adds **567 test evaluations** (21 × 3 methods × 3 seeds ×
3 datasets), **not** new trainings; excluding C²Former yields 378. The
clean and corrupted `dataset_digest` values necessarily differ, so fairness
checks are made within a condition, not by asserting that clean and noisy
inputs share a hash.

Statistics are measured from saved predictions: mean ± sample SD, Student-t
95% CI, paired-seed deltas, exploratory paired t-test and exact sign-flip
p-value. Three seeds are **not sufficient** for two-sided exact significance
at p<.05 (minimum possible p=.25). No training metrics are presented as test
results. Final CSV/LaTeX tables and PDF figures are generated only after
completeness/evaluator checks.

Timing is an **additional** native-model benchmark:

```bash
PYTHONPATH=/path/to/verified/source python paper_efficiency.py \
  --source /path/to/verified/source --checkpoint /path/to/best.pt \
  --output /path/to/latency.json
```

This measures native PyTorch model-forward latency, not identical NMS or
preprocessing, and does not invent FLOPs. Legacy MMRotate must be profiled
in its own environment and compared with equivalent settings.

## Restrictions and verification boundaries

- Original **ETFNet**: `https://github.com/lansimtech/ETFNet`, pinned git SHA;
  its code must actually import from the author's clone. A local CAFEM/TGF
  lookalike is not accepted as paper reproduction.
- Original **C²Former**: `https://github.com/yuanmaoxun/C2Former`, legacy
  MMRotate 0.3.x + its actual training/inference and polygon predictions.
  Its environment is **not** automatically built by the Kaggle script; no
  default modern Kaggle runtime is certified compatible.
- M3FD HBB converted polygons are **not native OBB**. Report results in a
  clearly labeled separate dataset row.
- Authors' dataset adapters, download mirrors and all complete GPU training
  runs remain **untested** here. Code-level and synthetic-format tests do not
  establish successful published-score reproduction.
- GPU run budget: 2,790 scheduled epochs per dataset (13×30 + 24×100),
  **8,370 scheduled epochs** across three datasets, excluding robustness and
  smoke tests. Training times vary with architecture and hardware.
