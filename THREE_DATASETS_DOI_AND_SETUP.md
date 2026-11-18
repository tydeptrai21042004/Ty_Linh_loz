# Three-dataset reproducible benchmark (MR-SJPA)

## Scientific protocol

The three requested datasets all have traceable **paper DOIs**. These are not
claimed as dataset-deposit DOIs. Never substitute a synthetic source or silently
convert HBB into "native OBB" ground truth.

| Alias | Original paper (DOI) | Native OBB? | Source |
|---|---|---|---|
| `vedai` | Razakarivony & Jurie, *Vehicle detection in aerial imagery: A small target detection benchmark*, `10.1016/j.jvcir.2015.11.002` | Yes | Official `https://downloads.greyc.fr/vedai/` |
| `dronevehicle` | Sun et al., *Drone-Based RGB-Infrared Cross-Modality Vehicle Detection Via Uncertainty-Aware Learning*, `10.1109/TCSVT.2022.3168279` | Yes | Author project `https://github.com/VisDrone/DroneVehicle`; automated public but **unofficial** HF mirror `https://huggingface.co/datasets/McCheng/DroneVehicle` |
| `m3fd` | Liu et al., *Target-Aware Dual Adversarial Learning and a Multi-Scenario Multi-Modality Benchmark To Fuse Infrared and Visible for Object Detection*, `10.1109/CVPR52688.2022.00571` | **No**: converted axis-aligned polygons | Author project `https://github.com/JinyuanLiu-CV/TarDAL` with public HF mirror `https://huggingface.co/datasets/nonameplease/M3FD_Detecion` as fallback |

## One command

```bash
python layout_manager.py restore
pip install -r requirements-data.txt
python paper_three_datasets.py --accept-license --output datasets
```

The script sequentially downloads each release, converts to synchronized RGB/IR
pairs with canonical OBB-compatible labels, ensures non-overlapping
train/validation/test sets, validates all splits, caches downloads, preserves
source-format images using hardlinks where possible to avoid unnecessary large
JPEG copies, and writes
`datasets/THREE_DATASET_BENCHMARK.json`. An automatic download can still fail
because of network, provider gating, changing mirrors, disk quota, or license
restrictions. It aborts on a source-format error; it never fabricates images.

You can isolate one dataset:

```bash
python paper_three_datasets.py --datasets vedai --accept-license
python paper_three_datasets.py --datasets dronevehicle --accept-license
python paper_three_datasets.py --datasets m3fd --accept-license
```

For provider fallback using source archives already on disk:

```bash
python paper_three_datasets.py --datasets dronevehicle --archive dronevehicle=/kaggle/input/dronevehicle --accept-license
```

For small integration tests only, add `--quick-limit 24`. Do not report the
reduced dataset as a full benchmark. Default dataset size is substantial:
DroneVehicle mirror archives alone are ~14 GB, excluding extraction, caching
and generated images. Kaggle sessions may need more storage or individual
dataset jobs rather than an entire corpus downloaded together.

## Label and preprocessing policy

- VEDAI: RGB/NIR pairing, source native polygons; official class-ID handling.
- DroneVehicle: original split folders and RGB-side XML polygon annotations;
  IR labels may differ. Crop 100-pixel borders when source images are 840×712,
  transform polygons to 640×512, reject any missing pair/annotation. Using
  RGB-side boxes as shared labels is a declared **new evaluation convention**,
  not a claim of perfect alignment or official shared annotation.
- M3FD: source HBB/YOLO labels become four-point *axis-aligned polygons*;
  an OBB evaluator will score those polygons but this does not test angle
  estimation as a native OBB task.
- Native source test data remains test-only. Where no source test exists, a
  deterministic reservation of source training images creates an independent
  test split, leaving validation untouched.
- The same dataset YAML, class map, split IDs and OBB evaluator must be passed
  to all published baselines. Each method's environment must be verified.

## Experiment counts for 3 datasets

The previously agreed experimental design is **37 training runs per dataset**:
13 single-seed sweep jobs, 9 main comparison trainings (3 methods × 3 seeds),
15 proposed-method ablations (5 × 3 seeds). Across three datasets this becomes
**111 training jobs**. Re-evaluations for robustness do not require retraining.

This script implements the **data stage**, not automatic orchestration of all
111 training jobs. It does not claim full GPU reproduction or official paper
numbers from re-evaluating these papers on different data splits.

## Paper baseline Kaggle runner for any of the three datasets

The existing author-source benchmark now accepts `--dataset {vedai,dronevehicle,m3fd}`
or `--all-datasets`. In a GPU Kaggle notebook, upload the full repository ZIP
and set `ETFNET_DATASET` environment variable before executing
`Kaggle_Run_PaperOnly_E2E.py`. Example:

```python
import os
os.environ['ETFNET_DATASET'] = 'm3fd'  # or 'vedai', 'dronevehicle', 'all'
exec(open('/kaggle/input/YOUR_SCRIPT/Kaggle_Run_PaperOnly_E2E.py').read())
```

Or with a restored checkout:

```bash
python paper_benchmark_runner.py --repo . --dataset dronevehicle --work /kaggle/working --accept-license
```

Each selected dataset gets separate outputs. **This runner implements 3-seed
MR-SJPA vs author ETFNet attempts only**, not the entire 37-run per-dataset
sweep and ablation plan; original C²Former still needs its compatible isolated
MMRotate Python. Do not report unverified official-paper numbers.
