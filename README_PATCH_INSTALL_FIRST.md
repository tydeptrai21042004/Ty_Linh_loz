# Four paper-supported RGB–IR baselines: corrected integration (20 Oct 2026)

**Start here.** This archive contains **only new integration files** for your `Ty_Linh_loz-main` GitHub repository. The original SJPA-TR layers, OBB head, experiments, and training code are **not modified**.

## What this patch actually provides

| Paper-supported baseline | Geometry | Implemented now | NOT claimed as done |
| --- | --- | --- | --- |
| C²Former, IEEE TGRS 2024, [DOI](https://doi.org/10.1109/TGRS.2024.3376819) | OBB | Official Git-source fetch/verification, correct `*_tir` paired image/annotation view, generated child MMRotate model/data config, official train/test launcher | Trained end-to-end model or measured OBB mAP |
| SuperYOLO, IEEE TGRS 2023, [DOI](https://doi.org/10.1109/TGRS.2023.3258666) | HBB | Official SR-YOLO MF full-branch launcher, strict 512/1024 input views, author YAML filename, guarded source path compatibility patch, stable source-ID map | Accurate model without real independently provided high-res 1024 data |
| ICAFusion, Pattern Recognition 2023, [DOI](https://doi.org/10.1016/j.patcog.2023.109913) | HBB | Paired-YOLO train/eval YAML, separated val and TEST labels, original-author model launcher, fixes author hard-coded eval path and wrong prediction TXT formatting | Trained checkpoint or independently verified author full GPU test |
| CFT, [arXiv:2111.00273](https://arxiv.org/abs/2111.00273) | HBB | Paired-YOLO dataset view, original transformer-x3 architecture guard and author train/test commands | Published journal status or paper-reproduction numerics |

**The author source code and weights are not bundled**. It must be cloned in Kaggle or on a GPU machine with Internet, with environments pinned to each original project's requirements. Old third-party frameworks are not guaranteed compatible with the Kaggle default environment. Do not report results as reproduced until official original code finishes training/testing on the actual common dataset.

## Quick offline verification (works without downloading external code)

```bash
python -m pip install pyyaml pillow
python -m unittest -v test_external_paper_baselines test_official_baseline_runs test_common_hbb_eval
python external_paper_baselines.py list
```

## Fetch actual four model implementations

```bash
python external_paper_baselines.py fetch --method all --workspace external_upstream
```

This command clones *only* the author-released repos and records Git commit hashes. It requires Internet. Each repo likely needs an isolated legacy environment. Do not try to run all four in a single unpinned package environment.

## Prepare dataset views

**Source input:** Your project paired RGB/IR OBB YAML must provide `path`, `names`, and for each `train`, `val`, `test`: keys `split` (RGB image directory), `split_ir` and `split_labels`. All source OBB label lines are 9 columns: `class x1 y1 x2 y2 x3 y3 x4 y4`, normalized to [0,1]. The IR image must share each RGB filename. Split filenames cannot overlap.

```bash
# HBB label conversion: shared with ICAFusion and CFT
python external_paper_baselines.py prepare-hbb \
  --data /path/to/data_rgb_ir_obb.yaml \
  --output /workspace/hbb_data
python official_baseline_datasets.py legacy --hbb-data /workspace/hbb_data/data_hbb.yaml

# Paired DOTA-compatible OBB view for author's C2Former DroneVehicleDataset
python official_baseline_datasets.py c2former \
  --obb-data /path/to/data_rgb_ir_obb.yaml \
  --output /workspace/c2former_paired
python create_c2former_config.py --repo external_upstream/c2former_obb \
  --view /workspace/c2former_paired 
```

The generated child config automatically uses the exact ordered class names from the C²Former view manifest; verify them before launch. The C²Former child config is written under its original source checkout; it inherits the author config and modifies only classes and data roots. C²Former input is trans-coded to JPG because its original paired loader requires `.jpg`. If other models are evaluated using different pixel decodes, explain this limitation; avoid claiming strict pixel-equivalent settings.

**ICAFusion I/O patch**: Run once after source clone and before standalone testing:

```bash
python official_upstream_compat.py --method icafusion_hbb --repo external_upstream/icafusion_hbb
```

It updates `test.py` output formatting to the predicted class + normalized YOLO boxes + confidence, and reads labels from dataset YAML. A `.orig` backup and SHA256 audit receipt are produced. Neither detector layers nor inference/NMS are changed.

**SuperYOLO full-paper SR baseline:** You need *two separately sourced* paired labeled datasets with identical 512×512 and 1024×1024 scene IDs (no fake 1024 obtained by upsampling 512). Convert *each* to HBB, then:

```bash
python official_baseline_datasets.py superyolo \
  --low-hbb /workspace/low_hbb/data_hbb.yaml \
  --high-hbb /workspace/high_hbb/data_hbb.yaml \
  --output /workspace/superyolo_view
python official_upstream_compat.py --method superyolo_hbb \
  --repo external_upstream/superyolo_hbb \
  --data-root /workspace/superyolo_view
```

Adapter generates `SRvedai.yaml` deliberately: the author's train script chooses its paired SR loader by that exact filename. It points train images to 1024 and validation to 512. It never creates fake HR. Its test output image IDs must be remapped using `id_mapping.json`.

## Launch original networks (dry-run by default)

Launchers are **plans only** unless `--execute` is supplied. For example:

```bash
# C2Former original MMRotate/S2ANet fusion and OBB head
python official_baseline_runs.py --method c2former_obb --workspace external_upstream \
  --data /path/to/data_rgb_ir_obb.yaml \
  --model external_upstream/c2former_obb/configs/s2anet/sjpa_c2former_paired_local.py \
  --out /workspace/runs/c2former --ack-dataset-adapter

# Full SuperYOLO with correct SR config, requiring authentic 1024 HR input
python official_baseline_runs.py --method superyolo_hbb --workspace external_upstream \
  --data /workspace/superyolo_view/SRvedai.yaml \
  --model external_upstream/superyolo_hbb/models/SRyolo_MF.yaml \
  --out /workspace/runs/superyolo --super-resolution --ack-sr-data

# ICAFusion (replace model config with your verified author fusion architecture)
python official_baseline_runs.py --method icafusion_hbb --workspace external_upstream \
  --data /workspace/hbb_data/data_legacy_rgb_ir.yaml \
  --model /ABS/PATH/INSIDE/external_upstream/icafusion_hbb/author_ICAFusion_model.yaml \
  --out /workspace/runs/icafusion --ack-paper-arch

# CFT (must select author's paper transformerx3 config, not a generic transformer)
python official_baseline_runs.py --method cft_hbb --workspace external_upstream \
  --data /workspace/hbb_data/data_legacy_rgb_ir.yaml \
  --model /ABS/PATH/INSIDE/external_upstream/cft_hbb/author_transformerx3_model.yaml \
  --out /workspace/runs/cft --ack-paper-arch
```

Replace placeholder paths with *real original author config files*. Supply `--execute` only after verifying data, original model, weights, CUDA dependencies. For standalone heldout ICAFusion test, create a separate view using `python official_baseline_datasets.py icafusion-test --hbb-data /workspace/hbb_data/data_hbb.yaml --output /workspace/ica_testview`. Supply `--phase test --checkpoint /path/best.pt --data /workspace/ica_testview/data_icafusion_heldout.yaml`. It intentionally uses TEST annotations only after checkpoint/architecture selection is finished. **CFT needs a heldout dataset alias too**: its original standalone `test.py` loads `val_rgb` and `val_ir` even when passed `--task test`; therefore supply the **same separate TEST-view YAML** (the `data_icafusion_heldout.yaml` filename is just an adapter label) for CFT testing, rather than its training YAML.

## One common HBB evaluator with **real** predictions

For original baselines configured with `--save-txt --save-conf`, locate the **actual** upstream `labels/` prediction output directory. It must contain six whitespace-separated columns per prediction: `class xc yc w h confidence` normalized to [0,1], one file per image ID. The ICAFusion I/O patch fixes the author's erroneous alternative output. Then:

```bash
python paper_common_hbb_eval.py --method icafusion_hbb \
  --hbb-data /workspace/hbb_data/data_hbb.yaml \
  --pred-dir /workspace/runs/icafusion/test/labels \
  --checkpoint /workspace/runs/icafusion/weights/best.pt \
  --out /workspace/metrics/icafusion
python paper_common_hbb_eval.py --method cft_hbb \
  --hbb-data /workspace/hbb_data/data_hbb.yaml \
  --pred-dir /workspace/runs/cft/test/labels \
  --checkpoint /workspace/runs/cft/weights/best.pt \
  --out /workspace/metrics/cft
python paper_common_hbb_eval.py --method superyolo_hbb \
  --hbb-data /workspace/hbb_data/data_hbb.yaml \
  --pred-dir /workspace/runs/superyolo/test/labels \
  --superyolo-id-map /workspace/superyolo_view/id_mapping.json \
  --checkpoint /workspace/runs/superyolo/weights/best.pt \
  --out /workspace/metrics/superyolo
python verified_fair_comparison.py \
  --receipts /workspace/metrics/icafusion/icafusion_hbb_receipt.json \
             /workspace/metrics/cft/cft_hbb_receipt.json \
             /workspace/metrics/superyolo/superyolo_hbb_receipt.json \
  --out /workspace/hbb_paper_comparison.csv
```

**Important:** Replace example output paths with those actually created by original training/testing. The code computes 101-recall-point AP50 and AP50:95 without COCO crowd/maxDets rules; **do not describe these as official COCO metrics**. The comparison checker refuses divergent geometry, classes, test-image bytes, split IDs, metrics, input size, postprocessing text and missing/checksum-invalid evidence files. It cannot independently prove weight provenance, equivalent training effort, authentic SR source, NMS equivalence, or visual preprocessing identity.

## FAIR scientific protocol

1. OBB table **SJPA-TR / ETFNet / C²Former**: exactly the same paired train/val/test, class order, rotated box convention and **the same OBB evaluator**. C²Former outputs an MMRotate results `.pkl` with its own box convention: **not convertible to SJPA's OBB mAP by relabeling upstream mAP**. This patch does NOT claim a complete common OBB evaluation conversion yet.
2. HBB table **SuperYOLO / ICAFusion / CFT**: use exactly the same paired train/val/test identities and OBB→HBB ground-truth conversion; preferably include retrained SJPA-TR and ETFNet configured for HBB if you want a direct ranking versus the proposal. **Do not put SJPA-TR OBB AP in this HBB table.**
3. Keep separate official-reproduction vs adapted-cross-dataset performance labels. Original-paper mAP from FLIR/VEDAI/DroneVehicle cannot be inserted as observed mAP on your project's heldout data.
4. Report image size, class mapping, learning schedule, seed, batch, augmentations, checkpoint choice on validation only, hardware, FPS measurement boundaries and confidence intervals. Original author models may use inherently different input sizes/training budgets; record and discuss rather than claiming equality by default.
5. Preserve *unaltered* original SJPA-TR network; new files live entirely at repo root.

## If browser says “Không thể hoàn thành thao tác này”

Use the shorter named patch ZIP, or the **full merged project ZIP** provided alongside it. Download via the clickable sandbox links in the assistant's answer. No SVG or embedded browser dependency is required to extract the archive.
