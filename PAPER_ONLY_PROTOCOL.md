# Published-only experimental protocol

## Eligible entries

1. **MR-SJPA**: proposal model, not a published baseline.
2. **ETFNet**, Nguyen, Tran & Nguyen, *Information Fusion* (2026), DOI
   10.1016/j.inffus.2026.104658, original authors'
   https://github.com/lansimtech/ETFNet. Its original benchmarks are
   DroneVehicle, VTUAV-Det and RGBTDronePerson. The supplied Kaggle runner uses
   the original authors' source but VEDAI-512 for cross-dataset evaluation:
   do not call its VEDAI scores a published-metric reproduction.
3. **C²Former**, Yuan & Wei, *IEEE TGRS* (2024), DOI
   10.1109/TGRS.2024.3376819, original authors'
   https://github.com/yuanmaoxun/C2Former, OBB/S²ANet/MMRotate. This needs
   its own environment and adapted matched RGB/IR dataset config.

No Concat/Add/CAFEM-only/TGF-only/GOCI/RTPF entries appear in the paper-baseline
ranking. Other local fusion modules remain solely in historical unit tests.

## Rules for accepting a paper-baseline run

- Upstream author repository URL, full git commit SHA and exact config checksum.
- Full training run with actual checkpoint and actual per-example prediction files.
- Same dataset version, class mapping, patient/image scene split, annotation
  geometry, input resolution, image normalization, evaluation IoU definition,
  and mAP calculation. Different detection head/training recipe are recorded.
- Any converted HBB annotations are clearly described as HBB, never native OBB.
- Results on validation and test are separate; never report the best validation
  epoch as an independent test result.
- No synthetic scores, manual copied published mAP or output of a failed run.
- C²Former predictions must be evaluated with a rigorously matched OBB evaluator,
  or listed in a separate table if evaluator cannot be harmonized.

## Important implementation limitations

- MR-SJPA uses per-sample covariance from sampled spatial points, not averaged
  anchors (avoids systematic resolution-dependent whitening magnification).
- Zero-initialized convolutional residual adapters and Frobenius projection
  guarantee a bounded difference from raw concatenation.
- Spatial shift selection uses overlap-only, normalized cross-covariance and a
  confidence margin, without the old absolute score gate. Candidates do not
  retain full-resolution activation stacks.
- Shift correlation uses the SAME Frobenius score in inference and tracing.
- MR-SJPA has no certified ONNX/TensorRT export, and its per-sample SVD/eigh
  should not be silently replaced with other operations for export.
- Inference benchmark, GPU time, mAP and official reproduction remain UNTESTED
  in this environment. Upstream compatibility may require a verified dataset
  adapter; errors are not converted into success.

## C²Former original-source commands (isolated GPU environment)

```bash
python external_paper_baselines.py fetch --method c2former_obb --workspace external_upstream
python official_baseline_datasets.py c2former \
    --obb-data /path/to/shared_obb_data.yaml \
    --output /path/to/c2former_paired
python create_c2former_config.py \
    --repo external_upstream/c2former_obb \
    --view /path/to/c2former_paired
```

Find the generated config filename in the printed JSON. After installing the
original MMRotate dependencies in a **separate Python environment**, use its
interpreter for training and testing:

```bash
python official_baseline_runs.py --method c2former_obb \
    --workspace external_upstream --data /path/to/shared_obb_data.yaml \
    --model /path/to/generated_c2former_child_config.py \
    --out /path/to/c2former_runs --ack-dataset-adapter --execute
```

The launcher verifies the official source, original-derived child config,
paired data adapter acknowledgement and checkpoint paths. A successful command
is **not** automatically an equivalently evaluated paper result; record test
predictions and evaluate with the common OBB evaluator before comparison.

## Known Kaggle limitations

The downloaded VEDAI dataset must be usable by the original ETFNet author's
six-channel input and paired-data loader. The exact upstream environment and
GPU compatibility were not tested here. Any incompatibility raises an error,
not a substituted local control. Dataset checksum, author commit, model
config checksum and imported `ultralytics` source path are recorded for audit.
