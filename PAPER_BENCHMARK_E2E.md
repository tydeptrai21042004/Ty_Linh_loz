# Paper-backed RGB/NIR OBB benchmark: implementation and limitations

## Research comparison

- **Proposed**: MR-SJPA, corrected residual fusion implementation in this repository.
- **ETFNet (Information Fusion 2026)**: original authors' implementation, `https://github.com/lansimtech/ETFNet`, fetched with origin / model path / commit recorded. **Not** a local pseudo-ETFNet fusion block.
- **C²Former (IEEE TGRS 2024)**: original authors' MMRotate implementation, `https://github.com/yuanmaoxun/C2Former`, trained/tested only in a compatible **separate** legacy Python/CUDA environment. Neither architecture is replaced by YOLO.

No Concat, Add, GOCI, RTPF, CAFEM-only, TGF-only or other custom fusion modules are included as independent paper baselines. The repository may still contain their unit tests for regression coverage.

## End-to-end execution

On Kaggle, upload `Ty_Linh_Paper_Benchmark_Full_Corrected.zip` as a Dataset; turn **Internet** and **GPU** on. Open `kaggle/PaperOnly_ThreeSeed_E2E_Kaggle.py`, execute the script. The runner:

1. Restores the repository using its source-layout manifest; checks imports and original ETFNet source origin.
2. Downloads the VEDAI RGB/NIR archive from the public official source using the existing resumable downloader. Network/provider failures stop the run. License acceptance is passed explicitly; confirm you have reviewed VEDAI usage terms.
3. Converts paired images/OBB labels and makes a new deterministic heldout test set by reserving 20% of *source training* pairs, retaining source validation unchanged. A source-provided test split is preserved. The three sets cannot share image identifiers. This is a **new** VEDAI split, not the paper's original DroneVehicle result.
4. Transcodes **both** RGB and IR modalities to common JPEG quality 100/subsampling 0. C²Former adapter preserves these exact JPEG image bytes instead of recompressing. The common evaluator measures image-level polygon IoU.
5. Trains the proposal and **authors' ETFNet** for matched seeds **0, 1, 2**, epochs, batch and input size; logs provenance and checkpoints. Resumes interrupted runs if a real checkpoint exists. **GPU/author compatibility has not been verified**; fail closed rather than fabricating.
6. Selects `best.pt` based on training/validation, then runs the **held-out test** once using `model.val(split='test', save_json=True)`, with common confidence/NMS flags. The actual detected polygon files are saved by the validator.
7. Evaluates saved OBB polygons with one common AP101 function for both methods, calculating mAP@0.5, mAP@0.5:0.95, per-class AP, image IDs, checkpoint SHA-256, source SHA, dataset integrity and split receipt.
8. Calculates mean ± sample SD across 3 seeds, Student t 95% confidence interval, matched seed deltas, exploratory paired t-test and exact sign-flip test. **Three seeds cannot establish strong significance**; two-sided exact sign-flip p has minimum 0.25.
9. Produces `per_seed_test_results.csv`, `paper_table.csv`, `paper_table.tex`, `statistical_report.json`, per-seed receipts and `PAPER_ONLY_THREE_SEED_RESULTS.zip`.

### Optional C²Former three-seed run

The separate `paper_c2former_3seed.py` script fetches the original official source, creates the original MMRotate `DroneVehicleDataset` image/annotation layout and an inheriting (not rewritten) child config, runs 3 seeds with upstream `tools/train.py` and `tools/test.py`, converts **actual** mmrotate predictions.pkl into oriented quadrilaterals, then feeds the same common OBB evaluator. It needs an existing compatible `--python` MMRotate 0.3.x interpreter; pip/torch/CUDA builds cannot be guaranteed on modern Kaggle. The upstream repository ships no verified Kaggle-install environment.

```bash
python paper_c2former_3seed.py \
  --python /path/to/legacy-mmrotate-python \
  --data /kaggle/working/vedai_heldout_paper/data.yaml \
  --workspace /kaggle/working/c2former_work --epochs 100
```

To combine the nine completed receipts, use `paper_benchmark_stats.py --receipts` with 3 ETFNet, 3 MR-SJPA and 3 C²Former common evaluator receipt JSONs. The aggregator **refuses** mixed evaluator/splits/class order. Results cannot be described as authors' originally reported benchmark metrics.

### Known limitations and verification boundaries

- The files implement the pipeline, but the full public download and all six/nine GPU trainings have **not** been executed in this environment. Especially, author ETFNet source and MMRotate may require compatibility patches; a successful local unit test does not certify GPU reproduction.
- Both YOLO-style systems share OBB AP101 on saved polygons; their own NMS implementations must still be audited (identical threshold parameters alone do not prove identical NMS algorithms).
- Current heldout split deliberately reserves some source training pairs as test rather than reusing validation. It is an **explicit new experimental protocol**. No original official ETFNet-published mAP is claimed.
- The C²Former model uses a different head and framework. Common polygon AP makes post-hoc scoring comparable, but architecture, training schedule and preprocessing must still be described.
- Source `orig` branch model may fail if its official six-channel loader cannot read the new paired YAML. This is a **hard error**, not silently corrected with the local imitation. An author-code compatibility change must be verified and audited against the pinned upstream SHA.
