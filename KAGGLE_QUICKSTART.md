# Kaggle quick start — three DOI datasets and 111 planned trainings

See `RESEARCH_111_RUN_PLAN.md` for the complete experiment protocol.

1. Upload `Ty_Linh_111_Experiments_Full_Corrected.zip` as a Kaggle Dataset input; enable **GPU** and **Internet**.
2. Place `Kaggle_Run_111_Experiments.py` in a notebook cell, or run its downloaded path.
3. Review the VEDAI, DroneVehicle and M3FD license and distribution terms; the entry script passes `--accept-license` only as an explicit user responsibility.
4. To train original C²Former alongside ETFNet, set `C2FORMER_PYTHON` to a compatible installed MMRotate 0.3.x interpreter. If not available, set `EXCLUDE_C2FORMER=1`: the pipeline becomes 102 training jobs, and C²Former is **not** reported as a completed baseline.
5. Save the Kaggle output `research/` and prepared `datasets/` across independent sessions. `MAX_NEW_TRAINING_JOBS=8` and `MAX_NEW_EVALUATIONS=63` bound work per Kaggle launch.
6. Final paper tables and figures are emitted only after all required real checkpoints and common-evaluator test receipts exist.

For a no-GPU preliminary count:

```bash
python paper_research_pipeline.py --stage plan
```

For a dataset download/format test:

```bash
python layout_manager.py restore
pip install -r requirements-data.txt
python paper_three_datasets.py --datasets vedai dronevehicle m3fd --accept-license
```

The uploaded package has local tests, **not** a verified complete remote download or GPU run. Source-provider failures, disk quotas and CUDA/MMRotate incompatibilities must be resolved before publication claims. There is no dummy data fallback.
