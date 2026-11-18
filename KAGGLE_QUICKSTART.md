# One-click Kaggle benchmark (paper baselines only)

**Artifacts**: `Ty_Linh_Paper_Benchmark_Full_Corrected.zip` (full repository), `Ty_Linh_Paper_Benchmark_Changes_Only.zip` (patch for the newest `Ty_Linh_loz-main(1).zip`), and `Kaggle_Run_PaperOnly_E2E.py` (notebook entry).

1. Upload the full corrected ZIP as a Kaggle Dataset input and enable **GPU** and **Internet**. Review/accept the VEDAI dataset terms.
2. Run `Kaggle_Run_PaperOnly_E2E.py` in a Kaggle notebook with `!python /kaggle/input/<notebook-dataset>/Kaggle_Run_PaperOnly_E2E.py`, or paste its code in a notebook cell.
3. The script searches exactly one corrected repository ZIP in `/kaggle/input/`, restores the source, downloads/preprocesses paired VEDAI-512 and creates a new 3-way deterministic train/val/test split. Both modalities are encoded to the SAME JPEG format for all methods.
4. It trains **MR-SJPA (proposal)** and **ETFNet (published original author code)** with three matched seeds (0/1/2) and attempts exact checkpoint resume where available. The paper baseline is **not** a local Concat/Add fusion substitute.
5. After training, it selects real `best.pt` weights using validation, runs one independent test evaluation per seed, saves actual OBB predictions, evaluates each with the SAME polygon-IoU AP101 implementation and emits a statistics package (mean, SD, 95% t CI, paired deltas/t-test, exact sign-flip p, LaTeX table).
6. Output: `/kaggle/working/PAPER_ONLY_THREE_SEED_RESULTS.zip`.

Optional: Set environment `ETFNET_AUTHOR_SHA` to a pinned original upstream commit; recommended for final published results. Set `C2FORMER_PYTHON` to a compatible separate MMRotate 0.3.x interpreter to automatically add the third *original-paper* C²Former baseline. The default Kaggle image is **not** known compatible with legacy C²Former; if no compatible environment is supplied, results are labelled **2-method only**, never 3-method.

**Reality check:** CPU unit/integration tests passed. Real dataset download, author-code GPU training and legacy C²Former environment have **not** been executed here. Public host availability, CUDA ABI and upstream loader compatibility may cause a clear error rather than silent substitution. This is a **new VEDAI held-out benchmark**, not exact reproduction of the papers' originally published numbers. See `PAPER_BENCHMARK_E2E.md`.
