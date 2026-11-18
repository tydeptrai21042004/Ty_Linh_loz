# Applying this correction to Ty_Linh_loz-main(1).zip

The **changes-only** archive contains an installer and replacement/deletion file manifest. It is not itself a stand-alone model repository.

1. Extract the *user's source* `Ty_Linh_loz-main(1).zip` to a folder containing `layout_manager.py`.
2. Extract `Ty_Linh_Paper_Benchmark_Changes_Only.zip` into a separate folder.
3. Run `python apply_changes.py --target /path/to/Ty_Linh_loz-main` from the extracted patch folder.
4. Verify with `python layout_manager.py check && python layout_manager.py restore --force` inside the updated repository.
5. Run `python -m pytest -q` (requires the project dependencies).

Alternatively, use the full corrected ZIP directly; it already has all patches applied.

The new primary script is `kaggle/PaperOnly_ThreeSeed_E2E_Kaggle.py`. See `KAGGLE_QUICKSTART.md` and `PAPER_BENCHMARK_E2E.md`.
