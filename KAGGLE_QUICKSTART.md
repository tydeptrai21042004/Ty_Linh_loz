# Kaggle — paper baseline + ablations + proposal

This release includes one official-paper-derived ETFNet–TGF baseline, seven ablations,
and the SJPA-TR proposal. The Kaggle notebook does **not** include arbitrary RGB-only
baselines or the obsolete CAFEM-SJPA variant.

1. Enable Kaggle **Internet** and **GPU** and ensure a usable dataset source.
2. Upload `ETFNet_SJPA_TR_Paper_Baseline_Final.zip` as an input Dataset, or clone your
   GitHub repository after pushing the flat source files.
3. After restoration, use `kaggle/ETFNet_Paper_Baseline_Ablations_Kaggle.py` as one cell.
4. Specify `SOURCE_MODE='uploaded_zip'` or `SOURCE_MODE='github'` in the script as desired.
5. Set `ETFNET_EPOCHS` and `ETFNET_SEEDS` explicitly if changing the defaults.

The runner trains one ETFNet paper-baseline candidate, seven ablations and one
proposal under the same dataset split and per-seed configuration, then writes
`comparison_group` and `source_paper` fields into CSV. Smoke tests do not establish
published-paper accuracy. Kaggle runtime, dataset download availability, GPU
memory and convergence have **not** been validated by this release.
