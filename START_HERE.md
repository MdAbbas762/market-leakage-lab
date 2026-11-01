# Start here — practical checklist

1. Open the project in VS Code and read `README.md`.
2. Run `python -m pip install -r requirements-dev.txt` and `python -m pytest -q`.
3. Run the synthetic example first (the result is **not** real market performance).
4. Obtain a **real** OHLCV CSV using the optional downloader or another legitimate source.
5. Run the experiment on real data and inspect `outputs/spy/REPORT.md` and `results.json`.
6. Review the model choices, write **your own** findings under a separate Results section of the README, including failures.
7. Make at least one meaningful improvement or independent audit, such as a rolling window comparison or data regime test.
8. Only after verifying your work, create your public GitHub repo and use it in the HFT application.

Do not copy the synthetic metrics into a resume. Do not label daily OHLCV as high-frequency/order-book data. If you change features/models after looking at the final test set, treat that test as consumed and use a genuinely later untouched holdout for the next evaluation.
