# Retail sales and customer analysis

This repository contains two historical R analyses and their CSV inputs:

- `whole_foods.csv`: quarterly Whole Foods sales from 1995 through 2016.
- `purchases_type.csv`: 1,000 customer examples with frequency, sex indicator,
  spending, and customer type.

The original R scripts use `whole_foods` and `purchases_type` objects; load each
CSV into the matching R object before running the relevant script. Their
dependencies include `tidyverse`, `mosaic`, `ggthemes`, `ggfortify`, and
`caret`. R and these packages are not installed or validated in this
environment.

## Reproducible Python baselines

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproducible_baselines.py
python -m unittest discover -s tests -v
```

The sales baseline fits a log-quadratic trend with quarterly effects, evaluates
the final eight quarters against a seasonal-naive baseline, and fits the model
to all history to forecast four quarters beyond 2016. Output includes MAE, RMSE,
and MAPE. The customer baseline compares standardized K-nearest neighbors and
logistic regression on a reproducible stratified 80/20 split.

These are reproducible reference baselines, not reproductions of every R model.
The short historical sales series and small customer sample limit reliability.
The forecast is an extrapolation, not current guidance. Customer associations
are not causal and should not be used for consequential decisions without
independent validation and fairness review.
