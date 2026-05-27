# Yogyank Round 1 Assessment

## Time Taken
Approx 90 mins

## What I Fixed

- Removed future leakage column
- Removed policy logic from target score
- Added sklearn preprocessing pipeline
- Used OneHotEncoder for categorical features
- Added deterministic validation split
- Saved model artifacts and metadata
- Added simple reason code generation

## Validation Approach

Used deterministic non-shuffled split to better simulate future scoring.

## Assumptions

Dataset rows are assumed to be in chronological order.

## Future Improvements

- Fairness metrics
- Drift monitoring
- SHAP explainability
- Better temporal validation

## Run Steps

Install dependencies:

```bash
pip install -r requirements.txt
python fixed_yogyank_training.py
