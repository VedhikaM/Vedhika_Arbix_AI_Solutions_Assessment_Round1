# Audit Memo

The original script had multiple issues which made the validation score unreliable. 
The biggest issue was data leakage because the feature 
"defaulted_in_next_12_months" uses future information that would not be available during actual scoring.

Another issue was mixing business policy directly into the target score using PM Kisan status. 
This is risky because policy rules should stay outside the ML model.

The script also used LabelEncoder for categorical columns and did not save preprocessing artifacts properly. 
Only the model file was saved, which makes reproducibility difficult.

## Changes Made

- Removed leakage column
- Removed policy modification from target
- Used OneHotEncoder with sklearn pipeline
- Added deterministic train/test split
- Saved pipeline and metadata
- Added simple reason codes

## Limitations

This is still a baseline solution. 
I would improve fairness checks, drift monitoring, and explainability with more time.

## Monitoring

After deployment I would monitor:
- crop type
- PM Kisan status
- land area groups
- missing value drift