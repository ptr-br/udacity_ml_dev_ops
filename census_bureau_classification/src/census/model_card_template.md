# Model Card
## Model Details
- `Name`: Census Income RandomForest v1
- `Type`: RandomForestClassifier (scikit-learn)
- `Inputs`: numeric features (age, education-num, hours-per-week, fnlgt, capital-gain, capital-loss) and one‑hot encoded categorical features (workclass, education, marital-status, occupation, relationship, race, sex, native-country)
- `Output`: binary income (>50K)

## Intended Use
- Exploratory and educational income classification on census-like tabular data.
- Not for high-stakes automated decisions (hiring, lending, etc.) without further validation and governance.

## Training Data
- Cleaned Census (Adult)-style dataset: normalized text, handled missing values, converted numerics, deduplicated, and one-hot encoded categoricals.
- Training split: typical 80/20 train/test (k-fold optional).


## Evaluation Data
- Held-out 20% test split from the cleaned dataset.
- Slice evaluation performed per categorical feature to surface subgroup behavior.
## Metrics
_Please include the metrics used and your model's performance on those metrics._

## Ethical Considerations
- Model can reflect and amplify biases correlated with sensitive attributes (race, sex, nationality).
- Potential harms include disparate errors across demographic groups and misuse in consequential decisions.

## Caveats and Recommendations
- Audit slice-level performance (race, sex, native-country) and apply mitigation if disparities appear.
- Retrain and re-evaluate on new data; consider calibration and hyperparameter tuning for production use.
- Remove/mask identifiers and require human oversight for critical decisions.