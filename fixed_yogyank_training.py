import pandas as pd
import json
import joblib
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error

from xgboost import XGBRegressor


MODEL_VERSION = "yogyank_safe_v1"


def load_data(path="farmer_scoring_sample_yogyank_round1_final.csv"):
    return pd.read_csv(path)


def main():

    df = load_data()

    target = "target_entitlement_score"

    # remove future leakage column
    leakage_cols = [
        "defaulted_in_next_12_months"
    ]

    feature_cols = [
        col for col in df.columns
        if col not in [target] + leakage_cols
    ]

    X = df[feature_cols]
    y = df[target]

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    # deterministic split for reproducibility
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        n_jobs=1,
        tree_method="hist"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")

    # reason codes
    feature_importance = pipeline.named_steps[
        "model"
    ].feature_importances_

    feature_names = pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": feature_importance
    })

    top_features = importance_df.sort_values(
        by="importance",
        ascending=False
    ).head(3)

    print("\nTop Reason Codes:")
    print(top_features)

    # save pipeline
    joblib.dump(
        pipeline,
        "artifacts/yogyank_pipeline.pkl"
    )

    metadata = {
        "model_version": MODEL_VERSION,
        "created_at": datetime.utcnow().isoformat(),
        "features": feature_cols,
        "removed_leakage_columns": leakage_cols
    }

    with open(
        "artifacts/model_metadata.json",
        "w"
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\nArtifacts saved successfully")


if __name__ == "__main__":
    main()