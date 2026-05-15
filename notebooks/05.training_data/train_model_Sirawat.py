import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

# ============================================================
# 0. โหลด data จาก pipeline
# ============================================================
pipepath = os.path.abspath("notebooks/04.pipeline_data")
sys.path.append(pipepath)
import Pipeline_Sirawat as pipe

pipeline_instance = pipe.pipeline("dataset/kgdataset.csv", "dataset/survey_data.xlsx")
df = pipeline_instance.run_pipeline()

x = df.drop("Screen Time Affects Sleep?", axis=1)
y = df["Screen Time Affects Sleep?"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=50,
    stratify=y
)

CLASSES = ["No", "Yes", "Not Sure"]


# ============================================================
# 1. Helper Functions
# ============================================================
def plot_confusion_matrix(y_test, y_pred, title: str):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    path = "confusion_matrix_rf_tuned.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_feature_importance(model, feature_names: list):
    importance = model.feature_importances_
    feat_df = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .head(10)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=feat_df, x="importance", y="feature", palette="Blues_r", ax=ax)
    ax.set_title("Feature Importance — Tuned Random Forest")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    path = "feature_importance_rf_tuned.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================
# 2. GridSearchCV
# ============================================================
param_grid = {
    "n_estimators"     : [100, 200, 300],
    "max_depth"        : [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf" : [1, 2, 4],
}

base_model = RandomForestClassifier(
    class_weight="balanced",
    random_state=50,
)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=1,
)

print("🔍 Running GridSearchCV for Random Forest...")
print(f"   Total combinations: {3 * 4 * 3 * 3} × 5 folds = {3*4*3*3*5} fits")
grid_search.fit(x_train, y_train)

print(f"\n✅ Best Params : {grid_search.best_params_}")
print(f"✅ Best CV F1  : {grid_search.best_score_:.4f}")


# ============================================================
# 3. Evaluate Best Model
# ============================================================
best_model = grid_search.best_estimator_
y_pred     = best_model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average="weighted")

cv_scores = cross_val_score(
    best_model, x_train, y_train,
    cv=5, scoring="f1_weighted"
)

print(f"\n{'='*45}")
print("  Tuned Random Forest Results")
print(f"{'='*45}")
print(classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0))
print(f"Test Accuracy : {acc:.4f}")
print(f"Test F1       : {f1:.4f}")
print(f"CV F1 Mean    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ============================================================
# 4. MLflow Tracking
# ============================================================
mlflow.set_experiment("Sleep Quality Prediction")

with mlflow.start_run(run_name="RandomForest_Tuned"):

    # Log best params
    mlflow.log_params(grid_search.best_params_)

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_weighted", f1)
    mlflow.log_metric("cv_f1_mean", cv_scores.mean())
    mlflow.log_metric("cv_f1_std", cv_scores.std())
    mlflow.log_metric("best_cv_score", grid_search.best_score_)

    # Log model
    mlflow.sklearn.log_model(best_model, artifact_path="RandomForest_Tuned")

    # Log confusion matrix
    cm_path = plot_confusion_matrix(y_test, y_pred, "Confusion Matrix — Tuned RF")
    mlflow.log_artifact(cm_path)

    # Log feature importance
    fi_path = plot_feature_importance(best_model, x.columns.tolist())
    mlflow.log_artifact(fi_path)

    print(f"\n✅ MLflow run logged!")
    print(f"   Run 'mlflow ui' → http://localhost:5000")


# ============================================================
# 5. เปรียบเทียบทั้งหมด
# ============================================================
print(f"\n{'='*55}")
print("  Full Comparison")
print(f"{'='*55}")
print(f"{'Model':<40} {'Accuracy':>9} {'F1':>9}")
print(f"{'-'*60}")
print(f"{'Logistic Regression (baseline)':<40} {'0.7300':>9} {'0.7400':>9}")
print(f"{'Logistic Regression (tuned)':<40} {'0.7257':>9} {'0.7369':>9}")
print(f"{'Random Forest (baseline)':<40} {'0.7200':>9} {'0.7100':>9}")
print(f"{'Random Forest (tuned)':<40} {acc:>9.4f} {f1:>9.4f}")