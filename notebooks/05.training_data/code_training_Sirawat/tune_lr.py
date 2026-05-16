import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)


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
    path = f"confusion_matrix_tuned.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_grid_search_results(cv_results: dict):
    """Plot accuracy ของแต่ละ C value จาก GridSearch"""
    results_df = pd.DataFrame(cv_results)
    fig, ax = plt.subplots(figsize=(8, 5))
    for solver in results_df["param_solver"].unique():
        subset = results_df[results_df["param_solver"] == solver]
        ax.plot(
            subset["param_C"].astype(float),
            subset["mean_test_score"],
            marker="o",
            label=solver
        )
    ax.set_xscale("log")
    ax.set_xlabel("C (Regularization)")
    ax.set_ylabel("F1 Score (weighted)")
    ax.set_title("GridSearchCV Results — Logistic Regression")
    ax.legend()
    plt.tight_layout()
    path = "gridsearch_results.png"
    fig.savefig(path)
    plt.close(fig)
    return path

param_grid = {
    "C"       : [0.01, 0.1, 1, 10, 100],
    "solver"  : ["lbfgs", "saga"],
    "max_iter": [1000],
}

base_model = LogisticRegression(
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

print("Running GridSearchCV...")
grid_search.fit(x_train, y_train)

print(f"\nBest Params : {grid_search.best_params_}")
print(f"Best CV F1  : {grid_search.best_score_:.4f}")


best_model = grid_search.best_estimator_
y_pred     = best_model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average="weighted")

cv_scores = cross_val_score(best_model, x_train, y_train, cv=5, scoring="f1_weighted")

print(f"\n{'='*45}")
print("  Tuned Logistic Regression Results")
print(f"{'='*45}")
print(classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0))
print(f"Test Accuracy : {acc:.4f}")
print(f"Test F1       : {f1:.4f}")
print(f"CV F1 Mean    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


mlflow.set_experiment("Sleep Quality Prediction")

with mlflow.start_run(run_name="LogisticRegression_Tuned"):

    mlflow.log_params(grid_search.best_params_)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_weighted", f1)
    mlflow.log_metric("cv_f1_mean", cv_scores.mean())
    mlflow.log_metric("cv_f1_std", cv_scores.std())
    mlflow.log_metric("best_cv_score", grid_search.best_score_)

    mlflow.sklearn.log_model(best_model, artifact_path="LogisticRegression_Tuned")

    cm_path = plot_confusion_matrix(y_test, y_pred, "Confusion Matrix — Tuned LR")
    mlflow.log_artifact(cm_path)

    gs_path = plot_grid_search_results(grid_search.cv_results_)
    mlflow.log_artifact(gs_path)

    print(f"\nMLflow run logged!")
    print(f"   Run 'mlflow ui' → http://localhost:5000")


print(f"\n{'='*45}")
print("  Baseline vs Tuned")
print(f"{'='*45}")
print(f"{'Model':<35} {'Accuracy':>9} {'F1':>9}")
print(f"{'-'*55}")
print(f"{'Logistic Regression (baseline)':<35} {'0.7300':>9} {'0.7400':>9}")
print(f"{'Logistic Regression (tuned)':<35} {acc:>9.4f} {f1:>9.4f}")