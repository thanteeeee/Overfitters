import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn
from sklearn.metrics import (confusion_matrix,accuracy_score, f1_score,classification_report)
import mlflow
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
pipepath = os.path.abspath("notebooks/04.pipeline_data")
sys.path.append(pipepath)
import Pipeline_Sirawat as pipe


pipelines_intance = pipe.pipeline("dataset/kgdataset.csv", "dataset/survey_data.xlsx")
df = pipelines_intance.run_pipeline()
x = df.drop("Screen Time Affects Sleep?", axis=1)
y = df["Screen Time Affects Sleep?"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50,stratify=y)
print(f"Train: {x_train.shape[0]} rows | Test: {x_test.shape[0]} rows")
print(f"Class distribution:\n{y.value_counts()}\n")
models ={
    "LogisticRegression": LogisticRegression(max_iter=1000,class_weight="balanced",random_state=50),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100,random_state=50,class_weight="balanced"),
    "XGBClassifier": XGBClassifier(random_state=50,use_label_encoder=False,eval_metric='mlogloss'),
}
CLASSES = ["No", "Yes", "Not Sure"]
def plot_confusion_matrix(y_test, y_pred, model_name: str):
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    path = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    fig.savefig(path)
    plt.close(fig)
    return path
 
 
def plot_feature_importance(model, feature_names: list, model_name: str):
    if not hasattr(model, "feature_importances_"):
        return None
    importance = model.feature_importances_
    feat_df = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .head(10)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=feat_df, x="importance", y="feature", palette="Blues_r", ax=ax)
    ax.set_title(f"Feature Importance — {model_name}")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    path = f"feature_importance_{model_name.replace(' ', '_')}.png"
    fig.savefig(path)
    plt.close(fig)
    return path
 
 
def plot_model_comparison(results: dict):
    # เปรียบเทียบ accuracy และ f1 ของทุก model
    names   = list(results.keys())
    acc     = [results[n]["accuracy"] for n in names]
    f1      = [results[n]["f1"] for n in names]
 
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - 0.2, acc, 0.4, label="Accuracy", color="#3b82f6")
    bars2 = ax.bar(x + 0.2, f1,  0.4, label="F1 (weighted)", color="#22c55e")
 
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Comparison — Accuracy vs F1 Score")
    ax.set_ylabel("Score")
    ax.legend()
 
    # แสดงตัวเลขบน bar
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=10)
 
    plt.tight_layout()
    path = "model_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    return path
mlflow.set_experiment("Sleep Quality Prediction")
results = {}
for name, model in models.items():

    with mlflow.start_run(run_name=name):
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        results[name] = {"accuracy": acc, "f1": f1}
        mlflow.log_params(model.get_params())
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(model, artifact_path=name)
        cm_path = plot_confusion_matrix(y_test, y_pred, name)
        mlflow.log_artifact(cm_path)
        fi_path = plot_feature_importance(model, x.columns.tolist(), name)
        if fi_path:
            mlflow.log_artifact(fi_path)
        print(classification_report(y_test, y_pred, target_names=CLASSES,zero_division=0))
        print(f"Accuracy: {acc:.4f} | F1 Score (weighted): {f1:.4f}\n")
comparison_path = plot_model_comparison(results)
 
with mlflow.start_run(run_name="Model Comparison"):
    mlflow.log_artifact(comparison_path)
    for name, metrics in results.items():
        mlflow.log_metric(f"{name}_accuracy", metrics["accuracy"])
        mlflow.log_metric(f"{name}_f1", metrics["f1"])
 
print(f"\n{'='*45}")
print("  Summary")
print(f"{'='*45}")
for name, metrics in results.items():
    print(f"{name:<25} accuracy={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}")
 
print("'mlflow ui' to view results at http://localhost:5000")