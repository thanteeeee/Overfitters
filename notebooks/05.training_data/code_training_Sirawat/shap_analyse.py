import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import mlflow
import mlflow.sklearn
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=50,
)
model.fit(x_train, y_train)

explainer   = shap.LinearExplainer(model, x_train)
shap_values = explainer(x_test)
print("SHAP values computed")
print(f"shap_values shape: {shap_values.shape}")

CLASS_NAMES = ["No", "Yes", "Not Sure"]

for i, class_name in enumerate(CLASS_NAMES):
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values.values[:, :, i],
        x_test,
        plot_type="dot",
        show=False
    )
    plt.title(f"SHAP Summary Plot — Class: {class_name}")
    plt.tight_layout()
    path = f"shap_summary_{class_name}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Summary plot saved — {class_name}")


# --- 3.2 Bar Plot — global importance รวมทุก class ---
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(
    shap_values.values[:, :, 1],  # ใช้ class "Yes" เป็นหลัก
    x_test,
    plot_type="bar",
    show=False
)
plt.title("SHAP Feature Importance — Class: Yes")
plt.tight_layout()
bar_path = "shap_bar.png"
plt.savefig(bar_path, bbox_inches="tight")
plt.close()
print("Bar plot saved")

cases = {
    "case_first" : 0,
    "case_middle": len(x_test) // 2,
    "case_last"  : len(x_test) - 1,
}
 
waterfall_paths = []
for case_name, idx in cases.items():
    shap.plots.waterfall(
        shap.Explanation(
            values      = shap_values.values[idx, :, 1],
            base_values = shap_values.base_values[idx, 1],
            data        = x_test.iloc[idx],
            feature_names = x_test.columns.tolist()
        ),
        show=False
    )
    plt.title(f"SHAP Waterfall — {case_name} (Class: Yes)")
    plt.tight_layout()
    path = f"shap_waterfall_{case_name}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    waterfall_paths.append(path)
 
print("Waterfall plots saved")
 
 

for i, class_name in enumerate(CLASS_NAMES):
    shap.plots.heatmap(
        shap.Explanation(
            values        = shap_values.values[:, :, i],
            base_values   = shap_values.base_values[:, i],
            data          = x_test,
            feature_names = x_test.columns.tolist()
        ),
        show=False
    )
    plt.title(f"SHAP Heatmap — Class: {class_name}")
    plt.tight_layout()
    path = f"shap_heatmap_{class_name}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved — {class_name}")
 

mlflow.set_experiment("Sleep Quality Prediction")
 
with mlflow.start_run(run_name="SHAP_Analysis"):
 
    mlflow.sklearn.log_model(model, artifact_path="LogisticRegression_Final")
 

    mlflow.log_artifact(bar_path)
    for path in waterfall_paths:
        mlflow.log_artifact(path)
 

    mean_shap = pd.DataFrame(
        np.abs(shap_values.values).mean(axis=0),
        index=x.columns,
        columns=["mean_abs_shap"]
    ).sort_values("mean_abs_shap", ascending=False)
 
    for feat, row in mean_shap.iterrows():
        mlflow.log_metric(f"shap_{feat}", row["mean_abs_shap"])
 
    print("\nMLflow run logged!")
    print("   Run 'mlflow ui' → http://localhost:5000")
 
 
print(f"\n{'='*45}")
print("  SHAP Feature Importance (Mean |SHAP|)")
print(f"{'='*45}")
print(mean_shap.to_string())