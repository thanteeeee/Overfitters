# 📱 Social Media & Sleep Analysis

วิเคราะห์ผลกระทบของการใช้ Social Media ต่อคุณภาพการนอนหลับ โดยใช้ Machine Learning

---

## 📌 Overview

โปรเจกต์นี้ศึกษาความสัมพันธ์ระหว่างพฤติกรรมการใช้ Social Media กับการนอนหลับ
โดยเก็บข้อมูลจาก 2 แหล่ง ได้แก่ Kaggle Dataset และ Survey ที่เก็บเองตาม Schema เดียวกัน
จากนั้นสร้าง ML Pipeline เพื่อทำนายว่า Screen Time ส่งผลต่อการนอนหลับหรือไม่

---

## 🔍 Key Findings

- **Screen Time** มี correlation กับ Sleep Hours ที่ **-0.67**
- **Stress Level** มี correlation กับ Sleep Hours ที่ **-0.62**
- Screen Time ส่งผลต่อการนอนทั้งทางตรง และทางอ้อมผ่าน Stress Level
- กลุ่มที่มี **Sleep Quality ดี** นอนได้เฉลี่ยมากกว่ากลุ่ม Sleep Quality แย่ถึง **3 ชั่วโมง**
- **Logistic Regression** ให้ผลดีที่สุด สอดคล้องกับ linear relationship ที่พบใน EDA
- **SHAP Analysis** พบว่า Sleep Hours และ Daily Screen Time เป็น 2 features ที่ model พึ่งพามากที่สุด

---

## 📁 Project Structure

```
project/
│
├── dataset/                        ← ไม่ได้ push ขึ้น GitHub (.gitignore)
│   ├── kgdataset.csv               ← Kaggle dataset
│   └── survey_data.xlsx            ← Survey ที่เก็บเอง
│
├── notebooks/
│   ├── 01_EDA.ipynb                ← Exploratory Data Analysis
│   ├── 02_cleaning.ipynb           ← Basic + Deep Cleaning
│   ├── 03_features.ipynb           ← Feature Engineering
│   ├── 04_pipeline_data/
│   │   └── Pipeline_Sirawat.py     ← Custom Pipeline Class
│   └── 05_training_data/
│       ├── train_model.py          ← Model Training + MLflow + CV
│       ├── tuning_lr.py            ← Logistic Regression Tuning
│       ├── tuning_rf.py            ← Random Forest Tuning
│       └── shap_analysis.py        ← SHAP Explainability
│
├── requirements.txt
└── .gitignore
```

---

## 🗂️ Dataset

| Source    | Rows      | Description                         |
|-----------|-----------|-------------------------------------|
| Kaggle    | ~1000     | Social Media & Sleep dataset        |
| Survey    | ~200      | เก็บเองตาม Kaggle schema (ภาษาไทย)  |
| **Total** | **~1200** | หลัง merge และ clean                |

### Features

| Column                     | Type     | Description                     |
|----------------------------|----------|---------------------------------|
| Age                        | Numeric  | อายุ                            |
| Sleep Hours                | Numeric  | จำนวนชั่วโมงนอน                |
| Daily Screen Time          | Numeric  | ชั่วโมงใช้ screen ต่อวัน        |
| Stress Level               | Numeric  | ระดับความเครียด (1-10)          |
| Use Before Sleep           | Binary   | ใช้มือถือก่อนนอนไหม            |
| Feel Rested                | Ordinal  | รู้สึกพักผ่อนเพียงพอไหม         |
| Anxiety/Low Mood           | Binary   | มีอาการวิตกกังวลไหม            |
| Wellness Apps              | Binary   | ใช้ Wellness App ไหม            |
| Sleep Quality              | Binary   | คุณภาพการนอน (Good/Bad)        |
| Screen Time Affects Sleep? | Target   | Screen Time ส่งผลต่อการนอนไหม  |

---

## ⚙️ Pipeline

```
Raw Data (Kaggle + Survey)
        ↓
DataLoader      → merge 2 แหล่งข้อมูล
        ↓
DataCleaner     → inconsistent_data → drop_duplicates
                → handle_missing → apply_cleaning → handle_outliers
        ↓
DataTransformer → encode categorical + ordinal + target
                → StandardScaler (exclude target)
        ↓
final_df ✓
```

**Libraries ที่ใช้ใน Pipeline**

| Library | ใช้ทำอะไร                   |
|---------|-----------------------------|
| pandas  | DataFrame operations        |
| numpy   | Numerical operations        |
| scipy   | Winsorize outliers          |
| sklearn | StandardScaler              |
| re      | Extract numbers จาก string  |

---

## 🤖 Models & Results

ทดลอง 3 models + Dummy Baseline พร้อม 5-Fold Cross Validation และ Hyperparameter Tuning
track ทุก experiment ด้วย **MLflow**

| Model                          | Accuracy | F1 (weighted) |
|--------------------------------|----------|---------------|
| Dummy (baseline)               | 0.5190   | 0.3546        |
| Logistic Regression (baseline) | 0.7300   | 0.7400 ✅     |
| Logistic Regression (tuned)    | 0.7257   | 0.7369        |
| Random Forest (baseline)       | 0.7215   | 0.7071        |
| Random Forest (tuned)          | 0.7131   | 0.7374        |
| XGBoost (baseline)             | 0.7131   | 0.7058        |

**Final Model: Logistic Regression**
- Accuracy: **0.7300**
- F1 Score: **0.7400**
- ดีกว่า Dummy baseline: Accuracy +0.22, F1 +0.39
- เหตุผล: dataset มี linear relationship ทำให้ Logistic Regression เหมาะสมที่สุด

---

## 🔎 SHAP Analysis

ใช้ **SHAP (SHapley Additive exPlanations)** อธิบาย model prediction ระดับ feature

**Feature Importance (Mean |SHAP|) — Class: Yes**

| Feature           | ความสำคัญ  |
|-------------------|-----------|
| Sleep Hours       | สูงสุด ⭐  |
| Daily Screen Time | สูง        |
| Stress Level      | ปานกลาง   |
| Sleep Quality     | ปานกลาง   |
| Age, Wellness Apps| ต่ำมาก    |

**Key Insight จาก SHAP**
- Sleep Hours และ Daily Screen Time เป็น 2 features ที่ model พึ่งพามากที่สุด
- Stress Level ทำหน้าที่เป็น mediator ระหว่าง Screen Time กับการนอน
- สอดคล้องกับ correlation heatmap ที่พบใน EDA

---

## 📊 Experiment Tracking

ใช้ **MLflow** track ทุก experiment

```bash
mlflow ui
# เปิด http://localhost:5000
```

สิ่งที่ track ไว้
- Parameters ของแต่ละ model
- Accuracy, F1 Score และ CV F1 (5-Fold)
- Confusion Matrix
- Feature Importance (RF, XGBoost)
- SHAP plots (Summary, Bar, Waterfall, Heatmap)
- Model artifacts

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. รัน Pipeline

```python
from notebooks.pipeline_data.Pipeline_Sirawat import pipeline

pipe     = pipeline("dataset/kgdataset.csv", "dataset/survey_data.xlsx")
final_df = pipe.run_pipeline()
```

### 3. Train Models

```bash
python notebooks/05_training_data/train_model.py
```

### 4. Hyperparameter Tuning

```bash
python notebooks/05_training_data/tuning_lr.py
python notebooks/05_training_data/tuning_rf.py
```

### 5. SHAP Analysis

```bash
python notebooks/05_training_data/shap_analysis.py
```

### 6. ดูผลใน MLflow

```bash
mlflow ui
```

---

## 📈 EDA Highlights

**Correlation Heatmap**

| Feature Pair                      | Correlation |
|-----------------------------------|-------------|
| Sleep Hours vs Daily Screen Time  | **-0.67**   |
| Sleep Hours vs Stress Level       | **-0.62**   |
| Daily Screen Time vs Stress Level | **+0.49**   |
| Age vs Sleep Hours                | ~0.02       |

**Key Insight:** Screen Time ส่งผลต่อการนอนทั้งทางตรง และทางอ้อมผ่าน Stress Level

---

## ⚠️ Limitations

- Sample size ของ Survey ที่เก็บเองค่อนข้างน้อย (~200 rows) อาจมี selection bias
- ข้อมูลเป็น self-reported ทำให้มี subjective bias ได้
- ไม่มีข้อมูล platform ที่ใช้ ทำให้วิเคราะห์แยกตาม platform ไม่ได้

---
