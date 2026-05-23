from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Sleep Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลด model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../../model/logistic_regression_Sirawat.pkl")
model = joblib.load(MODEL_PATH)

LABEL_MAP = {0: "No", 1: "Yes", 2: "Not Sure"}
MESSAGE_MAP = {
    "Yes"     : "การใช้ Screen Time ของคุณมีแนวโน้มส่งผลต่อการนอนหลับ",
    "No"      : "การใช้ Screen Time ของคุณดูเหมือนจะไม่ส่งผลต่อการนอนหลับมากนัก",
    "Not Sure": "ยังไม่สามารถสรุปได้ชัดเจน ลองสังเกตพฤติกรรมการนอนของตัวเองเพิ่มเติม",
}

# Input schema
class SleepInput(BaseModel):
    age: float
    sleep_hours: float
    daily_screen_time: float
    stress_level: float
    use_before_sleep: int
    feel_rested: int
    anxiety_low_mood: int
    wellness_apps: int
    sleep_quality: int

# Output schema
class SleepOutput(BaseModel):
    prediction: str
    probability: dict
    message: str


@app.get("/")
def root():
    return {"message": "Sleep Analysis API is running!"}


@app.post("/predict", response_model=SleepOutput)
def predict(data: SleepInput):
    features = np.array([[
        data.age,
        data.sleep_hours,
        data.daily_screen_time,
        data.stress_level,
        data.use_before_sleep,
        data.feel_rested,
        data.anxiety_low_mood,
        data.wellness_apps,
        data.sleep_quality,
    ]])

    pred    = model.predict(features)[0]
    proba   = model.predict_proba(features)[0]
    label   = LABEL_MAP[pred]
    classes = model.classes_

    probability = {
        LABEL_MAP[c]: round(float(p) * 100, 1)
        for c, p in zip(classes, proba)
    }

    return SleepOutput(
        prediction  = label,
        probability = probability,
        message     = MESSAGE_MAP[label],
    )
print(type(predict))