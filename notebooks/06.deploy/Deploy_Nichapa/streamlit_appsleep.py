import streamlit as st
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats.mstats import winsorize

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Sleep & Screen Time Predictor",
    page_icon="🌙",
    layout="centered"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-sub { color: #94a3b8; font-size: 1rem; margin-bottom: 2rem; }
.result-box {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-no  { background: linear-gradient(135deg, #064e3b, #065f46); border: 1px solid #34d399; }
.result-maybe { background: linear-gradient(135deg, #78350f, #92400e); border: 1px solid #fbbf24; }
.result-yes { background: linear-gradient(135deg, #4c0519, #7f1d1d); border: 1px solid #f87171; }
.result-label { font-size: 1.8rem; font-weight: 800; margin-bottom: 0.5rem; color: white !important; font-family: 'Syne', sans-serif; }
.result-desc  { font-size: 0.95rem; color: white !important; }
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 1rem;
    margin-top: 1.5rem;
}
.tip-box {
    background: #1e293b;
    border-left: 3px solid #a78bfa;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    font-size: 0.9rem;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# ==================== WINSORIZE FUNCTION ====================
def winsorize_data(X):
    if isinstance(X, pd.DataFrame):
        result = X.copy().astype(float)
        for col in result.columns:
            result[col] = winsorize(result[col].values, limits=(0.1, 0.1))
        return result
    else:
        result = X.copy().astype(float)
        for i in range(result.shape[1]):
            result[:, i] = winsorize(result[:, i], limits=(0.1, 0.1))
        return result

# ==================== RRC MODEL ARCHITECTURE ====================
class RRCModel(nn.Module):
    def __init__(self):
        super(RRCModel, self).__init__()
        self.base_fc     = nn.Linear(9, 64)
        self.residual_fc = nn.Linear(64 + 3, 64)  # ← แก้จาก 67 (64+3 ไม่ใช่ 64+9)
        self.output_fc   = nn.Linear(64, 3)
        self.relu        = nn.ReLU()

    def forward(self, x):
        base_out = self.relu(self.base_fc(x))
        # concat base_out กับ 3 features แรก ไม่ใช่ทั้งหมด
        combined = torch.cat([base_out, x[:, :3]], dim=1)  # 64 + 3 = 67
        res_out  = self.relu(self.residual_fc(combined))
        return self.output_fc(res_out)

# ==================== LOAD ALL MODELS ====================
@st.cache_resource
def load_models():
    try:
        pipeline  = joblib.load("sleep_data_pipeline.joblib")
        lgbm      = joblib.load("base_lgbm.joblib")
        svm       = joblib.load("base_svm.joblib")
        meta      = joblib.load("final_meta_learner.joblib")
        le        = joblib.load("target_label_encoder.joblib")

        # Load RRC (PyTorch)
        rrc = RRCModel()
        rrc.load_state_dict(torch.load("base_rrc_weights.pth", map_location="cpu"))
        rrc.eval()

        return pipeline, lgbm, svm, rrc, meta, le
    except Exception as e:
        st.error(f"โหลดโมเดลไม่ได้: {e}")
        return None, None, None, None, None, None

pipeline, lgbm, svm, rrc, meta, le = load_models()

# ==================== HEADER ====================
st.markdown('<div class="hero-title">🌙 Sleep Impact Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">วิเคราะห์ว่าการใช้มือถือส่งผลต่อการนอนของคุณอย่างไร</div>', unsafe_allow_html=True)

if meta is None:
    st.warning("⚠️ ไม่พบไฟล์โมเดล — กรุณาวางไฟล์ .joblib และ .pth ในโฟลเดอร์เดียวกับ appsleep.py")

st.divider()

# ==================== INPUT FORM ====================
st.markdown('<div class="section-label">📋 ข้อมูลส่วนตัว</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("อายุ (ปี)", min_value=10, max_value=80, value=22, step=1)
with col2:
    sleep_hours = st.slider("ชั่วโมงการนอนต่อคืน", 2, 12, 7, 1)

st.markdown('<div class="section-label">📱 พฤติกรรมการใช้มือถือ</div>', unsafe_allow_html=True)
daily_screen_time = st.slider("ชั่วโมงใช้มือถือต่อวัน", 0, 12, 4, 1)
use_before_sleep  = st.selectbox("ใช้มือถือก่อนนอน?", options=[0,1], format_func=lambda x: "No 🚫" if x==0 else "Yes 📱")
wellness_apps     = st.selectbox("ใช้แอปสุขภาพ/การนอน?", options=[0,1], format_func=lambda x: "No" if x==0 else "Yes ✅")

st.markdown('<div class="section-label">😴 สภาพร่างกายและจิตใจ</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    stress_level = st.slider("ระดับความเครียด", 1, 10, 5)
    feel_rested  = st.selectbox("รู้สึกพักผ่อนเพียงพอ?", options=[0,1,2], format_func=lambda x: ["No 😩","Sometimes 😐","Yes 😊"][x])
with col4:
    anxiety_mood  = st.selectbox("มีความวิตกกังวล?", options=[0,1], format_func=lambda x: "No 😌" if x==0 else "Yes 😟")
    sleep_quality = st.selectbox("คุณภาพการนอน", options=[0,1], format_func=lambda x: ["Bad 😫","Good 😴"][x])

st.divider()

# ==================== PREDICT ====================
if st.button("🔍 วิเคราะห์ผลกระทบ", use_container_width=True, type="primary"):
    if meta is None:
        st.error("❌ โหลดโมเดลไม่ได้")
    else:
        # 1. สร้าง input DataFrame
        input_df = pd.DataFrame([[age, sleep_hours, daily_screen_time, stress_level,
                                   use_before_sleep, anxiety_mood, wellness_apps,
                                   feel_rested, sleep_quality]],
                                columns=['Age','Sleep Hours','Daily Screen Time','Stress Level',
                                         'Use Before Sleep_enc','Anxiety/Low Mood_enc',
                                         'Wellness Apps_enc','Feel Rested_enc','Sleep Quality_enc'])

        # 2. Transform ด้วย Pipeline
        X_transformed = pipeline.transform(input_df)

        # 3. Base model predictions (probabilities)
        lgbm_proba = lgbm.predict_proba(X_transformed)   # (1, 3)
        svm_proba  = svm.predict_proba(X_transformed)    # (1, 3)

        # RRC prediction
        X_tensor   = torch.tensor(X_transformed.astype(np.float32))
        with torch.no_grad():
            rrc_logits = rrc(X_tensor)
            rrc_proba  = torch.softmax(rrc_logits, dim=1).numpy()  # (1, 3)

        # 4. Stack meta features
        X_meta = np.hstack([lgbm_proba, svm_proba, rrc_proba])  # (1, 9)

        # 5. Meta learner predict
        prediction = meta.predict(X_meta)[0]
        proba      = meta.predict_proba(X_meta)[0]

        # 6. แสดงผล
        labels = {
            0: ("ไม่มีผลกระทบ", "result-no",    "🟢", "การใช้มือถือของคุณดูเหมือนจะไม่ส่งผลกระทบต่อการนอน"),
            1: ("ไม่แน่ใจ",      "result-maybe", "🟡", "ยังไม่ชัดเจน อาจมีหรือไม่มีผลกระทบต่อการนอน"),
            2: ("มีผลกระทบ",    "result-yes",   "🔴", "การใช้มือถือน่าจะส่งผลกระทบต่อคุณภาพการนอนของคุณ")
        }
        label, css_class, icon, desc = labels[prediction]

        st.markdown(f"""
        <div class="result-box {css_class}">
            <div style="font-size:3rem">{icon}</div>
            <div class="result-label">{label}</div>
            <div class="result-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:1.5rem">📊 ความมั่นใจของโมเดล</div>', unsafe_allow_html=True)
        for name, prob in zip(["No Effect 🟢","Not Sure 🟡","Yes Effect 🔴"], proba):
            st.write(f"{name}: **{prob*100:.1f}%**")
            st.progress(float(prob))

        tips = {
            2: "💡 <b>เคล็ดลับ:</b> ลองวางมือถือก่อนนอนอย่างน้อย 30–60 นาที เปิด Night Mode และใช้แอปจับเวลาหน้าจอ",
            1: "🤔 <b>สังเกตตัวเอง:</b> ลองบันทึกคุณภาพการนอนสักสัปดาห์ แล้วดูความสัมพันธ์กับการใช้มือถือ",
            0: "✅ <b>ดีมาก!</b> คุณดูแลสุขภาพการนอนได้ดี รักษาพฤติกรรมนี้ต่อไปนะครับ"
        }
        st.markdown(f'<div class="tip-box">{tips[prediction]}</div>', unsafe_allow_html=True)

# ==================== FOOTER ====================
st.divider()
st.markdown('<p style="text-align:center; color:#475569; font-size:0.8rem">Meta-Stacking Model · Accuracy 74.26% · Class 1 F1: 0.51</p>', unsafe_allow_html=True)
