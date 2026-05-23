import streamlit as st
import requests

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Sleep & Screen Time Analysis",
    page_icon="📱",
    layout="centered",
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=IBM+Plex+Sans+Thai:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans Thai', sans-serif;
    background-color: #080b14;
    color: #e2e8f0;
}
.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    line-height: 1.2;
}
.sub-title {
    text-align: center;
    color: #64748b;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}
.disclaimer-box {
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.3);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1.5rem;
    font-size: 0.85rem;
    color: #fbbf24;
    line-height: 1.6;
}
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #60a5fa;
    margin: 1.2rem 0 0.4rem 0;
}
.result-box {
    border-radius: 16px;
    padding: 1.8rem;
    margin-top: 1.5rem;
    text-align: center;
}
.result-yes {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.35);
}
.result-no {
    background: rgba(52,211,153,0.1);
    border: 1px solid rgba(52,211,153,0.35);
}
.result-notsure {
    background: rgba(251,191,36,0.1);
    border: 1px solid rgba(251,191,36,0.35);
}
.result-emoji { font-size: 2.5rem; }
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    margin: 0.4rem 0;
}
.result-message {
    font-size: 0.95rem;
    color: #cbd5e1;
    line-height: 1.6;
}
.result-note {
    font-size: 0.78rem;
    color: #475569;
    margin-top: 0.8rem;
    font-style: italic;
}
.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07);
    margin: 1.2rem 0;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Header
# ============================================================
st.markdown('<div class="main-title">📱 Screen Time<br>& Sleep Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">วิเคราะห์ว่าการใช้มือถือส่งผลต่อการนอนหลับของคุณไหม</div>', unsafe_allow_html=True)

# ============================================================
# Disclaimer
# ============================================================
st.markdown("""
<div class="disclaimer-box">
⚠️ <strong>Disclaimer</strong><br>
ผลลัพธ์จาก tool นี้เป็นการประเมินเบื้องต้นจาก Machine Learning Model เท่านั้น
<strong>ไม่ใช่การวินิจฉัยทางการแพทย์</strong>
ความแม่นยำของ model อยู่ที่ประมาณ 73%
หากมีปัญหาการนอนหลับควรปรึกษาแพทย์หรือผู้เชี่ยวชาญด้านสุขภาพ
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ============================================================
# Input Form
# ============================================================
st.markdown('<div class="section-label">🧑 ข้อมูลส่วนตัว</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.slider("1. คุณอายุเท่าไหร่", min_value=10, max_value=70, value=25)
with col2:
    sleep_hours = st.slider("2. ระยะเวลาการนอนโดยเฉลี่ยต่อวัน (ชั่วโมง)", min_value=1, max_value=12, value=7)

st.markdown('<div class="section-label">📱 พฤติกรรม Screen Time</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    daily_screen_time = st.slider("4. ระยะเวลาการใช้หน้าจอต่อวัน (ชั่วโมง)", min_value=0, max_value=12, value=4)
with col4:
    stress_level = st.slider("6. ระดับความเครียดที่คุณรู้สึก (1 น้อยสุด — 10 มากสุด)", min_value=1, max_value=10, value=5)

use_before_sleep = st.radio(
    "5. คุณได้ใช้มือถือก่อนนอนหรือไม่",
    options=["ใช่", "ไม่ใช่"],
    horizontal=True,
)

st.markdown('<div class="section-label">😴 คุณภาพการนอน</div>', unsafe_allow_html=True)
col5, col6 = st.columns(2)
with col5:
    sleep_quality = st.radio(
        "9. คุณภาพการนอน",
        options=["ดี", "ไม่ดี"],
        horizontal=True,
    )
with col6:
    feel_rested = st.radio(
        "3. ความสดชื่นหลังตื่นนอน",
        options=["สดชื่น", "บางครั้ง", "ไม่สดชื่น"],
        horizontal=True,
    )

col7, col8 = st.columns(2)
with col7:
    anxiety = st.radio(
        "7. คุณรู้สึกหงุดหงิด/วิตกกังวล เมื่อไม่ได้ใช้มือถือเป็นเวลานานไหม",
        options=["ใช่", "ไม่ใช่"],
        horizontal=True,
    )
with col8:
    wellness = st.radio(
        "8. คุณได้ใช้งานแอปพลิเคชันเพื่อสุขภาพหรือไม่",
        options=["ใช่", "ไม่ใช่"],
        horizontal=True,
    )

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ============================================================
# Predict
# ============================================================
if st.button("🔍 วิเคราะห์ผลกระทบ", use_container_width=True, type="primary"):

    payload = {
        "age"               : age,
        "sleep_hours"       : sleep_hours,
        "daily_screen_time" : daily_screen_time,
        "stress_level"      : stress_level,
        "use_before_sleep"  : 1 if use_before_sleep == "ใช่" else 0,
        "feel_rested"       : {"สดชื่น": 2, "บางครั้ง": 1, "ไม่สดชื่น": 0}[feel_rested],
        "anxiety_low_mood"  : 1 if anxiety == "ใช่" else 0,
        "wellness_apps"     : 1 if wellness == "ใช่" else 0,
        "sleep_quality"     : 1 if sleep_quality == "ดี" else 0,
    }

    try:
        response = requests.post("http://localhost:8000/predict", json=payload, timeout=5)
        result   = response.json()

        pred    = result["prediction"]
        message = result["message"]
        proba   = result["probability"]

        # กำหนด style ตาม prediction
        if pred == "Yes":
            box_class  = "result-yes"
            emoji      = "⚠️"
            label_text = "มีแนวโน้มส่งผล"
            color      = "#ef4444"
        elif pred == "No":
            box_class  = "result-no"
            emoji      = "✅"
            label_text = "ไม่ส่งผลมากนัก"
            color      = "#34d399"
        else:
            box_class  = "result-notsure"
            emoji      = "🤔"
            label_text = "ยังไม่แน่ชัด"
            color      = "#fbbf24"

        # แสดงผล
        st.markdown(f"""
        <div class="result-box {box_class}">
            <div class="result-emoji">{emoji}</div>
            <div class="result-label" style="color:{color}">{label_text}</div>
            <div class="result-message">{message}</div>
            <div class="result-note">
                ผลนี้มาจากการเปรียบเทียบข้อมูลของคุณกับกลุ่มตัวอย่าง ~1,200 คน<br>
                ไม่ใช่ผลทางการแพทย์ — ความแม่นยำของ model อยู่ที่ประมาณ 73%
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability bars
        st.markdown('<div class="section-label" style="margin-top:1.5rem">📊 ความน่าจะเป็น</div>', unsafe_allow_html=True)
        label_th = {"Yes": "ส่งผล", "No": "ไม่ส่งผล", "Not Sure": "ไม่แน่ชัด"}
        for key, val in sorted(proba.items(), key=lambda x: -x[1]):
            st.progress(int(val), text=f"{label_th.get(key, key)}: {val}%")

    except requests.exceptions.ConnectionError:
        st.error("ไม่สามารถเชื่อมต่อ API ได้ กรุณารัน FastAPI ก่อน\n\n```bash\nuvicorn api.main:app --reload\n```")
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")