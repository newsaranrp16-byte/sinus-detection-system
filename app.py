import streamlit as st
import pandas as pd
import os
from streamlit_mic_recorder import mic_recorder

st.title("🧠 Sinus Detection System (Tamil + Voice)")

# =========================
# FILE CHECK
# =========================
st.write("📂 Files:", os.listdir())

file_path = "dataset.csv"

# =========================
# LOAD DATASET (TAB FIX)
# =========================
if not os.path.exists(file_path):
    st.error("dataset.csv NOT FOUND")
else:
    df = pd.read_csv(file_path, sep='\t')

    df.columns = df.columns.str.strip()

    # =========================
    # PREPROCESS
    # =========================
    df['Symptoms'] = df['Symptoms'].astype(str).str.lower()
    df['Symptoms_List'] = df['Symptoms'].apply(lambda x: x.split(', '))

    # =========================
    # DISEASE INFO
    # =========================
    disease_info = {
        "Sinusitis": {
            "advice": ["Avoid cold items", "Steam inhalation", "Drink warm fluids"],
            "tamil": "இது சைனஸ் பிரச்சனை. மூக்கடைப்பு மற்றும் தலைவலி இருக்கும்."
        },
        "Common Cold": {
            "advice": ["Take rest", "Drink warm water", "Use steam"],
            "tamil": "இது சாதாரண சளி. ஓய்வு எடுத்துக் கொள்ளவும்."
        },
        "Influenza": {
            "advice": ["Take rest", "Drink fluids", "Consult doctor"],
            "tamil": "இது காய்ச்சல். மருத்துவரை அணுகவும்."
        },
        "Asthma": {
            "advice": ["Avoid dust", "Use inhaler", "Consult doctor"],
            "tamil": "இது ஆஸ்துமா. மூச்சு சிரமம் ஏற்படும்."
        },
        "Diabetes": {
            "advice": ["Control sugar", "Exercise", "Check regularly"],
            "tamil": "இது நீரிழிவு நோய்."
        }
    }

    # =========================
    # INPUT SECTION
    # =========================
    st.subheader("📝 Enter Symptoms")

    symptoms = st.text_input("Type symptoms (comma separated):")

    st.subheader("🎤 Tamil Voice Input")
    audio = mic_recorder(start_prompt="Start recording", stop_prompt="Stop")

    if audio:
        st.success("Voice recorded (convert externally if needed)")

    days = st.number_input("How many days?", min_value=0)
    cold_food = st.selectbox("Cold items?", ["no", "yes"])
    fever = st.selectbox("Fever?", ["no", "yes"])

    # =========================
    # PREDICTION
    # =========================
    if st.button("Predict"):

        if symptoms.strip() == "":
            st.warning("Enter symptoms")
        else:
            user_symptoms = symptoms.lower().split(', ')

            def calculate_match(row_symptoms, user_symptoms):
                match = len(set(row_symptoms) & set(user_symptoms))
                total = len(row_symptoms)
                return match / total if total > 0 else 0

            df['Match_Percentage'] = df['Symptoms_List'].apply(
                lambda x: calculate_match(x, user_symptoms)
            )

            # Top 3
            st.subheader("🔝 Top 3 Predictions")
            top3 = df.sort_values(by='Match_Percentage', ascending=False).head(3)

            for i, row in top3.iterrows():
                st.write(f"{row['Disease']} - {round(row['Match_Percentage']*100,2)}%")

            result = top3.iloc[0]

            # =========================
            # SINUS LOGIC
            # =========================
            sinus_keywords = ["headache", "nasal congestion", "facial pain", "pressure around eyes", "runny nose"]

            sinus_match = len(set(sinus_keywords) & set(user_symptoms))

            if sinus_match >= 1:
                disease = "Sinusitis"
                confidence = 70 + sinus_match * 10
                st.warning("⚠️ Possible Sinus Detected")
            else:
                disease = result['Disease']
                confidence = round(result['Match_Percentage'] * 100, 2)

            # =========================
            # IMPROVE CONFIDENCE
            # =========================
            if len(user_symptoms) >= 2:
                confidence += 10
            if days > 3:
                confidence += 10
            if cold_food == "yes":
                confidence += 5
            if fever == "yes":
                confidence += 5

            confidence = min(confidence, 100)

            # =========================
            # SEVERITY
            # =========================
            if confidence < 40:
                severity = "Mild"
            elif confidence < 70:
                severity = "Moderate"
            else:
                severity = "Severe"

            # =========================
            # OUTPUT
            # =========================
            st.subheader("🔍 Final Result")
            st.write("Disease:", disease)
            st.write("Confidence:", confidence, "%")
            st.write("Severity:", severity)

            if confidence > 80:
                st.error("⚠️ High risk - consult doctor")

            # =========================
            # ADVICE
            # =========================
            st.subheader("💊 Advice")

            if disease in disease_info:
                for tip in disease_info[disease]["advice"]:
                    st.write("•", tip)
            else:
                st.write("• Take rest")
                st.write("• Drink fluids")

            # =========================
            # TAMIL OUTPUT
            # =========================
            st.subheader("🧠 Tamil Explanation")

            if disease in disease_info:
                st.write(disease_info[disease]["tamil"])
            else:
                st.write("இது பொதுவான உடல்நல பிரச்சனை ஆக இருக்கலாம்.")
