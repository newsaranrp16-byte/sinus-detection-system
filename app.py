import streamlit as st
import pandas as pd
import os

st.title("🧠 Sinus Detection System")

# =========================
# DEBUG: CHECK FILES
# =========================
st.write("📂 Files in directory:", os.listdir())

file_path = "dataset.csv"

# =========================
# LOAD DATASET (TAB FIX)
# =========================
if not os.path.exists(file_path):
    st.error("❌ dataset.csv NOT FOUND")
else:
    try:
        df = pd.read_csv(file_path, sep='\t')   # 🔥 FIX HERE
    except:
        df = pd.read_csv(file_path, sep='\t', encoding='latin1')

    st.success("✅ Dataset Loaded Successfully")

    # Debug columns
    st.write("Columns before fix:", df.columns)

    # =========================
    # FIX COLUMN NAMES
    # =========================
    df.columns = df.columns.str.strip()

    st.write("Columns after fix:", df.columns)

    # =========================
    # PREPROCESS
    # =========================
    df['Symptoms'] = df['Symptoms'].astype(str).str.lower()
    df['Symptoms_List'] = df['Symptoms'].apply(lambda x: x.split(', '))

    # =========================
    # USER INPUT
    # =========================
    symptoms = st.text_input("Enter symptoms (comma separated):")
    days = st.number_input("How many days have you had this problem?", min_value=0)
    cold_food = st.selectbox("Did you consume cold items?", ["no", "yes"])
    fever = st.selectbox("Do you have fever?", ["no", "yes"])

    # =========================
    # PREDICTION
    # =========================
    if st.button("Predict"):

        if symptoms.strip() == "":
            st.warning("Please enter symptoms")
        else:
            user_symptoms = symptoms.lower().split(', ')

            # Matching function
            def calculate_match(row_symptoms, user_symptoms):
                match = len(set(row_symptoms) & set(user_symptoms))
                total = len(row_symptoms)
                return match / total if total > 0 else 0

            df['Match_Percentage'] = df['Symptoms_List'].apply(
                lambda x: calculate_match(x, user_symptoms)
            )

            # =========================
            # TOP 3
            # =========================
            st.subheader("🔝 Top 3 Predictions")
            top3 = df.sort_values(by='Match_Percentage', ascending=False).head(3)

            for i, row in top3.iterrows():
                st.write(f"{row['Disease']} - {round(row['Match_Percentage']*100,2)}%")

            result = top3.iloc[0]

            # =========================
            # SINUS LOGIC
            # =========================
            sinus_keywords = ["headache", "nasal congestion", "facial pain", "pressure around eyes"]
            sinus_match = len(set(sinus_keywords) & set(user_symptoms))

            if sinus_match >= 2:
                disease = "Sinusitis"
                confidence = 60 + sinus_match * 10
                st.warning("⚠️ Possible Sinus Detected")
            else:
                disease = result['Disease']
                confidence = round(result['Match_Percentage'] * 100, 2)

            # =========================
            # UPDATE CONFIDENCE
            # =========================
            if days > 3:
                confidence += 10
            if cold_food == "yes":
                confidence += 5
            if fever == "yes":
                confidence += 10

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
            # FINAL OUTPUT
            # =========================
            st.subheader("🔍 Final Result")
            st.write("**Predicted Disease:**", disease)
            st.write("**Confidence:**", confidence, "%")
            st.write("**Severity Level:**", severity)

            if confidence > 80:
                st.error("⚠️ High risk - consult doctor")

            # =========================
            # ADVICE
            # =========================
            st.subheader("💊 Advice")
            if disease == "Sinusitis":
                st.write("• Avoid cold items")
                st.write("• Take steam inhalation")
                st.write("• Drink warm fluids")
            else:
                st.write("• Take rest")
                st.write("• Drink fluids")
                st.write("• Consult doctor if needed")

            # =========================
            # TAMIL OUTPUT
            # =========================
            st.subheader("🧠 Tamil Explanation")
            if disease == "Sinusitis":
                st.write("இது சைனஸ் பிரச்சனையாக இருக்க வாய்ப்பு அதிகம்.")
                st.write("மூக்கடைப்பு மற்றும் தலைவலி முக்கிய அறிகுறிகள்.")
                st.write("நீராவி பிடிக்கவும், குளிர்பானங்களை தவிர்க்கவும்.")
            else:
                st.write("இது பொதுவான உடல்நல பிரச்சனையாக இருக்கலாம்.")
                st.write("தேவையான ஓய்வு எடுக்கவும் மற்றும் மருத்துவரை அணுகவும்.")
