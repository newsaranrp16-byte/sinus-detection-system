import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

st.title("🧠 Tamil Medical Assistant - Sinus Detection System")

# =========================
# LOAD DATASET
# =========================
file_path = "dataset.csv"

if not os.path.exists(file_path):
    st.error("❌ dataset.csv NOT FOUND")
    st.stop()

try:
    df = pd.read_csv(file_path, sep='\t')
except:
    df = pd.read_csv(file_path, sep=',', encoding='latin1')

df.columns = df.columns.str.strip()

# =========================
# PREPROCESS
# =========================
df['Symptoms'] = df['Symptoms'].astype(str).str.lower()
df['Symptoms_List'] = df['Symptoms'].apply(lambda x: [i.strip() for i in x.split(',')])

# =========================
# AUTO SYMPTOM CORRECTION
# =========================
symptom_map = {
    "stomach upset": "abdominal pain",
    "stomach pain": "abdominal pain",
    "cold": "runny nose",
    "blocked nose": "nasal congestion",
    "nose block": "nasal congestion",
    "head pain": "headache",
    "body pain": "muscle pain",
    "tired": "fatigue",
    "feeling weak": "fatigue",
    "throwing up": "vomiting"
}

def correct_symptoms(user_input):
    corrected = []
    words = user_input.lower().split(',')

    for w in words:
        w = w.strip()
        corrected.append(symptom_map.get(w, w))

    return corrected

# =========================
# DISEASE INFO (use your full dictionary here)
# =========================
disease_info = {

    "Allergy": {
        "advice": ["Avoid allergens", "Take antihistamines", "Consult doctor"],
        "english": "If you have allergies, it is important to identify and avoid triggers such as dust, pollen, certain foods, or environmental factors. Keeping your surroundings clean and maintaining good hygiene can help reduce exposure. Washing hands frequently and following a healthy lifestyle will improve immunity. If symptoms like sneezing, itching, or breathing difficulty become severe, consult a doctor and take prescribed medications properly.",
        "tamil": "உங்களுக்கு அலர்ஜி இருந்தால், அதற்கு காரணமான தூசி, சில உணவுகள் அல்லது பூமருவுகள் போன்றவற்றை தவிர்க்க வேண்டும். உங்கள் சுற்றுப்புறத்தை சுத்தமாக வைத்துக் கொள்ளுங்கள், அடிக்கடி கைகளை கழுவுங்கள், மற்றும் ஆரோக்கியமான வாழ்க்கை முறையை பின்பற்றுங்கள். அறிகுறிகள் அதிகமாக இருந்தால் மருத்துவரை அணுகி மருந்துகளை சரியாக எடுத்துக்கொள்ளுங்கள்."
    },

    "Thyroid Disorder": {
        "advice": ["Check hormone levels", "Take medicines regularly"],
        "english": "People with thyroid disorders should take medications regularly as prescribed by their doctor. Maintaining a balanced diet with proper iodine intake is essential for hormonal balance. Regular physical activity helps improve metabolism and overall health. It is also important to undergo periodic medical checkups to monitor hormone levels and adjust treatment accordingly.",
        "tamil": "தைராய்டு கோளாறு உள்ளவர்கள் நேரத்திற்கு மருந்துகளை எடுத்துக்கொள்வது மிகவும் முக்கியம். ஆரோக்கியமான உணவு முறையைப் பின்பற்றி, ஐயோடின் அளவு சரியாக உள்ள உணவுகளை எடுத்துக்கொள்ளுங்கள். உடற்பயிற்சி செய்வது உடல் சமநிலையை மேம்படுத்தும். அடிக்கடி மருத்துவரை சந்தித்து பரிசோதனை செய்து, ஹார்மோன் அளவை கண்காணிப்பதும் அவசியம்."
    },

    "Influenza": {
        "advice": ["Take rest", "Drink fluids"],
        "english": "Influenza (flu) requires proper rest and care to allow the body to recover. Drinking plenty of fluids such as warm water, soups, and juices helps maintain hydration and supports recovery. Consuming nutritious food improves immunity. If symptoms like fever, cough, or fatigue persist or worsen, it is important to consult a doctor and take medications as advised.",
        "tamil": "இன்ஃப்ளூயன்சா (காய்ச்சல்) வந்தால் அதிக ஓய்வு எடுத்து, சூடான நீர் மற்றும் சத்தான உணவுகளை உட்கொள்ள வேண்டும். தண்ணீர் அதிகமாக குடித்து உடலை நீர்ச்சத்து குறையாமல் பாதுகாத்துக்கொள்ளுங்கள். இருமல், காய்ச்சல் அதிகமாக இருந்தால் மருத்துவரை அணுகி மருந்துகளை சரியாக எடுத்துக்கொள்ள வேண்டும்."
    },

    "Common Cold": {
        "advice": ["Take rest", "Drink warm fluids"],
        "english": "Common cold is usually mild. Rest well and stay hydrated.",
        "tamil": "சாதாரண சளி. ஓய்வு எடுத்து வெந்நீர் குடிக்கவும்."
    },
    
    "Heart Disease": {
        "advice": ["Avoid oily food", "Exercise regularly"],
        "english": "To manage or prevent heart disease, it is important to follow a healthy lifestyle. Avoid foods that are high in oil, fat, and cholesterol. Regular exercise helps improve heart function and overall fitness. It is also essential to monitor blood pressure and sugar levels regularly. If symptoms such as chest pain, breathlessness, or fatigue occur, immediate medical consultation is necessary.",
        "tamil": "இதய நோய்களைத் தவிர்க்க சீரான உணவு முறையை பின்பற்றி, எண்ணெய் மற்றும் அதிக கொழுப்பு உள்ள உணவுகளை குறைக்க வேண்டும். தினமும் உடற்பயிற்சி செய்வது இதய ஆரோக்கியத்தை மேம்படுத்தும். புகைபிடித்தல், மதுபானம் போன்றவற்றை தவிர்த்து, இரத்த அழுத்தம் மற்றும் சர்க்கரை அளவை கட்டுப்பாட்டில் வைத்திருப்பதும் முக்கியம். மார்பு வலி, மூச்சுத்திணறல் போன்ற அறிகுறிகள் இருந்தால் உடனே மருத்துவரை அணுக வேண்டும்."
    },

    "Sinusitis": {
        "advice": ["Steam inhalation", "Avoid cold items"],
        "english": "Sinusitis occurs when the nasal passages become inflamed, leading to symptoms like nasal blockage, headache, and facial pressure. Steam inhalation helps clear nasal congestion and improves breathing. Drinking warm fluids keeps the body hydrated and reduces discomfort. Avoid exposure to cold air, dust, and allergens. If symptoms persist for several days or worsen, consult a doctor for proper treatment.",
        "tamil": "சைனஸைட்டிஸ் (Sinusitis) இருந்தால் வெதுவெதுப்பான நீர் ஆவி பிடித்தல் மூக்கடைப்பை குறைக்க உதவும். அதிகமாக தண்ணீர் குடித்து உடலை நீர்ச்சத்துடன் வைத்துக்கொள்ளுங்கள். தூசி, குளிர் காற்று போன்றவற்றை தவிர்க்கவும். தலைவலி, மூக்கடைப்பு நீடித்தால் மருத்துவரை அணுகி சரியான சிகிச்சை பெறுவது அவசியம்."
    }
}

# =========================
# INPUT
# =========================
symptoms = st.text_input("Enter symptoms (comma separated):")
days = st.number_input("How many days?", min_value=0)
cold_food = st.selectbox("Cold items?", ["no", "yes"])
fever = st.selectbox("Fever?", ["no", "yes"])

# =========================
# PREDICT
# =========================
if st.button("Predict"):

    if symptoms.strip() == "":
        st.warning("Please enter symptoms")
        st.stop()

    user_symptoms = correct_symptoms(symptoms)

    st.write("🔧 Corrected Symptoms:", user_symptoms)

    # =========================
    # IMPROVED MATCHING
    # =========================
    def match(row):
        common = set(row) & set(user_symptoms)
        return len(common)

    df['Match_Count'] = df['Symptoms_List'].apply(match)

    # Normalize score
    df['Match_Percentage'] = df['Match_Count'] / df['Symptoms_List'].apply(len)

    # Sort
    top = df.sort_values(by='Match_Count', ascending=False).head(5)

    # =========================
    # HANDLE ZERO MATCH
    # =========================
    if top['Match_Count'].max() == 0:
        disease = "General Checkup Recommended"
        confidence = 35
    else:
        best = top.iloc[0]
        disease = best['Disease']
        confidence = round(best['Match_Percentage'] * 100, 2)

    # =========================
    # BOOST CONFIDENCE
    # =========================
    if days > 3:
        confidence += 10
    if cold_food == "yes":
        confidence += 5
    if fever == "yes":
        confidence += 10

    confidence = min(confidence, 100)

    # =========================
    # SEVERITY COLOR
    # =========================
    if confidence < 40:
        severity = "Mild"
        color = "green"
    elif confidence < 70:
        severity = "Moderate"
        color = "orange"
    else:
        severity = "Severe"
        color = "red"

    # =========================
    # RESULT
    # =========================
    st.subheader("🔍 Final Result")
    st.write("**Predicted Disease:**", disease)
    st.write("**Confidence:**", confidence, "%")
    st.markdown(f"### Severity: <span style='color:{color}'>{severity}</span>", unsafe_allow_html=True)

    if severity == "Severe":
        st.error("⚠️ High risk - consult doctor")

    # =========================
    # BAR CHART (FIXED)
    # =========================
    st.subheader("📊 Disease Probability Distribution")

    chart_data = top[top['Match_Count'] > 0]

    if chart_data.empty:
        st.warning("No matching data to display chart")
    else:
        fig, ax = plt.subplots()
        ax.bar(chart_data['Disease'], chart_data['Match_Count'])
        plt.xticks(rotation=45)
        ax.set_title("Top Matching Diseases")
        st.pyplot(fig)

    # =========================
    # ADVICE
    # =========================
    st.subheader("💊 Advice")

    if disease in disease_info:
        for tip in disease_info[disease]["advice"]:
            st.write("•", tip)
    else:
        st.write("• Please consult doctor")

    # =========================
    # ENGLISH
    # =========================
    st.subheader("📘 English Explanation")

    if disease in disease_info:
        st.write(disease_info[disease]["english"])
    else:
        st.write("General symptoms detected. Please consult doctor.")

    # =========================
    # TAMIL
    # =========================
    st.subheader("🧠 தமிழ் விளக்கம்")

    if disease in disease_info:
        st.write(disease_info[disease]["tamil"])
    else:
        st.write("மருத்துவரை அணுகவும்.")
