import streamlit as st
import pandas as pd
import os

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

# Fix column names
df.columns = df.columns.str.strip()

# =========================
# PREPROCESS
# =========================
df['Symptoms'] = df['Symptoms'].astype(str).str.lower()
df['Symptoms_List'] = df['Symptoms'].apply(lambda x: x.split(', '))

# =========================
# DISEASE INFO (ALL 30)
# =========================
disease_info = {
    "Allergy": {
        "advice": ["Avoid allergens", "Take antihistamines"],
        "english": "Avoid allergens like dust, pollen, and certain foods. Maintain hygiene and consult a doctor if symptoms worsen.",
        "tamil": "அலர்ஜி இருந்தால் தூசி மற்றும் காரணிகளை தவிர்க்கவும். சுத்தமாக இருங்கள்."
    },
    "Thyroid Disorder": {
        "advice": ["Take medicines", "Regular checkup"],
        "english": "Maintain hormone balance with medication and regular checkups.",
        "tamil": "தைராய்டு மருந்துகளை சரியாக எடுத்துக்கொள்ள வேண்டும்."
    },
    "Influenza": {
        "advice": ["Rest", "Drink fluids"],
        "english": "Take rest and stay hydrated to recover faster.",
        "tamil": "ஓய்வு எடுத்து நீர் குடிக்கவும்."
    },
    "Stroke": {
        "advice": ["Emergency care"],
        "english": "Stroke is an emergency. Seek immediate medical help.",
        "tamil": "ஸ்ட்ரோக் அவசரநிலை. உடனே மருத்துவரை அணுகவும்."
    },
    "Heart Disease": {
        "advice": ["Exercise", "Healthy diet"],
        "english": "Maintain a healthy lifestyle and avoid fatty foods.",
        "tamil": "ஆரோக்கியமான உணவு மற்றும் உடற்பயிற்சி அவசியம்."
    },
    "Food Poisoning": {
        "advice": ["ORS", "Rest"],
        "english": "Stay hydrated and avoid outside food.",
        "tamil": "ORS குடித்து ஓய்வு எடுக்கவும்."
    },
    "Bronchitis": {
        "advice": ["Avoid smoke", "Rest"],
        "english": "Avoid smoke and take rest.",
        "tamil": "புகை தவிர்த்து ஓய்வு எடுக்கவும்."
    },
    "COVID-19": {
        "advice": ["Isolate", "Mask"],
        "english": "Isolate and follow safety precautions.",
        "tamil": "தனிமைப்படுத்தி முககவசம் அணியவும்."
    },
    "Dermatitis": {
        "advice": ["Use creams"],
        "english": "Keep skin clean and use prescribed creams.",
        "tamil": "தோலை சுத்தமாக வைத்துக் கொள்ளுங்கள்."
    },
    "Diabetes": {
        "advice": ["Control sugar"],
        "english": "Maintain sugar levels and exercise.",
        "tamil": "சர்க்கரை அளவை கட்டுப்படுத்தவும்."
    },
    "Arthritis": {
        "advice": ["Exercise"],
        "english": "Do light exercise and avoid stress on joints.",
        "tamil": "மூட்டுகளை கவனமாக வைத்துக்கொள்ளுங்கள்."
    },
    "Sinusitis": {
        "advice": ["Steam inhalation"],
        "english": "Use steam inhalation and avoid cold.",
        "tamil": "நீராவி பிடிக்கவும், குளிரை தவிர்க்கவும்."
    },
    "Dementia": {
        "advice": ["Care"],
        "english": "Provide mental support and medical care.",
        "tamil": "மனஅழுத்தமின்றி கவனிக்கவும்."
    },
    "Parkinson's": {
        "advice": ["Medication"],
        "english": "Follow medication and exercise.",
        "tamil": "மருந்துகளை தவறாமல் எடுத்துக்கொள்ளவும்."
    },
    "Obesity": {
        "advice": ["Diet", "Exercise"],
        "english": "Maintain diet and exercise regularly.",
        "tamil": "உடல் எடையை கட்டுப்படுத்தவும்."
    },
    "Asthma": {
        "advice": ["Inhaler"],
        "english": "Avoid triggers and use inhaler.",
        "tamil": "தூசி தவிர்த்து இன்ஹேலர் பயன்படுத்தவும்."
    },
    "Depression": {
        "advice": ["Talk", "Relax"],
        "english": "Talk to someone and relax.",
        "tamil": "நண்பர்களுடன் பேசுங்கள்."
    },
    "Gastritis": {
        "advice": ["Avoid spicy food"],
        "english": "Eat regularly and avoid spicy food.",
        "tamil": "காரம் தவிர்க்கவும்."
    },
    "Liver Disease": {
        "advice": ["Avoid alcohol"],
        "english": "Avoid alcohol and follow healthy diet.",
        "tamil": "மதுபானம் தவிர்க்கவும்."
    },
    "Epilepsy": {
        "advice": ["Medication"],
        "english": "Take medicines regularly.",
        "tamil": "மருந்துகளை தவறாமல் எடுத்துக்கொள்ளுங்கள்."
    },
    "IBS": {
        "advice": ["Diet control"],
        "english": "Maintain diet and reduce stress.",
        "tamil": "உணவு முறையை சரி செய்யுங்கள்."
    },
    "Tuberculosis": {
        "advice": ["Complete treatment"],
        "english": "Complete full course of medication.",
        "tamil": "மருந்துகளை முழுமையாக எடுத்துக்கொள்ளுங்கள்."
    },
    "Pneumonia": {
        "advice": ["Rest"],
        "english": "Take rest and medications.",
        "tamil": "ஓய்வு எடுக்கவும்."
    },
    "Anemia": {
        "advice": ["Iron food"],
        "english": "Eat iron-rich foods.",
        "tamil": "இரும்புச்சத்து உணவுகள் எடுத்துக்கொள்ளுங்கள்."
    },
    "Migraine": {
        "advice": ["Avoid stress"],
        "english": "Avoid stress and take rest.",
        "tamil": "மனஅழுத்தம் தவிர்க்கவும்."
    },
    "Common Cold": {
        "advice": ["Rest"],
        "english": "Take rest and drink warm fluids.",
        "tamil": "ஓய்வு எடுக்கவும்."
    },
    "Anxiety": {
        "advice": ["Meditation"],
        "english": "Practice relaxation and meditation.",
        "tamil": "தியானம் செய்யுங்கள்."
    },
    "Chronic Kidney Disease": {
        "advice": ["Monitor kidney"],
        "english": "Regular checkups and diet control.",
        "tamil": "சிறுநீரக பரிசோதனை அவசியம்."
    },
    "Ulcer": {
        "advice": ["Avoid spicy food"],
        "english": "Avoid spicy food and eat on time.",
        "tamil": "காரம் தவிர்க்கவும்."
    },
    "Hypertension": {
        "advice": ["Reduce salt"],
        "english": "Reduce salt and exercise.",
        "tamil": "உப்பு குறைக்கவும்."
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
# PREDICTION
# =========================
if st.button("Predict"):

    user_symptoms = symptoms.lower().split(', ')

    def match(row):
        return len(set(row) & set(user_symptoms)) / len(row)

    df['Match'] = df['Symptoms_List'].apply(match)

    top3 = df.sort_values(by='Match', ascending=False).head(3)

    st.subheader("🔝 Top 3 Predictions")
    for _, row in top3.iterrows():
        st.write(f"{row['Disease']} - {round(row['Match']*100,2)}%")

    disease = top3.iloc[0]['Disease']
    confidence = round(top3.iloc[0]['Match'] * 100, 2)

    # Adjust confidence
    if days > 3: confidence += 10
    if cold_food == "yes": confidence += 5
    if fever == "yes": confidence += 10
    confidence = min(confidence, 100)

    # =========================
    # OUTPUT
    # =========================
    st.subheader("🔍 Final Result")
    st.write("Disease:", disease)
    st.write("Confidence:", confidence, "%")

    st.subheader("💊 Advice")
    for tip in disease_info[disease]["advice"]:
        st.write("•", tip)

    st.subheader("📘 English Explanation")
    st.write(disease_info[disease]["english"])

    st.subheader("🧠 தமிழ் விளக்கம்")
    st.write(disease_info[disease]["tamil"])
