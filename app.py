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
df['Symptoms_List'] = df['Symptoms'].apply(lambda x: x.split(', '))

# =========================
# 🔥 PASTE YOUR FULL 30 DISEASE DATA HERE
# =========================
disease_info = {

    "Allergy": {
        "advice": ["Avoid allergens", "Take antihistamines", "Consult doctor"],
        "english": "If you are experiencing allergies, it is important to identify and avoid triggers such as dust, pollen, certain foods, or environmental factors. Keeping your surroundings clean and maintaining good hygiene helps reduce exposure. Regular hand washing and following a healthy lifestyle can improve your immunity. If symptoms such as sneezing, itching, or breathing difficulty become severe, consult a doctor and take medications as prescribed.",
        "tamil": "உங்களுக்கு அலர்ஜி இருந்தால், அதற்கு காரணமான தூசி, சில உணவுகள் அல்லது பூமருவுகள் போன்றவற்றை தவிர்க்க வேண்டும். உங்கள் சுற்றுப்புறத்தை சுத்தமாக வைத்துக் கொள்ளுங்கள், அடிக்கடி கைகளை கழுவுங்கள், மற்றும் ஆரோக்கியமான வாழ்க்கை முறையை பின்பற்றுங்கள். அறிகுறிகள் அதிகமாக இருந்தால் மருத்துவரை அணுகி மருந்துகளை சரியாக எடுத்துக்கொள்ளுங்கள்."
    },

    "Thyroid Disorder": {
        "advice": ["Check hormone levels", "Take prescribed medicines"],
        "english": "People with thyroid disorders should take medications regularly as prescribed by their doctor. Maintaining a balanced diet with proper iodine intake is important for hormonal balance. Regular physical activity improves metabolism and overall health. Periodic medical checkups are necessary to monitor hormone levels and adjust treatment accordingly.",
        "tamil": "தைராய்டு கோளாறு உள்ளவர்கள் நேரத்திற்கு மருந்துகளை எடுத்துக்கொள்வது மிகவும் முக்கியம். ஆரோக்கியமான உணவு முறையைப் பின்பற்றி, ஐயோடின் அளவு சரியாக உள்ள உணவுகளை எடுத்துக்கொள்ளுங்கள். உடற்பயிற்சி செய்வது உடல் சமநிலையை மேம்படுத்தும். அடிக்கடி மருத்துவரை சந்தித்து பரிசோதனை செய்து, ஹார்மோன் அளவை கண்காணிப்பதும் அவசியம்."
    },

    "Influenza": {
        "advice": ["Take rest", "Drink fluids", "Consult doctor"],
        "english": "Influenza (flu) requires proper rest and hydration to allow the body to recover. Drinking warm fluids such as water, soups, and juices helps maintain hydration and supports recovery. Eating nutritious food strengthens the immune system. If symptoms like fever, cough, or fatigue worsen or persist, consult a doctor for proper treatment.",
        "tamil": "இன்ஃப்ளூயன்சா (காய்ச்சல்) வந்தால் அதிக ஓய்வு எடுத்து, சூடான நீர் மற்றும் சத்தான உணவுகளை உட்கொள்ள வேண்டும். தண்ணீர் அதிகமாக குடித்து உடலை நீர்ச்சத்து குறையாமல் பாதுகாத்துக்கொள்ளுங்கள். இருமல், காய்ச்சல் அதிகமாக இருந்தால் மருத்துவரை அணுகி மருந்துகளை சரியாக எடுத்துக்கொள்ள வேண்டும்."
    },

    "Stroke": {
        "advice": ["Immediate medical attention", "Do not delay"],
        "english": "Stroke is a serious medical emergency and requires immediate attention. It is important to control blood pressure, sugar levels, and cholesterol to prevent it. Following a healthy lifestyle and avoiding smoking and alcohol can reduce risk. If symptoms such as sudden weakness, speech difficulty, or dizziness occur, seek medical help immediately.",
        "tamil": "ஸ்ட்ரோக் ஏற்படாமல் இருக்க இரத்த அழுத்தம், சர்க்கரை அளவு போன்றவற்றை கட்டுப்பாட்டில் வைத்திருக்க வேண்டும். ஆரோக்கியமான உணவு முறையை பின்பற்றி, புகைபிடித்தல் மற்றும் மதுபானத்தை தவிர்க்கவும். திடீர் தலைச்சுற்றல், பேச முடியாமை, உடல் ஒரு பக்கம் பலவீனம் போன்ற அறிகுறிகள் தெரிந்தால் உடனே மருத்துவரை அணுகுவது மிகவும் அவசியம்."
    },

    "Heart Disease": {
        "advice": ["Avoid oily food", "Exercise regularly", "Consult doctor"],
        "english": "To manage or prevent heart disease, it is important to follow a healthy lifestyle. Avoid foods high in oil, fat, and cholesterol. Regular exercise helps improve heart function and overall health. Monitoring blood pressure and sugar levels is essential. If symptoms such as chest pain or breathlessness occur, consult a doctor immediately.",
        "tamil": "இதய நோய்களைத் தவிர்க்க சீரான உணவு முறையை பின்பற்றி, எண்ணெய் மற்றும் அதிக கொழுப்பு உள்ள உணவுகளை குறைக்க வேண்டும். தினமும் உடற்பயிற்சி செய்வது இதய ஆரோக்கியத்தை மேம்படுத்தும். புகைபிடித்தல், மதுபானம் போன்றவற்றை தவிர்த்து, இரத்த அழுத்தம் மற்றும் சர்க்கரை அளவை கட்டுப்பாட்டில் வைத்திருப்பதும் முக்கியம். மார்பு வலி, மூச்சுத்திணறல் போன்ற அறிகுறிகள் இருந்தால் உடனே மருத்துவரை அணுக வேண்டும்."
    },

    "Food Poisoning": {
        "advice": ["Drink ORS", "Avoid outside food", "Rest"],
        "english": "Food poisoning occurs due to contaminated food or water. It is important to stay hydrated by drinking ORS or plenty of fluids. Eat light and easily digestible foods. Avoid outside or unhygienic food. If symptoms like vomiting or diarrhea persist, consult a doctor immediately.",
        "tamil": "உணவு விஷத்தன்மை (Food poisoning) ஏற்பட்டால் உடனே ஓய்வு எடுத்து, அதிகமாக தண்ணீர் அல்லது ORS குடித்து உடல் நீர்ச்சத்தை பேண வேண்டும். சுத்தமான, எளிதில் செரிமானமாகும் உணவுகளை மட்டும் எடுத்துக்கொள்ளுங்கள். வாந்தி, வயிற்றுப்போக்கு அதிகமாக இருந்தால் அல்லது நீடித்தால் உடனே மருத்துவரை அணுகுவது அவசியம்."
    },

    "Sinusitis": {
        "advice": ["Steam inhalation", "Avoid cold items"],
        "english": "Sinusitis occurs when nasal passages become inflamed, causing symptoms like headache, nasal blockage, and facial pressure. Steam inhalation helps clear congestion and improves breathing. Drink warm fluids and avoid cold foods. If symptoms persist for several days, consult a doctor for treatment.",
        "tamil": "சைனஸைட்டிஸ் (Sinusitis) இருந்தால் வெதுவெதுப்பான நீர் ஆவி பிடித்தல் (steam inhalation) மூக்கடைப்பை குறைக்க உதவும். அதிகமாக தண்ணீர் குடித்து உடலை நீர்ச்சத்துடன் வைத்துக்கொள்ளுங்கள். தூசி, குளிர் காற்று போன்றவற்றை தவிர்க்கவும். தலைவலி, மூக்கடைப்பு நீடித்தால் மருத்துவரை அணுகி சரியான சிகிச்சை பெறுவது அவசியம்."
    },

    "Diabetes": {
        "advice": ["Control sugar", "Exercise", "Regular checkup"],
        "english": "Diabetes requires proper management of blood sugar levels. Follow a balanced diet and avoid excessive sugar intake. Regular exercise improves insulin sensitivity. Monitor sugar levels regularly and follow medications as prescribed by your doctor.",
        "tamil": "நீரிழிவு (Diabetes) உள்ளவர்கள் சர்க்கரை அளவை கட்டுப்பாட்டில் வைத்திருக்க வேண்டும். சீரான உணவு முறையை பின்பற்றி, இனிப்புகள் மற்றும் அதிக கார்போஹைட்ரேட் உள்ள உணவுகளை குறைக்க வேண்டும். தினமும் உடற்பயிற்சி செய்து, மருத்துவரின் ஆலோசனையின்படி மருந்துகள் அல்லது இன்சுலின் எடுத்துக்கொள்வது மிகவும் முக்கியம்."
    },

    "Common Cold": {
        "advice": ["Rest", "Warm fluids"],
        "english": "Common cold is usually mild and can be managed with rest and proper hydration. Drinking warm fluids helps relieve symptoms. Maintain hygiene and avoid spreading infection. If symptoms persist or worsen, consult a doctor.",
        "tamil": "சாதாரண சளி (Common cold) ஏற்பட்டால் அதிகமாக ஓய்வு எடுத்து, சூடான நீர் மற்றும் சூப் போன்றவற்றை குடிப்பது உதவும். கைகளை அடிக்கடி கழுவி சுத்தமாக வைத்துக் கொள்ளுங்கள். இருமல், மூக்கோட்டம் நீடித்தால் அல்லது காய்ச்சல் அதிகமாக இருந்தால் மருத்துவரை அணுகுவது நல்லது."
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

    if symptoms.strip() == "":
        st.warning("Please enter symptoms")
        st.stop()

    user_symptoms = symptoms.lower().split(', ')

    def match(row):
        return len(set(row) & set(user_symptoms)) / len(row)

    df['Match'] = df['Symptoms_List'].apply(match)

    # 🔥 TAKE TOP 5 FOR PIE CHART
    top = df.sort_values(by='Match', ascending=False).head(5)

    best = top.iloc[0]
    disease = best['Disease']
    confidence = round(best['Match'] * 100, 2)

    # =========================
    # IMPROVE CONFIDENCE
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
        st.error("⚠️ High risk - consult doctor immediately")

    # =========================
    # 📊 PIE CHART
    # =========================
    st.subheader("📊 Disease Probability Distribution")

    labels = top['Disease']
    values = top['Match']

    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%')
    ax.set_title("Top Possible Diseases")

    st.pyplot(fig)

    # =========================
    # ADVICE
    # =========================
    st.subheader("💊 Advice")

    if disease in disease_info:
        for tip in disease_info[disease]["advice"]:
            st.write("•", tip)
    else:
        st.write("• Take rest")
        st.write("• Consult doctor")

    # =========================
    # ENGLISH
    # =========================
    st.subheader("📘 English Explanation")

    if disease in disease_info:
        st.write(disease_info[disease]["english"])
    else:
        st.write("General health issue. Please consult doctor.")

    # =========================
    # TAMIL
    # =========================
    st.subheader("🧠 தமிழ் விளக்கம்")

    if disease in disease_info:
        st.write(disease_info[disease]["tamil"])
    else:
        st.write("மருத்துவரை அணுகவும்.")

