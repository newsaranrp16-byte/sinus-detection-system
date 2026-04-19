import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Smart Tamil Medical Assistant", layout="wide")

st.title("🧠 Smart Tamil Medical Assistant (ML Powered)")

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("dataset.csv", sep="\t")
df.columns = df.columns.str.strip()

df['Symptoms'] = df['Symptoms'].astype(str).str.lower()

# =========================
# TRAIN ML MODEL
# =========================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Symptoms'])
y = df['Disease']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# =========================
# DISEASE RULES (HYBRID)
# =========================
age_restrictions = {
    "Stroke": 40,
    "Heart Disease": 35,
    "Hypertension": 30
}

# =========================
# DISEASE INFO
# =========================
disease_info = {
    "Allergy": {
        "advice": ["To prevent and manage Allergies, try to identify and avoid triggers such as dust, pollen, certain foods, or strong smells. Keep your surroundings clean by regularly dusting and washing bedding to reduce exposure to allergens. Maintain good personal hygiene, like washing hands and face after coming from outside. Eating a healthy diet and staying hydrated can support your immune system. If symptoms occur, follow medical advice and take prescribed medicines if needed. Regular check-ups can also help in controlling allergies effectively."],
        "tamil": "உங்களுக்கு அலர்ஜி இருந்தால், அதற்கு காரணமான தூசி, சில உணவுகள் அல்லது பூமருவுகள் போன்றவற்றை தவிர்க்க வேண்டும். உங்கள் சுற்றுப்புறத்தை சுத்தமாக வைத்துக் கொள்ளுங்கள், அடிக்கடி கைகளை கழுவுங்கள், மற்றும் ஆரோக்கியமான வாழ்க்கை முறையை பின்பற்றுங்கள். அறிகுறிகள் அதிகமாக இருந்தால் மருத்துவரை அணுகி மருந்துகளை சரியாக எடுத்துக்கொள்ளுங்கள்." 
    },
    "Thyroid Disorder": {
        "advice": ["To manage Thyroid disorders, it is important to maintain a healthy lifestyle and follow medical advice. Eat a balanced diet that includes iodine-rich foods like iodized salt, dairy, and certain vegetables, as iodine supports proper thyroid function. Regular exercise helps maintain body weight and energy levels. Take prescribed medicines on time if advised by a doctor, and go for regular check-ups to monitor hormone levels. Managing stress, getting enough sleep, and avoiding sudden changes in diet or medication can help keep the thyroid condition under control."],
        "tamil": "தைராய்டு கோளாறு உள்ளவர்கள் நேரத்திற்கு மருந்துகளை எடுத்துக்கொள்வது மிகவும் முக்கியம். ஆரோக்கியமான உணவு முறையைப் பின்பற்றி, ஐயோடின் அளவு சரியாக உள்ள உணவுகளை எடுத்துக்கொள்ளுங்கள். உடற்பயிற்சி செய்வது உடல் சமநிலையை மேம்படுத்தும். அடிக்கடி மருத்துவரை சந்தித்து பரிசோதனை செய்து, ஹார்மோன் அளவை கண்காணிப்பதும் அவசியம். "
    },
    "Influenza": {
        "advice": ["To prevent Influenza, maintain good hygiene and a healthy routine. Wash your hands regularly with soap, especially before eating and after being in public places. Cover your mouth and nose with a tissue or your elbow when coughing or sneezing to stop the spread of germs. Avoid close contact with people who are sick, and try to stay home and rest if you feel unwell. Eating nutritious food, drinking enough water, and getting proper sleep can strengthen your immune system. Taking a yearly flu vaccine also helps protect you from serious illness."]
        "tamil": "இன்ஃப்ளூயன்சா (காய்ச்சல்) வந்தால் அதிக ஓய்வு எடுத்து, சூடான நீர் மற்றும் சத்தான உணவுகளை உட்கொள்ள வேண்டும். தண்ணீர் அதிகமாக குடித்து உடலை நீர்ச்சத்து குறையாமல் பாதுகாத்துக்கொள்ளுங்கள். இருமல், காய்ச்சல் அதிகமாக இருந்தால் மருத்துவரை அணுகி மருந்துகளை சரியாக எடுத்துக்கொள்ள வேண்டும்." 
    },
    "Stroke": {
        "advice": ["To reduce the risk of stroke, maintain a healthy lifestyle by eating a balanced diet with plenty of fruits, vegetables, whole grains, and low-fat foods while avoiding excess salt and processed items. Regular physical activity, such as walking or light exercise for at least 30 minutes a day, helps improve blood circulation and keeps the body fit. It is also important to manage stress, get enough sleep, and keep conditions like high blood pressure and diabetes under control through regular check-ups. Taking these steps can help protect your brain and overall health."],
        "tamil": "ஸ்ட்ரோக் ஏற்படாமல் இருக்க இரத்த அழுத்தம், சர்க்கரை அளவு போன்றவற்றை கட்டுப்பாட்டில் வைத்திருக்க வேண்டும். ஆரோக்கியமான உணவு முறையை பின்பற்றி, புகைபிடித்தல் மற்றும் மதுபானத்தை தவிர்க்கவும். திடீர் தலைச்சுற்றல், பேச முடியாமை, உடல் ஒரு பக்கம் பலவீனம் போன்ற அறிகுறிகள் தெரிந்தால் உடனே மருத்துவரை அணுகுவது மிகவும் அவசியம்." 
    },
    "Heart Disease": {
        "advice": ["To reduce the risk of heart disease, follow a healthy lifestyle by eating a balanced diet rich in fruits, vegetables, whole grains, and lean proteins while avoiding too much oily, salty, or processed food. Regular physical activity, such as walking or simple exercises for at least 30 minutes a day, helps keep the heart strong. Managing stress through relaxation techniques, proper sleep, and staying calm is also important. Regular health check-ups can help detect problems early and keep your heart healthy."],
        "tamil": "இதய நோய்களைத் தவிர்க்க சீரான உணவு முறையை பின்பற்றி, எண்ணெய் மற்றும் அதிக கொழுப்பு உள்ள உணவுகளை குறைக்க வேண்டும். தினமும் உடற்பயிற்சி செய்வது இதய ஆரோக்கியத்தை மேம்படுத்தும். புகைபிடித்தல், மதுபானம் போன்றவற்றை தவிர்த்து, இரத்த அழுத்தம் மற்றும் சர்க்கரை அளவை கட்டுப்பாட்டில் வைத்திருப்பதும் முக்கியம். மார்பு வலி, மூச்சுத்திணறல் போன்ற அறிகுறிகள் இருந்தால் உடனே மருத்துவரை அணுக வேண்டும்." 
    },
    "Food Poisoning": {
        "advice": ["To prevent and manage Food poisoning, always eat fresh and properly cooked food and avoid items that look or smell spoiled. Wash fruits and vegetables thoroughly before eating, and keep your hands, utensils, and cooking area clean to stop the spread of germs. Drink safe and clean water, and avoid eating from unhygienic places. Store food properly, especially perishable items, to prevent contamination. If symptoms like vomiting or stomach pain occur, rest well, stay hydrated, and seek medical help if the condition becomes severe."],
        "tamil": "உணவு விஷத்தன்மை (Food poisoning) ஏற்பட்டால் உடனே ஓய்வு எடுத்து, அதிகமாக தண்ணீர் அல்லது ORS குடித்து உடல் நீர்ச்சத்தை பேண வேண்டும். சுத்தமான, எளிதில் செரிமானமாகும் உணவுகளை மட்டும் எடுத்துக்கொள்ளுங்கள். வாந்தி, வயிற்றுப்போக்கு அதிகமாக இருந்தால் அல்லது நீடித்தால் உடனே மருத்துவரை அணுகுவது அவசியம்." 
    },
    "Bronchitis": {
        "advice": ["To prevent and manage Bronchitis, maintain good respiratory health by avoiding smoke, dust, and polluted air that can irritate the lungs. Do not smoke, and stay away from people who are smoking. Drink plenty of fluids to keep the throat moist and help loosen mucus. Get enough rest and eat healthy foods to support your immune system. Practicing good hygiene, like washing hands regularly, can help prevent infections. If symptoms like persistent cough or breathing difficulty occur, seek medical advice for proper treatment."],
        "tamil": "பிராங்கைட்டிஸ் (Bronchitis) ஏற்பட்டால் அதிகமாக ஓய்வு எடுத்து, வெதுவெதுப்பான நீர் குடிப்பது மிகவும் உதவும். புகை, தூசி போன்றவற்றை தவிர்க்க வேண்டும், ஏனெனில் அவை இருமலை அதிகரிக்கலாம். நீடித்த இருமல், சளி அல்லது மூச்சுத்திணறல் இருந்தால் மருத்துவரை அணுகி சரியான சிகிச்சை பெற வேண்டும்."
    },
    "COVID-19": {
        "advice": ["To prevent and manage COVID-19, follow good hygiene and safety practices. Wash your hands regularly with soap or use sanitizer, especially after being in public places. Cover your mouth and nose when coughing or sneezing, and avoid close contact with people who are sick. Wearing a mask in crowded or poorly ventilated areas can help reduce the spread of infection. Eat nutritious food, stay hydrated, and get enough rest to keep your immune system strong. If you feel unwell, isolate yourself and seek medical advice if symptoms become severe."],
        "tamil": "COVID-19 ஏற்பட்டால் வீட்டிலேயே ஓய்வு எடுத்து, தனிமைப்படுத்திக் கொள்ள வேண்டும். முககவசம் அணிந்து, கைகளை அடிக்கடி சுத்தம் செய்து, பிறரிடம் தொற்று பரவாமல் கவனிக்க வேண்டும். காய்ச்சல், இருமல், மூச்சுத்திணறல் போன்ற அறிகுறிகள் அதிகமாக இருந்தால் உடனே மருத்துவரை அணுகி சரியான சிகிச்சை பெறுவது மிகவும் அவசியம்."
    },
    "Dermatitis": {
        "advice": ["To prevent and manage Dermatitis, keep your skin clean and moisturized using mild soaps and gentle creams. Avoid contact with substances that irritate your skin, such as harsh chemicals, strong detergents, or certain fabrics. Wear comfortable, breathable clothing to reduce irritation. Do not scratch affected areas, as it can worsen the condition. Eating a healthy diet, staying hydrated, and managing stress can also support skin health. If symptoms like redness, itching, or swelling persist, consult a doctor for proper treatment."],
        "tamil": "டெர்மட்டிட்டிஸ் (Dermatitis) இருந்தால் தோலை சுத்தமாகவும் ஈரப்பதமாகவும் வைத்துக்கொள்ள வேண்டும். கடுமையான சோப்பு, ரசாயன பொருட்கள் போன்றவற்றை தவிர்க்கவும். அரிப்பு அல்லது சிவப்பு அதிகமாக இருந்தால் சொறியாமல் இருந்து, மருத்துவரின் ஆலோசனையின்படி மருந்துகள் அல்லது கிரீம்களை பயன்படுத்துவது முக்கியம்."
    },
    "Diabetes": {
        "advice": ["To prevent and manage Diabetes, maintain a healthy lifestyle by eating a balanced diet with less sugar and refined foods, and more fruits, vegetables, and whole grains. Regular physical activity, such as walking or exercise for at least 30 minutes a day, helps control blood sugar levels. Maintain a healthy body weight and avoid smoking. It is important to monitor blood sugar regularly and follow the doctor’s advice, including taking medicines if prescribed. Getting enough sleep and managing stress also help in keeping diabetes under control."],
        "tamil": "நீரிழிவு (Diabetes) உள்ளவர்கள் சர்க்கரை அளவை கட்டுப்பாட்டில் வைத்திருக்க வேண்டும். சீரான உணவு முறையை பின்பற்றி, இனிப்புகள் மற்றும் அதிக கார்போஹைட்ரேட் உள்ள உணவுகளை குறைக்க வேண்டும். தினமும் உடற்பயிற்சி செய்து, மருத்துவரின் ஆலோசனையின்படி மருந்துகள் அல்லது இன்சுலின் எடுத்துக்கொள்வது மிகவும் முக்கியம்."
    },
    "Arthritis": {
        "advice": ["To prevent and manage Arthritis, maintain a healthy lifestyle by staying physically active with gentle exercises like walking or stretching to keep joints flexible. Maintain a healthy weight to reduce pressure on the joints. Eat a balanced diet that includes calcium and vitamin-rich foods to support bone health. Avoid overstraining your joints and take proper rest when needed. Applying warm or cold compresses may help reduce pain and stiffness. If symptoms persist, follow medical advice and take prescribed treatment to manage the condition effectively."],
        "tamil": "ஆர்திரைட்டிஸ் (Arthritis) உள்ளவர்கள் மூட்டுகளை அதிகமாக அழுத்தம் தரும் செயல்களை தவிர்க்க வேண்டும். லேசான உடற்பயிற்சி மற்றும் நீட்டிப்பு பயிற்சிகள் மூட்டுகளின் இயக்கத்தை மேம்படுத்த உதவும். உடல் எடையை கட்டுப்பாட்டில் வைத்திருப்பதும் முக்கியம். வலி அல்லது வீக்கம் அதிகமாக இருந்தால் மருத்துவரை அணுகி சரியான சிகிச்சை பெற வேண்டும்." 
    },
    "Sinusitis": {
        "advice": ["To prevent and manage Sinusitis, maintain good hygiene and avoid exposure to dust, smoke, and pollution that can irritate the nasal passages. Drink plenty of fluids to keep mucus thin and promote drainage. Inhaling steam can help relieve nasal congestion and ease breathing. Get enough rest and eat healthy foods to support your immune system. Keep your surroundings clean to reduce allergens. If symptoms like facial pain, headache, or blocked nose persist, seek medical advice for proper treatment."],
        "tamil": "சைனஸைட்டிஸ் (Sinusitis) இருந்தால் வெதுவெதுப்பான நீர் ஆவி பிடித்தல் (steam inhalation) மூக்கடைப்பை குறைக்க உதவும். அதிகமாக தண்ணீர் குடித்து உடலை நீர்ச்சத்துடன் வைத்துக்கொள்ளுங்கள். தூசி, குளிர் காற்று போன்றவற்றை தவிர்க்கவும். தலைவலி, மூக்கடைப்பு நீடித்தால் மருத்துவரை அணுகி சரியான சிகிச்சை பெறுவது அவசியம்."
    },
    "Dementia": {
        "advice": ["To support and manage Dementia, maintain a healthy lifestyle by eating a balanced diet rich in fruits, vegetables, and whole grains. Keep the brain active through reading, puzzles, or learning new skills. Regular physical activity, such as walking, helps improve blood flow to the brain. Ensure proper sleep and manage stress, as both affect memory and thinking ability. Staying socially active by talking and spending time with others can also support mental health. Regular medical check-ups are important for early detection and proper care."],
        "tamil": "டிமென்ஷியா (Dementia) உள்ளவர்களுக்கு அமைதியான மற்றும் பாதுகாப்பான சூழலை உருவாக்குவது முக்கியம். நினைவாற்றலை தூண்டும் செயல்கள் (எளிய விளையாட்டுகள், உரையாடல்) உதவும். நேரத்திற்கு உணவு மற்றும் மருந்துகளை கொடுத்து, அவர்களை கவனமாக பார்த்துக் கொள்ள வேண்டும். அறிகுறிகள் அதிகரித்தால் மருத்துவரை அணுகி ஆலோசனை பெறுவது அவசியம்."
    },
    "Parkinson's": {
        "advice": ["To manage Parkinson's disease, follow a healthy routine with regular exercise such as walking or simple stretching to improve movement and balance. Eat a balanced diet with nutritious foods to support overall health. Take medicines exactly as prescribed by the doctor and attend regular check-ups. Getting enough rest and managing stress can help reduce symptoms. Practicing daily activities carefully and staying active socially and mentally can also improve quality of life."],
        "tamil": "பார்கின்சன் நோய் (Parkinson’s disease) உள்ளவர்கள் மருந்துகளை நேரத்திற்கு எடுத்துக்கொள்வது மிகவும் முக்கியம். லேசான உடற்பயிற்சி மற்றும் உடல் சமநிலையை மேம்படுத்தும் பயிற்சிகள் உதவும். சீரான உணவு மற்றும் போதுமான ஓய்வு அவசியம். நடுக்கம், இயக்கம் மந்தமாகுதல் போன்ற அறிகுறிகள் அதிகரித்தால் மருத்துவரை அணுகி சரியான சிகிச்சை பெற வேண்டும்."
    },
    "Obesity": {
        "advice": ["To prevent and manage Obesity, follow a healthy lifestyle by eating a balanced diet with more fruits, vegetables, and whole grains while reducing sugary, oily, and junk foods. Regular physical activity, such as walking, cycling, or simple exercises for at least 30 minutes a day, helps maintain a healthy weight. Drink plenty of water and avoid overeating. Getting enough sleep and managing stress are also important, as they can affect weight. Regular health check-ups can help monitor progress and maintain overall health."],
        "tamil": "அதிக உடல் எடை (Obesity) குறைக்க சீரான உணவு முறையை பின்பற்றி, அதிக கொழுப்பு மற்றும் ஜங்க் உணவுகளை தவிர்க்க வேண்டும். தினமும் உடற்பயிற்சி செய்வது மிகவும் முக்கியம். போதுமான தூக்கம் மற்றும் தண்ணீர் உட்கொள்ளலும் உதவும். உடல் எடையை ஆரோக்கியமாக கட்டுப்படுத்த மருத்துவரின் ஆலோசனையையும் பின்பற்றலாம்."
    },
    "Asthma": {
        "advice": ["To prevent and manage Asthma, avoid triggers such as dust, smoke, pollution, and strong smells that can cause breathing problems. Keep your surroundings clean and well-ventilated. Use prescribed inhalers or medicines regularly as advised by a doctor. Regular light exercise and breathing exercises can help improve lung function. Drink plenty of fluids and get enough rest to support your health. If symptoms like wheezing or shortness of breath worsen, seek medical help promptly."],
        "tamil": "ஆஸ்துமா (Asthma) உள்ளவர்கள் தூசி, புகை, குளிர் காற்று போன்ற தூண்டுதல்களை தவிர்க்க வேண்டும். மருத்துவர் கூறிய இன்ஹேலர் அல்லது மருந்துகளை நேரத்திற்கு பயன்படுத்துவது முக்கியம். சுவாச பயிற்சிகள் மற்றும் சீரான வாழ்க்கை முறை உதவும். மூச்சுத்திணறல் அதிகமாக இருந்தால் உடனே மருத்துவரை அணுக வேண்டும்."
    },
    "Depression": {
        "advice": ["To manage Depression, it is important to take care of both your mind and body. Try to maintain a daily routine, get enough sleep, and eat healthy, balanced meals. Regular physical activity, even simple exercises like walking, can help improve mood. Stay connected with family and friends and talk about your feelings instead of keeping them inside. Engaging in activities you enjoy, such as hobbies or listening to music, can also help. If feelings of sadness or low energy continue for a long time, it is important to seek support from a trusted adult or a healthcare professional."],
        "tamil": "மனச்சோர்வு (Depression) ஏற்பட்டால் தனியாக இருக்காமல் நம்பிக்கையுள்ள நண்பர்கள் அல்லது குடும்பத்தினருடன் பேசுவது உதவும். தினசரி சிறிய செயல்களில் ஈடுபட்டு, ஒழுங்கான தூக்கம் மற்றும் உணவு பழக்கத்தை பேண வேண்டும். நீண்ட நாட்களாக சோகம், ஆர்வமின்மை நீடித்தால் மனநல மருத்துவரை அணுகி ஆலோசனை பெறுவது மிகவும் முக்கியம்."
    },
    "Gastritis": {
        "advice": ["To prevent and manage Gastritis, eat regular, balanced meals and avoid very spicy, oily, or junk foods that can irritate the stomach. Do not skip meals, as an empty stomach can worsen symptoms. Drink plenty of water and choose light, easily digestible foods. Manage stress through relaxation and proper rest, as stress can affect digestion. Avoid overeating and eat slowly. If you have frequent stomach pain, nausea, or discomfort, seek medical advice for proper treatment."],
        "tamil": "காஸ்ட்ரைட்டிஸ் (Gastritis) இருந்தால் காரம், எண்ணெய் மற்றும் அமிலம் அதிகமான உணவுகளை தவிர்க்க வேண்டும். நேரத்திற்கு உணவு எடுத்துக்கொண்டு, வெதுவெதுப்பான நீர் குடிப்பது உதவும். காலியான வயிற்றில் அதிக நேரம் இருக்காமல் பார்த்துக் கொள்ளுங்கள். வயிற்று வலி அல்லது எரிச்சல் நீடித்தால் மருத்துவரை அணுகி சிகிச்சை பெற வேண்டும்."
    },
    "Liver Disease": {
        "advice": ["To prevent and manage Liver disease, follow a healthy lifestyle by eating a balanced diet with more fruits, vegetables, and whole grains while avoiding very oily and processed foods. Maintain a healthy weight and stay physically active with regular exercise. Drink clean water and ensure good hygiene to prevent infections. Avoid taking medicines unnecessarily and only use them as prescribed by a doctor, as some medicines can affect the liver. Getting enough rest and regular medical check-ups can help keep the liver healthy and detect problems early."],
        "tamil": "கல்லீரல் நோய் (Liver disease) உள்ளவர்கள் மதுபானத்தை முழுமையாக தவிர்க்க வேண்டும். சீரான, சத்தான உணவு முறையை பின்பற்றி எண்ணெய் மற்றும் கொழுப்பு அதிகமான உணவுகளை குறைக்க வேண்டும். மருத்துவர் கூறிய மருந்துகளை நேரத்திற்கு எடுத்துக்கொண்டு, அடிக்கடி பரிசோதனை செய்து கல்லீரல் செயல்பாட்டை கண்காணிப்பதும் மிகவும் முக்கியம்." 
    },
    "Epilepsy": {
        "advice": ["To manage epilepsy, it is important to take medicines regularly as prescribed by a doctor and never skip doses. Get enough sleep, as lack of rest can trigger seizures. Avoid stress and try to stay calm and relaxed. Eat a healthy, balanced diet and stay hydrated. Keep a regular routine and avoid anything that may trigger seizures, such as flashing lights (for some people). Regular check-ups with a doctor help in monitoring the condition and adjusting treatment if needed."],
        "tamil": "முர்ச்சை நோய் (Epilepsy) உள்ளவர்கள் மருந்துகளை தவறாமல் நேரத்திற்கு எடுத்துக்கொள்வது மிகவும் முக்கியம். போதுமான தூக்கம் பெற வேண்டும் மற்றும் அதிகமான மன அழுத்தத்தை தவிர்க்க வேண்டும். திடீர் fits ஏற்பட்டால் பாதுகாப்பாக படுக்க வைத்து, உடனே மருத்துவரை அணுக வேண்டும்."
    },
    "IBS": {
        "advice": ["To manage Irritable Bowel Syndrome, follow a healthy routine by eating regular, balanced meals and avoiding foods that trigger symptoms, such as very spicy, oily, or gas-producing foods. Drink plenty of water and include fiber-rich foods to support digestion, but increase fiber slowly. Managing stress through relaxation, exercise, or hobbies is important, as stress can worsen symptoms. Get enough sleep and maintain a regular eating schedule. If symptoms like stomach pain, bloating, or irregular bowel movements continue, consult a doctor for proper guidance and treatment."],
        "tamil": "இரிடபிள் பவல் சிண்ட்ரோம் (IBS) உள்ளவர்கள் சீரான உணவு முறையை பின்பற்றி, காரம் மற்றும் எண்ணெய் அதிகமான உணவுகளை தவிர்க்க வேண்டும். மன அழுத்தத்தை குறைக்க முயற்சி செய்து, போதுமான தண்ணீர் குடிக்கவும். நார்ச்சத்து (fiber) உள்ள உணவுகளை மெதுவாக சேர்த்துக் கொள்ளலாம். வயிற்று வலி, மலச்சிக்கல் அல்லது வயிற்றுப்போக்கு தொடர்ந்து இருந்தால் மருத்துவரை அணுகுவது அவசியம்."
    },
    "Tuberculosis": {
        "advice": ["To prevent and manage Tuberculosis, maintain good hygiene and ensure proper ventilation in living spaces. Cover your mouth and nose while coughing or sneezing to prevent the spread of infection. Eat a nutritious diet and get enough rest to strengthen your immune system. If diagnosed, take all prescribed medicines regularly and complete the full course of treatment without stopping early. Avoid close contact with others until advised by a doctor, and attend regular check-ups to monitor recovery."],
        "tamil": "காசநோய் (Tuberculosis) உள்ளவர்கள் மருத்துவர் கொடுத்த மருந்துகளை முழுமையாகவும் நேரத்திற்கு எடுத்துக்கொள்வது மிகவும் முக்கியம். சத்தான உணவு மற்றும் போதுமான ஓய்வு உடல் நலத்தை மேம்படுத்தும். இருமும் போது வாயை மூடி, பிறருக்கு தொற்று பரவாமல் கவனிக்க வேண்டும். நீண்ட நாட்கள் இருமல், உடல் எடை குறைவு போன்ற அறிகுறிகள் இருந்தால் உடனே மருத்துவரை அணுக வேண்டும்."
    },
    "Pneumonia": {
        "advice": ["To prevent and manage Pneumonia, maintain good hygiene by washing hands regularly and avoiding close contact with people who are sick. Keep your surroundings clean and well-ventilated. Eat a healthy diet, drink plenty of fluids, and get enough rest to support your immune system. Stay warm and avoid exposure to cold or polluted air. Vaccination can help protect against certain types of pneumonia. If symptoms like fever, cough, or difficulty breathing occur, seek medical attention promptly."],
        "tamil": "நிமோனியா (Pneumonia) ஏற்பட்டால் போதுமான ஓய்வு எடுத்து, அதிகமாக தண்ணீர் மற்றும் சூடான பானங்கள் குடிப்பது உதவும். மருத்துவர் கூறிய ஆன்டிபயாட்டிக் அல்லது மருந்துகளை நேரத்திற்கு எடுத்துக்கொள்ள வேண்டும். இருமல், காய்ச்சல், மூச்சுத்திணறல் அதிகமாக இருந்தால் உடனே மருத்துவரை அணுகுவது மிகவும் அவசியம்."
    },
    "Anemia": {
        "advice": ["To prevent and manage Anemia, eat a balanced diet rich in iron, such as green leafy vegetables, beans, dates, and whole grains. Include foods rich in vitamin C, like citrus fruits, to help the body absorb iron better. Maintain good nutrition and avoid skipping meals. Get enough rest and stay active with light exercise. Regular health check-ups can help monitor hemoglobin levels. If symptoms like tiredness, weakness, or pale skin appear, consult a doctor for proper treatment."],
        "tamil": "அனீமியா (Anemia) உள்ளவர்கள் இரும்புச்சத்து (iron) அதிகம் உள்ள உணவுகள் যেমন கீரை, பேரீச்சம் பழம், பருப்பு வகைகள் ஆகியவற்றை உணவில் சேர்க்க வேண்டும். சீரான உணவு முறையுடன், மருத்துவர் பரிந்துரைக்கும் இரும்பு மாத்திரைகளை எடுத்துக்கொள்வதும் முக்கியம். அதிக சோர்வு, தலைச்சுற்றல் போன்ற அறிகுறிகள் இருந்தால் மருத்துவரை அணுக வேண்டும்."
    },
    "Migraine": {
        "advice": ["To prevent and manage Migraine, maintain a regular daily routine with proper sleep and balanced meals. Avoid triggers such as stress, bright lights, loud noise, or skipping meals, as these can bring on headaches. Drink plenty of water to stay hydrated. Rest in a quiet, dark room when a headache starts, and try relaxation techniques to reduce stress. Regular exercise can also help improve overall health. If migraines occur frequently or become severe, consult a doctor for proper guidance and treatment."],
        "tamil": "மைக்ரேன் (Migraine) உள்ளவர்கள் தூக்கம் குறைவாக இருக்காமல் பார்த்துக் கொண்டு, அதிக ஒலி மற்றும் வெளிச்சத்தை தவிர்க்க வேண்டும். நேரத்திற்கு உணவு எடுத்துக்கொண்டு, மன அழுத்தத்தை குறைக்க முயற்சி செய்யுங்கள். தலைவலி அதிகமாக இருந்தால் மருத்துவர் கூறிய மருந்துகளை எடுத்துக்கொண்டு, அடிக்கடி தாக்கம் வந்தால் மருத்துவரை அணுகுவது அவசியம்."
    },
    "Common Cold": {
        "advice": ["To prevent and manage the Common cold, maintain good hygiene by washing your hands regularly and avoiding close contact with people who are sick. Cover your mouth and nose while coughing or sneezing to prevent the spread of germs. Drink plenty of fluids, eat nutritious food, and get enough rest to help your body recover. Keep your surroundings clean and stay warm. If symptoms like runny nose, cough, or mild fever persist, take proper care and consult a doctor if needed."],
        "tamil": "சாதாரண சளி (Common cold) ஏற்பட்டால் அதிகமாக ஓய்வு எடுத்து, சூடான நீர் மற்றும் சூப் போன்றவற்றை குடிப்பது உதவும். கைகளை அடிக்கடி கழுவி சுத்தமாக வைத்துக் கொள்ளுங்கள். இருமல், மூக்கோட்டம் நீடித்தால் அல்லது காய்ச்சல் அதிகமாக இருந்தால் மருத்துவரை அணுகுவது நல்லது."
    },
    "Anxiety": {
        "advice": ["To manage Anxiety disorder, try to maintain a calm and balanced routine. Practice relaxation techniques like deep breathing, meditation, or simple mindfulness to reduce worry. Regular physical activity, such as walking, can help improve mood and reduce tension. Get enough sleep and eat healthy meals to support your overall well-being. Stay connected with family and friends and talk about your feelings instead of keeping them inside. If anxiety becomes strong or continues for a long time, seek support from a trusted adult or a healthcare professional."],
        "tamil": "பதட்டம் (Anxiety) இருந்தால் ஆழ்ந்த சுவாச பயிற்சிகள் செய்து மனதை அமைதியாக வைத்துக் கொள்ள முயற்சி செய்யுங்கள். ஒழுங்கான தூக்கம் மற்றும் தினசரி உடற்பயிற்சி மனநிலையை மேம்படுத்த உதவும். நம்பிக்கையுள்ளவர்களுடன் பேசுவது பயனுள்ளதாக இருக்கும். பதட்டம் நீண்ட நாட்கள் தொடர்ந்தால் மருத்துவர் அல்லது மனநல நிபுணரை அணுகுவது முக்கியம்."
    },
    "Chronic Kidney Disease": {
        "advice": ["To prevent and manage Chronic kidney disease, follow a healthy lifestyle by eating a balanced diet with less salt and processed foods to reduce strain on the kidneys. Drink enough water as advised by your doctor and maintain a healthy blood pressure and blood sugar level. Regular physical activity helps improve overall health. Avoid taking medicines unnecessarily, especially painkillers, without medical advice, as they can affect kidney function. Get regular check-ups to monitor kidney health and follow the doctor’s instructions carefully."],
        "tamil": "நீண்டகால சிறுநீரக நோய் (Chronic Kidney Disease) உள்ளவர்கள் உப்பு மற்றும் புரதம் அளவை கட்டுப்படுத்திய உணவு முறையை பின்பற்ற வேண்டும். மருத்துவர் கூறிய மருந்துகளை நேரத்திற்கு எடுத்துக்கொண்டு, ரத்த அழுத்தம் மற்றும் சர்க்கரை அளவை கட்டுப்பாட்டில் வைத்திருக்க வேண்டும். போதுமான தண்ணீர் குடிப்பது மற்றும் அடிக்கடி பரிசோதனை செய்வதும் மிகவும் முக்கியம்."
    },
    "Ulcer": {
        "advice": ["To prevent and manage Peptic ulcer, eat regular, balanced meals and avoid very spicy, oily, or acidic foods that can irritate the stomach lining. Do not skip meals, and try to eat at the same time each day. Manage stress through relaxation and proper rest, as stress can worsen symptoms. Avoid taking medicines without a doctor’s advice, especially painkillers that may harm the stomach. Drink plenty of water and choose light, easily digestible foods. If symptoms like burning stomach pain or discomfort persist, consult a doctor for proper treatment."],
        "tamil": "புண் (Ulcer) இருந்தால் காரம், புளிப்பு மற்றும் எண்ணெய் அதிகமான உணவுகளை தவிர்க்க வேண்டும். நேரத்திற்கு உணவு எடுத்துக்கொண்டு, காலியான வயிற்றில் நீண்ட நேரம் இருக்காமல் பார்த்துக் கொள்ளுங்கள். மன அழுத்தத்தை குறைப்பதும் உதவும். வயிற்று வலி அல்லது எரிச்சல் நீடித்தால் மருத்துவரை அணுகி சரியான சிகிச்சை பெறுவது அவசியம்." 
    },
    
    "Hypertension": {
        "advice": ["To prevent and manage Hypertension, follow a healthy lifestyle by eating a balanced diet with less salt and more fruits, vegetables, and whole grains. Regular physical activity, such as walking or simple exercise for at least 30 minutes a day, helps keep blood pressure under control. Maintain a healthy body weight and avoid smoking. Managing stress through relaxation, proper sleep, and staying calm is also important. Regular health check-ups can help monitor blood pressure and prevent complications."],
        "tamil": "உயர் இரத்த அழுத்தம் (Hypertension) உள்ளவர்கள் உப்பு அளவை குறைத்து, சீரான மற்றும் ஆரோக்கியமான உணவு முறையை பின்பற்ற வேண்டும். தினமும் உடற்பயிற்சி செய்து, மன அழுத்தத்தை கட்டுப்படுத்துவது முக்கியம். புகைபிடித்தல் மற்றும் மதுபானத்தை தவிர்க்கவும். ரத்த அழுத்தத்தை அடிக்கடி பரிசோதித்து, மருத்துவர் கூறிய மருந்துகளை நேரத்திற்கு எடுத்துக்கொள்ள வேண்டும்."
    },
}

# =========================
# SYMPTOMS LIST
# =========================
all_symptoms = sorted(set(
    [sym.strip() for row in df['Symptoms'] for sym in row.split(',')]
))

# =========================
# INPUT
# =========================
st.subheader("🩺 Select Symptoms")
selected_symptoms = st.multiselect("Choose symptoms", all_symptoms)

age = st.number_input("Age", 1, 100)
days = st.number_input("How many days?", 0, 30)
fever = st.selectbox("Fever?", ["No", "Yes"])

# =========================
# PREDICT
# =========================
if st.button("Predict"):

    if not selected_symptoms:
        st.warning("Please select symptoms")
        st.stop()

    user_input = " ".join(selected_symptoms)
    user_vec = vectorizer.transform([user_input])

    # ML Prediction
    probs = model.predict_proba(user_vec)[0]
    classes = model.classes_

    results = dict(zip(classes, probs * 100))

    # =========================
    # APPLY AGE FILTER
    # =========================
    for disease in results:
        if disease in age_restrictions:
            if age < age_restrictions[disease]:
                results[disease] *= 0.1   # reduce strongly

    # =========================
    # BOOST CONDITIONS
    # =========================
    if fever == "Yes":
        for disease in results:
            if "fever" in disease.lower():
                results[disease] += 5

    # =========================
    # SORT RESULTS
    # =========================
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    top3 = sorted_results[:3]

    best_disease, confidence = top3[0]

    # =========================
    # EMERGENCY ALERT
    # =========================
    if "chest pain" in selected_symptoms or "breathing difficulty" in selected_symptoms:
        st.error("🚨 Please consult doctor immediately")

    # =========================
    # OUTPUT
    # =========================
    st.subheader("🔝 Top 3 Predictions")
    for d, c in top3:
        st.write(f"{d} → {round(c,2)}%")

    st.subheader("🔍 Final Result")
    st.success(f"🩺 {best_disease}")
    st.write("Confidence:", round(confidence,2), "%")

    # Severity
    if confidence < 40:
        st.markdown("### 🟢 Mild")
    elif confidence < 70:
        st.markdown("### 🟠 Moderate")
    else:
        st.markdown("### 🔴 Severe")

    # WHY
    st.subheader("🤖 Why this disease?")
    st.write("Based on ML pattern matching of your symptoms")

    # =========================
    # ADVICE
    # =========================
    st.subheader("💊 Advice")
    info = disease_info.get(best_disease)

    if info:
        for tip in info["advice"]:
            st.write("•", tip)
    else:
        st.write("• Please consult doctor")

    # =========================
    # TAMIL
    # =========================
    st.subheader("🧠 தமிழ் விளக்கம்")

    if info:
        st.write(info["tamil"])
    else:
        st.write("மருத்துவரை அணுகவும்.")
