import streamlit as st

st.set_page_config(page_title="Smart Tamil Medical Assistant", layout="wide")

st.title("🧠 Smart Tamil Medical Assistant")

# =========================
# SYMPTOMS LIST
# =========================
all_symptoms = [
    "chest pain","shortness of breath","fatigue","abdominal pain","vomiting",
    "nausea","headache","nasal congestion","facial pain","runny nose","sneezing",
    "cough","wheezing","weight loss","frequent urination","diarrhea","insomnia",
    "anxiety","dizziness","fever","joint pain","swelling","rash","itching",
    "breathing difficulty","light sensitivity"
]

# =========================
# DISEASE RULES
# =========================
disease_rules = {
    "Heart Disease": ["chest pain", "shortness of breath", "fatigue"],
    "Gastritis": ["abdominal pain", "vomiting", "nausea"],
    "Sinusitis": ["headache", "nasal congestion", "facial pain"],
    "Common Cold": ["runny nose", "sneezing", "cough"],
    "Asthma": ["shortness of breath", "wheezing", "cough"],
    "Diabetes": ["fatigue", "weight loss", "frequent urination"],
    "Food Poisoning": ["vomiting", "diarrhea", "abdominal pain"],
    "Depression": ["fatigue", "insomnia", "anxiety"],
    "Hypertension": ["headache", "dizziness", "chest pain"],
    "Migraine": ["headache", "nausea", "light sensitivity"],
    "Tuberculosis": ["cough", "weight loss", "fever"],
    "Pneumonia": ["fever", "cough", "chest pain"],
    "Arthritis": ["joint pain", "swelling"],
    "Dermatitis": ["rash", "itching"],
    "COVID-19": ["fever", "cough", "breathing difficulty"],
    "Allergy": ["sneezing", "rash", "runny nose"],
    "Bronchitis": ["cough", "fatigue", "breathing difficulty"],
    "Stroke": ["dizziness", "headache"],
    "Thyroid Disorder": ["fatigue", "weight loss"],
    "Anemia": ["fatigue", "dizziness"],
    "Obesity": ["fatigue"],
    "Parkinson's": ["fatigue"],
    "Epilepsy": ["dizziness"],
    "IBS": ["abdominal pain", "diarrhea"],
    "Liver Disease": ["abdominal pain", "fatigue"],
    "Anxiety": ["anxiety", "insomnia"],
    "Chronic Kidney Disease": ["fatigue"],
    "Ulcer": ["abdominal pain"],
    "Influenza": ["fever", "cough"],
    "Sinus Infection": ["headache", "nasal congestion"]
}

# =========================
# AGE RESTRICTIONS (STRICT)
# =========================
age_restrictions = {
    "Stroke": 40,
    "Heart Disease": 35,
    "Hypertension": 30
}

# =========================
# STRONG SYMPTOMS REQUIRED
# =========================
strong_rules = {
    "Stroke": ["dizziness"],
    "Heart Disease": ["chest pain"],
    "Asthma": ["breathing difficulty"]
}

# =========================
# DISEASE INFO
# =========================
disease_info = {
    "Allergy": {
        "advice": ["Avoid allergens", "Take antihistamines", "Consult doctor"],
        "tamil": "உங்களுக்கு அலர்ஜி இருந்தால், அதற்கு காரணமான தூசி, சில உணவுகள் அல்லது பூமருவுகள் போன்றவற்றை தவிர்க்க வேண்டும். உங்கள் சுற்றுப்புறத்தை சுத்தமாக வைத்துக் கொள்ளுங்கள், அடிக்கடி கைகளை கழுவுங்கள், மற்றும் ஆரோக்கியமான வாழ்க்கை முறையை பின்பற்றுங்கள். அறிகுறிகள் அதிகமாக இருந்தால் மருத்துவரை அணுகி மருந்துகளை சரியாக எடுத்துக்கொள்ளுங்கள்." 
    },
    "Thyroid Disorder": {
        "advice": ["Check hormone levels", "Take prescribed medicines"],
        "tamil": "தைராய்டு கோளாறு உள்ளவர்கள் நேரத்திற்கு மருந்துகளை எடுத்துக்கொள்வது மிகவும் முக்கியம். ஆரோக்கியமான உணவு முறையைப் பின்பற்றி, ஐயோடின் அளவு சரியாக உள்ள உணவுகளை எடுத்துக்கொள்ளுங்கள். உடற்பயிற்சி செய்வது உடல் சமநிலையை மேம்படுத்தும். அடிக்கடி மருத்துவரை சந்தித்து பரிசோதனை செய்து, ஹார்மோன் அளவை கண்காணிப்பதும் அவசியம். "
    },
    "Influenza": {
        "advice": ["Take rest", "Drink fluids", "Consult doctor"],
        "tamil": "இன்ஃப்ளூயன்சா (காய்ச்சல்) வந்தால் அதிக ஓய்வு எடுத்து, சூடான நீர் மற்றும் சத்தான உணவுகளை உட்கொள்ள வேண்டும். தண்ணீர் அதிகமாக குடித்து உடலை நீர்ச்சத்து குறையாமல் பாதுகாத்துக்கொள்ளுங்கள். இருமல், காய்ச்சல் அதிகமாக இருந்தால் மருத்துவரை அணுகி மருந்துகளை சரியாக எடுத்துக்கொள்ள வேண்டும்." 
    },
    "Stroke": {
        "advice": ["Immediate medical attention", "Do not delay"],
        "tamil": "ஸ்ட்ரோக் ஏற்படாமல் இருக்க இரத்த அழுத்தம், சர்க்கரை அளவு போன்றவற்றை கட்டுப்பாட்டில் வைத்திருக்க வேண்டும். ஆரோக்கியமான உணவு முறையை பின்பற்றி, புகைபிடித்தல் மற்றும் மதுபானத்தை தவிர்க்கவும். திடீர் தலைச்சுற்றல், பேச முடியாமை, உடல் ஒரு பக்கம் பலவீனம் போன்ற அறிகுறிகள் தெரிந்தால் உடனே மருத்துவரை அணுகுவது மிகவும் அவசியம்." 
    },
    "Heart Disease": {
        "advice": ["Avoid oily food", "Exercise regularly", "Consult doctor"],
        "tamil": "இதய நோய்களைத் தவிர்க்க சீரான உணவு முறையை பின்பற்றி, எண்ணெய் மற்றும் அதிக கொழுப்பு உள்ள உணவுகளை குறைக்க வேண்டும். தினமும் உடற்பயிற்சி செய்வது இதய ஆரோக்கியத்தை மேம்படுத்தும். புகைபிடித்தல், மதுபானம் போன்றவற்றை தவிர்த்து, இரத்த அழுத்தம் மற்றும் சர்க்கரை அளவை கட்டுப்பாட்டில் வைத்திருப்பதும் முக்கியம். மார்பு வலி, மூச்சுத்திணறல் போன்ற அறிகுறிகள் இருந்தால் உடனே மருத்துவரை அணுக வேண்டும்." 
    },
    "Food Poisoning": {
        "advice": ["Drink ORS", "Avoid outside food", "Rest"],
        "tamil": "உணவு விஷத்தன்மை (Food poisoning) ஏற்பட்டால் உடனே ஓய்வு எடுத்து, அதிகமாக தண்ணீர் அல்லது ORS குடித்து உடல் நீர்ச்சத்தை பேண வேண்டும். சுத்தமான, எளிதில் செரிமானமாகும் உணவுகளை மட்டும் எடுத்துக்கொள்ளுங்கள். வாந்தி, வயிற்றுப்போக்கு அதிகமாக இருந்தால் அல்லது நீடித்தால் உடனே மருத்துவரை அணுகுவது அவசியம்." 
    },
    "Bronchitis": {
        "advice": ["Avoid smoke", "Use inhaler", "Rest"],
        "tamil": "பிராங்கைட்டிஸ் (Bronchitis) ஏற்பட்டால் அதிகமாக ஓய்வு எடுத்து, வெதுவெதுப்பான நீர் குடிப்பது மிகவும் உதவும். புகை, தூசி போன்றவற்றை தவிர்க்க வேண்டும், ஏனெனில் அவை இருமலை அதிகரிக்கலாம். நீடித்த இருமல், சளி அல்லது மூச்சுத்திணறல் இருந்தால் மருத்துவரை அணுகி சரியான சிகிச்சை பெற வேண்டும்."
    },
    "COVID-19": {
        "advice": ["Isolate", "Wear mask", "Consult doctor"],
        "tamil": "COVID-19 ஏற்பட்டால் வீட்டிலேயே ஓய்வு எடுத்து, தனிமைப்படுத்திக் கொள்ள வேண்டும். முககவசம் அணிந்து, கைகளை அடிக்கடி சுத்தம் செய்து, பிறரிடம் தொற்று பரவாமல் கவனிக்க வேண்டும். காய்ச்சல், இருமல், மூச்சுத்திணறல் போன்ற அறிகுறிகள் அதிகமாக இருந்தால் உடனே மருத்துவரை அணுகி சரியான சிகிச்சை பெறுவது மிகவும் அவசியம்."
    },
    "Dermatitis": {
        "advice": ["Avoid irritants", "Use creams"],
        "tamil": "டெர்மட்டிட்டிஸ் (Dermatitis) இருந்தால் தோலை சுத்தமாகவும் ஈரப்பதமாகவும் வைத்துக்கொள்ள வேண்டும். கடுமையான சோப்பு, ரசாயன பொருட்கள் போன்றவற்றை தவிர்க்கவும். அரிப்பு அல்லது சிவப்பு அதிகமாக இருந்தால் சொறியாமல் இருந்து, மருத்துவரின் ஆலோசனையின்படி மருந்துகள் அல்லது கிரீம்களை பயன்படுத்துவது முக்கியம்."
    },
    "Diabetes": {
        "advice": ["Control sugar", "Exercise", "Regular checkup"],
        "tamil": "நீரிழிவு (Diabetes) உள்ளவர்கள் சர்க்கரை அளவை கட்டுப்பாட்டில் வைத்திருக்க வேண்டும். சீரான உணவு முறையை பின்பற்றி, இனிப்புகள் மற்றும் அதிக கார்போஹைட்ரேட் உள்ள உணவுகளை குறைக்க வேண்டும். தினமும் உடற்பயிற்சி செய்து, மருத்துவரின் ஆலோசனையின்படி மருந்துகள் அல்லது இன்சுலின் எடுத்துக்கொள்வது மிகவும் முக்கியம்."
    },
    "Arthritis": {
        "advice": ["Exercise", "Pain relief therapy"],
        "tamil": "ஆர்திரைட்டிஸ் (Arthritis) உள்ளவர்கள் மூட்டுகளை அதிகமாக அழுத்தம் தரும் செயல்களை தவிர்க்க வேண்டும். லேசான உடற்பயிற்சி மற்றும் நீட்டிப்பு பயிற்சிகள் மூட்டுகளின் இயக்கத்தை மேம்படுத்த உதவும். உடல் எடையை கட்டுப்பாட்டில் வைத்திருப்பதும் முக்கியம். வலி அல்லது வீக்கம் அதிகமாக இருந்தால் மருத்துவரை அணுகி சரியான சிகிச்சை பெற வேண்டும்." 
    },
    "Sinusitis": {
        "advice": ["Steam inhalation", "Avoid cold items"],
        "tamil": "சைனஸைட்டிஸ் (Sinusitis) இருந்தால் வெதுவெதுப்பான நீர் ஆவி பிடித்தல் (steam inhalation) மூக்கடைப்பை குறைக்க உதவும். அதிகமாக தண்ணீர் குடித்து உடலை நீர்ச்சத்துடன் வைத்துக்கொள்ளுங்கள். தூசி, குளிர் காற்று போன்றவற்றை தவிர்க்கவும். தலைவலி, மூக்கடைப்பு நீடித்தால் மருத்துவரை அணுகி சரியான சிகிச்சை பெறுவது அவசியம்."
    },
    "Dementia": {
        "advice": ["Mental exercise", "Medical care"],
        "tamil": "டிமென்ஷியா (Dementia) உள்ளவர்களுக்கு அமைதியான மற்றும் பாதுகாப்பான சூழலை உருவாக்குவது முக்கியம். நினைவாற்றலை தூண்டும் செயல்கள் (எளிய விளையாட்டுகள், உரையாடல்) உதவும். நேரத்திற்கு உணவு மற்றும் மருந்துகளை கொடுத்து, அவர்களை கவனமாக பார்த்துக் கொள்ள வேண்டும். அறிகுறிகள் அதிகரித்தால் மருத்துவரை அணுகி ஆலோசனை பெறுவது அவசியம்."
    },
    "Parkinson's": {
        "advice": ["Medication", "Exercise"],
        "tamil": "பார்கின்சன் நோய் (Parkinson’s disease) உள்ளவர்கள் மருந்துகளை நேரத்திற்கு எடுத்துக்கொள்வது மிகவும் முக்கியம். லேசான உடற்பயிற்சி மற்றும் உடல் சமநிலையை மேம்படுத்தும் பயிற்சிகள் உதவும். சீரான உணவு மற்றும் போதுமான ஓய்வு அவசியம். நடுக்கம், இயக்கம் மந்தமாகுதல் போன்ற அறிகுறிகள் அதிகரித்தால் மருத்துவரை அணுகி சரியான சிகிச்சை பெற வேண்டும்."
    },
    "Obesity": {
        "advice": ["Diet control", "Exercise"],
        "tamil": "அதிக உடல் எடை (Obesity) குறைக்க சீரான உணவு முறையை பின்பற்றி, அதிக கொழுப்பு மற்றும் ஜங்க் உணவுகளை தவிர்க்க வேண்டும். தினமும் உடற்பயிற்சி செய்வது மிகவும் முக்கியம். போதுமான தூக்கம் மற்றும் தண்ணீர் உட்கொள்ளலும் உதவும். உடல் எடையை ஆரோக்கியமாக கட்டுப்படுத்த மருத்துவரின் ஆலோசனையையும் பின்பற்றலாம்."
    },
    "Asthma": {
        "advice": ["Avoid dust", "Use inhaler"],
        "tamil": "ஆஸ்துமா (Asthma) உள்ளவர்கள் தூசி, புகை, குளிர் காற்று போன்ற தூண்டுதல்களை தவிர்க்க வேண்டும். மருத்துவர் கூறிய இன்ஹேலர் அல்லது மருந்துகளை நேரத்திற்கு பயன்படுத்துவது முக்கியம். சுவாச பயிற்சிகள் மற்றும் சீரான வாழ்க்கை முறை உதவும். மூச்சுத்திணறல் அதிகமாக இருந்தால் உடனே மருத்துவரை அணுக வேண்டும்."
    },
    "Depression": {
        "advice": ["Talk to someone", "Relaxation"],
        "tamil": "மனச்சோர்வு (Depression) ஏற்பட்டால் தனியாக இருக்காமல் நம்பிக்கையுள்ள நண்பர்கள் அல்லது குடும்பத்தினருடன் பேசுவது உதவும். தினசரி சிறிய செயல்களில் ஈடுபட்டு, ஒழுங்கான தூக்கம் மற்றும் உணவு பழக்கத்தை பேண வேண்டும். நீண்ட நாட்களாக சோகம், ஆர்வமின்மை நீடித்தால் மனநல மருத்துவரை அணுகி ஆலோசனை பெறுவது மிகவும் முக்கியம்."
    },
    "Gastritis": {
        "advice": ["Avoid spicy food", "Eat regularly"],
        "tamil": "காஸ்ட்ரைட்டிஸ் (Gastritis) இருந்தால் காரம், எண்ணெய் மற்றும் அமிலம் அதிகமான உணவுகளை தவிர்க்க வேண்டும். நேரத்திற்கு உணவு எடுத்துக்கொண்டு, வெதுவெதுப்பான நீர் குடிப்பது உதவும். காலியான வயிற்றில் அதிக நேரம் இருக்காமல் பார்த்துக் கொள்ளுங்கள். வயிற்று வலி அல்லது எரிச்சல் நீடித்தால் மருத்துவரை அணுகி சிகிச்சை பெற வேண்டும்."
    },
    "Liver Disease": {
        "advice": ["Avoid alcohol", "Healthy diet"],
        "tamil": "கல்லீரல் நோய் (Liver disease) உள்ளவர்கள் மதுபானத்தை முழுமையாக தவிர்க்க வேண்டும். சீரான, சத்தான உணவு முறையை பின்பற்றி எண்ணெய் மற்றும் கொழுப்பு அதிகமான உணவுகளை குறைக்க வேண்டும். மருத்துவர் கூறிய மருந்துகளை நேரத்திற்கு எடுத்துக்கொண்டு, அடிக்கடி பரிசோதனை செய்து கல்லீரல் செயல்பாட்டை கண்காணிப்பதும் மிகவும் முக்கியம்." 
    },
    "Epilepsy": {
        "advice": ["Take medicines regularly"],
        "tamil": "முர்ச்சை நோய் (Epilepsy) உள்ளவர்கள் மருந்துகளை தவறாமல் நேரத்திற்கு எடுத்துக்கொள்வது மிகவும் முக்கியம். போதுமான தூக்கம் பெற வேண்டும் மற்றும் அதிகமான மன அழுத்தத்தை தவிர்க்க வேண்டும். திடீர் fits ஏற்பட்டால் பாதுகாப்பாக படுக்க வைத்து, உடனே மருத்துவரை அணுக வேண்டும்."
    },
    "IBS": {
        "advice": ["Diet control", "Avoid stress"],
        "tamil": "இரிடபிள் பவல் சிண்ட்ரோம் (IBS) உள்ளவர்கள் சீரான உணவு முறையை பின்பற்றி, காரம் மற்றும் எண்ணெய் அதிகமான உணவுகளை தவிர்க்க வேண்டும். மன அழுத்தத்தை குறைக்க முயற்சி செய்து, போதுமான தண்ணீர் குடிக்கவும். நார்ச்சத்து (fiber) உள்ள உணவுகளை மெதுவாக சேர்த்துக் கொள்ளலாம். வயிற்று வலி, மலச்சிக்கல் அல்லது வயிற்றுப்போக்கு தொடர்ந்து இருந்தால் மருத்துவரை அணுகுவது அவசியம்."
    },
    "Tuberculosis": {
        "advice": ["Complete medication", "Isolate"],
        "tamil": "காசநோய் (Tuberculosis) உள்ளவர்கள் மருத்துவர் கொடுத்த மருந்துகளை முழுமையாகவும் நேரத்திற்கு எடுத்துக்கொள்வது மிகவும் முக்கியம். சத்தான உணவு மற்றும் போதுமான ஓய்வு உடல் நலத்தை மேம்படுத்தும். இருமும் போது வாயை மூடி, பிறருக்கு தொற்று பரவாமல் கவனிக்க வேண்டும். நீண்ட நாட்கள் இருமல், உடல் எடை குறைவு போன்ற அறிகுறிகள் இருந்தால் உடனே மருத்துவரை அணுக வேண்டும்."
    },
    "Pneumonia": {
        "advice": ["Take antibiotics", "Rest"],
        "tamil": "நிமோனியா (Pneumonia) ஏற்பட்டால் போதுமான ஓய்வு எடுத்து, அதிகமாக தண்ணீர் மற்றும் சூடான பானங்கள் குடிப்பது உதவும். மருத்துவர் கூறிய ஆன்டிபயாட்டிக் அல்லது மருந்துகளை நேரத்திற்கு எடுத்துக்கொள்ள வேண்டும். இருமல், காய்ச்சல், மூச்சுத்திணறல் அதிகமாக இருந்தால் உடனே மருத்துவரை அணுகுவது மிகவும் அவசியம்."
    },
    "Anemia": {
        "advice": ["Iron-rich food", "Supplements"],
        "tamil": "அனீமியா (Anemia) உள்ளவர்கள் இரும்புச்சத்து (iron) அதிகம் உள்ள உணவுகள் যেমন கீரை, பேரீச்சம் பழம், பருப்பு வகைகள் ஆகியவற்றை உணவில் சேர்க்க வேண்டும். சீரான உணவு முறையுடன், மருத்துவர் பரிந்துரைக்கும் இரும்பு மாத்திரைகளை எடுத்துக்கொள்வதும் முக்கியம். அதிக சோர்வு, தலைச்சுற்றல் போன்ற அறிகுறிகள் இருந்தால் மருத்துவரை அணுக வேண்டும்."
    },
    "Migraine": {
        "advice": ["Avoid stress", "Take rest"],
        "tamil": "மைக்ரேன் (Migraine) உள்ளவர்கள் தூக்கம் குறைவாக இருக்காமல் பார்த்துக் கொண்டு, அதிக ஒலி மற்றும் வெளிச்சத்தை தவிர்க்க வேண்டும். நேரத்திற்கு உணவு எடுத்துக்கொண்டு, மன அழுத்தத்தை குறைக்க முயற்சி செய்யுங்கள். தலைவலி அதிகமாக இருந்தால் மருத்துவர் கூறிய மருந்துகளை எடுத்துக்கொண்டு, அடிக்கடி தாக்கம் வந்தால் மருத்துவரை அணுகுவது அவசியம்."
    },
    "Common Cold": {
        "advice": ["Rest", "Warm fluids"],
        "tamil": "சாதாரண சளி (Common cold) ஏற்பட்டால் அதிகமாக ஓய்வு எடுத்து, சூடான நீர் மற்றும் சூப் போன்றவற்றை குடிப்பது உதவும். கைகளை அடிக்கடி கழுவி சுத்தமாக வைத்துக் கொள்ளுங்கள். இருமல், மூக்கோட்டம் நீடித்தால் அல்லது காய்ச்சல் அதிகமாக இருந்தால் மருத்துவரை அணுகுவது நல்லது."
    },
    "Anxiety": {
        "advice": ["Relax", "Meditation"],
        "tamil": "பதட்டம் (Anxiety) இருந்தால் ஆழ்ந்த சுவாச பயிற்சிகள் செய்து மனதை அமைதியாக வைத்துக் கொள்ள முயற்சி செய்யுங்கள். ஒழுங்கான தூக்கம் மற்றும் தினசரி உடற்பயிற்சி மனநிலையை மேம்படுத்த உதவும். நம்பிக்கையுள்ளவர்களுடன் பேசுவது பயனுள்ளதாக இருக்கும். பதட்டம் நீண்ட நாட்கள் தொடர்ந்தால் மருத்துவர் அல்லது மனநல நிபுணரை அணுகுவது முக்கியம்."
    },
    "Chronic Kidney Disease": {
        "advice": ["Monitor kidney", "Diet control"],
        "tamil": "நீண்டகால சிறுநீரக நோய் (Chronic Kidney Disease) உள்ளவர்கள் உப்பு மற்றும் புரதம் அளவை கட்டுப்படுத்திய உணவு முறையை பின்பற்ற வேண்டும். மருத்துவர் கூறிய மருந்துகளை நேரத்திற்கு எடுத்துக்கொண்டு, ரத்த அழுத்தம் மற்றும் சர்க்கரை அளவை கட்டுப்பாட்டில் வைத்திருக்க வேண்டும். போதுமான தண்ணீர் குடிப்பது மற்றும் அடிக்கடி பரிசோதனை செய்வதும் மிகவும் முக்கியம்."
    },
    "Ulcer": {
        "advice": ["Avoid spicy food"],
        "tamil": "புண் (Ulcer) இருந்தால் காரம், புளிப்பு மற்றும் எண்ணெய் அதிகமான உணவுகளை தவிர்க்க வேண்டும். நேரத்திற்கு உணவு எடுத்துக்கொண்டு, காலியான வயிற்றில் நீண்ட நேரம் இருக்காமல் பார்த்துக் கொள்ளுங்கள். மன அழுத்தத்தை குறைப்பதும் உதவும். வயிற்று வலி அல்லது எரிச்சல் நீடித்தால் மருத்துவரை அணுகி சரியான சிகிச்சை பெறுவது அவசியம்." 
    },

    "Common Cold": {
        "advice": ["Rest", "Warm fluids"],
        "english": "Take rest and drink warm fluids. Maintain hygiene to avoid spread.",
        "tamil": "சாதாரண சளிக்கு ஓய்வு மற்றும் சூடான நீர் உதவும்."
    },
    
    "Hypertension": {
        "advice": ["Reduce salt", "Exercise"],
        "tamil": "உயர் இரத்த அழுத்தம் (Hypertension) உள்ளவர்கள் உப்பு அளவை குறைத்து, சீரான மற்றும் ஆரோக்கியமான உணவு முறையை பின்பற்ற வேண்டும். தினமும் உடற்பயிற்சி செய்து, மன அழுத்தத்தை கட்டுப்படுத்துவது முக்கியம். புகைபிடித்தல் மற்றும் மதுபானத்தை தவிர்க்கவும். ரத்த அழுத்தத்தை அடிக்கடி பரிசோதித்து, மருத்துவர் கூறிய மருந்துகளை நேரத்திற்கு எடுத்துக்கொள்ள வேண்டும்."
    },
}

# =========================
# INPUT UI
# =========================
st.subheader("🩺 Select Symptoms")
selected_symptoms = st.multiselect("Choose symptoms", all_symptoms)

st.subheader("📋 Condition Details")
age = st.number_input("Age", 1, 100)
days = st.number_input("How many days?", 0, 30)

fever_level = st.selectbox("Fever level?", ["No fever", "Mild", "High"])
pain = st.selectbox("Pain level?", ["Low", "Medium", "High"])
progress = st.selectbox("Condition?", ["Same", "Better", "Worse"])

# =========================
# PREDICTION
# =========================
if st.button("Predict"):

    scores = {}

    for disease, symptoms in disease_rules.items():

        match = set(selected_symptoms) & set(symptoms)
        match_count = len(match)

        base_score = match_count / len(symptoms)
        score = base_score * 100

        # MINIMUM MATCH RULE
        if match_count < 2:
            score *= 0.3

        # STRONG SYMPTOM RULE
        if disease in strong_rules:
            if not any(sym in selected_symptoms for sym in strong_rules[disease]):
                score *= 0.2

        # 🔴 AGE HARD BLOCK
        if disease in age_restrictions:
            if age < age_restrictions[disease]:
                score = 0

        # 🟢 YOUNG AGE BOOST
        if age < 18:
            if disease in ["Asthma", "Pneumonia", "Anxiety"]:
                score += 20

        # EXTRA BOOSTS
        if fever_level == "High" and "fever" in symptoms:
            score += 10

        if pain == "High":
            score += 5

        if progress == "Worse":
            score += 5

        scores[disease] = min(score, 100)

    # SORT + NORMALIZE
    sorted_diseases = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    top3 = sorted_diseases[:3]

    total = sum([s for _, s in top3]) or 1

    normalized_top3 = [(d, (s/total)*100) for d, s in top3]

    best_disease, confidence = normalized_top3[0]

    # 🚨 EMERGENCY ALERT
    if "chest pain" in selected_symptoms or "breathing difficulty" in selected_symptoms:
        st.error("🚨 Please consult doctor immediately")

    # =========================
    # OUTPUT
    # =========================
    st.subheader("🔝 Top 3 Predictions")
    for d, s in normalized_top3:
        st.write(f"{d} → {round(s,2)}%")

    st.subheader("🔍 Final Result")
    st.success(f"🩺 {best_disease}")
    st.write("Confidence:", round(confidence,2), "%")

    # SEVERITY
    if confidence < 40:
        st.markdown("### 🟢 Mild")
    elif confidence < 70:
        st.markdown("### 🟠 Moderate")
    else:
        st.markdown("### 🔴 Severe")

    # WHY
    st.subheader("🤖 Why this disease?")
    matched = list(set(selected_symptoms) & set(disease_rules[best_disease]))
    st.write("Matched symptoms:", matched)

    # ADVICE
    st.subheader("💊 Advice")
    info = disease_info.get(best_disease)

    if info:
        for tip in info["advice"]:
            st.write("•", tip)
    else:
        st.write("• Please consult doctor")

    # TAMIL
    st.subheader("🧠 தமிழ் விளக்கம்")
    if info:
        st.write(info["tamil"])
    else:
        st.write("மருத்துவரை அணுகவும்.")
