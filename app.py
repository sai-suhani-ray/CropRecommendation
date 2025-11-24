import streamlit as st
import pandas as pd
import joblib

st.title("🌾 Crop Recommendation System")
import streamlit as st

# ---- DESCRIPTION ----
st.markdown("""
Welcome to the Crop Recommendation System!  
यह वेबसाइट किसानों को उनके मिट्टी के पोषक तत्व और मौसम की स्थिति के आधार पर सबसे अच्छी फसल चुनने में मदद करती है।
फसल की सिफारिश पाने के लिए, बस अपने खेत के लिए आवश्यक जानकारी नीचे दर्ज करें।

आपको ये जानकारी देनी होगी:

N (नाइट्रोजन)

P (फॉस्फोरस)

K (पोटेशियम)

तापमान (Temperature)

मिट्टी का pH

वर्षा / नमी (Rainfall / Humidity)

हर पैरामीटर के पास ℹ️ बटन पर क्लिक करें, ताकि आप जान सकें कि इसका क्या मतलब है और आपकी फसल पर इसका क्या असर होता है।  
""")

st.header("🌱 Parameter Info (पैरामीटर जानकारी)")

# Nitrogen
with st.expander("ℹ️ N (Nitrogen)"):
    st.write("""
Nitrogen is a key nutrient that helps plants grow healthy leaves and stems.  
नाइट्रोजन पौधों के पत्तों और तनों को स्वस्थ रूप से बढ़ने में मदद करता है।
""")

# Phosphorus
with st.expander("ℹ️ P (Phosphorus)"):
    st.write("""
Phosphorus is essential for root development and flowering of plants.  
फॉस्फोरस जड़ों के विकास और फूलने में आवश्यक है।
""")

# Potassium
with st.expander("ℹ️ K (Potassium)"):
    st.write("""
Potassium helps in overall plant health and improves resistance to diseases.  
पोटेशियम पौधों की समग्र सेहत में मदद करता है और रोगों के प्रति प्रतिरोध बढ़ाता है।
""")

# Temperature
with st.expander("ℹ️ Temperature"):
    st.write("""
Optimal temperature is crucial for crop growth and yield.  
फसल की वृद्धि और उपज के लिए आदर्श तापमान महत्वपूर्ण है।
""")

# pH
with st.expander("ℹ️ pH"):
    st.write("""
Soil pH indicates acidity or alkalinity, affecting nutrient availability.  
मिट्टी का pH अम्लीय या क्षारीय होने को दर्शाता है, जो पोषक तत्वों की उपलब्धता को प्रभावित करता है।
""")


model = joblib.load("crop_model.pkl")

st.header("Enter Soil & Climate Values")

N = st.number_input("Nitrogen (N)")
P = st.number_input("Phosphorus (P)")
K = st.number_input("Potassium (K)")
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("pH Value")
rainfall = st.number_input("Rainfall (mm)")

if st.button("Predict Crop"):
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                               columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
    result = model.predict(input_data)
    st.success(f"Recommended Crop: **{result[0]}**")
