import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import plotly.express as px

st.set_page_config(page_title="Heart Risk Prediction", layout="wide")

# -------- CSS Styling --------
st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#e8f5e9,#c8e6c9);
}

h1{
color:#1B5E3A;
font-weight:bold;
text-align:center;
}

/* Predict Button */
.stButton>button{
background-color:#1B5E3A;
color:white;
border-radius:10px;
padding:10px 18px;
font-size:16px;
font-weight:bold;
border:none;
}

/* Hover Effect */
.stButton>button:hover{
background-color:#145A32;
color:white;
}

button[data-baseweb="tab"]{
font-size:16px;
font-weight:bold;
color:#1b4332;
}

/* Input labels */
label{
color:#1b4332 !important;
font-weight:600;
}

/* Sub headings */
h2,h3{
color:#1B5E3A;
}

/* Tabs container */
div[data-baseweb="tab-list"]{
background-color:#e8f5e9;
padding:6px;
border-radius:12px;
}

/* Normal tab */
button[data-baseweb="tab"]{
background-color:#c8e6c9;
color:#1B5E3A;
font-weight:600;
border-radius:10px;
margin-right:6px;
padding:8px 16px;
transition:0.3s;
}

/* Hover effect */
button[data-baseweb="tab"]:hover{
background-color:#a5d6a7;
color:#0d3d28;
}

/* Active tab */
button[data-baseweb="tab"][aria-selected="true"]{
background-color:#1B5E3A;
color:white;
font-weight:bold;
}

     
div[data-testid="stHorizontalBlock"]{
justify-content:center;
max-width:500px;
margin-left:auto;
margin-right:auto;
max-width: 500px
}

            
/* Number input field */
input[type="number"]{
background-color:#ffffff !important;
color:#1B5E3A !important;
border:2px solid #a5d6a7 !important;
border-radius:8px !important;
padding:6px !important;
}

/* Selectbox */
div[data-baseweb="select"] > div{
background-color:#ffffff !important;
color:#1B5E3A !important;
border:2px solid #a5d6a7 !important;
border-radius:8px !important;
}

/* Dropdown text */
div[data-baseweb="select"] span{
color:#1B5E3A !important;
}


/* Hover effect */
input[type="number"]:hover,
div[data-baseweb="select"] > div:hover{
border-color:#1B5E3A !important;
}
            


.result-good{
background:#c8e6c9;
padding:15px;
border-radius:10px;
color:#1B5E3A;
text-align:center;
font-size:20px;
}

.result-bad{
background:#ffcdd2;
padding:15px;
border-radius:10px;
color:#b71c1c;
text-align:center;
font-size:20px;
}

</style>
""",unsafe_allow_html=True)

# -------- Download CSV Function --------
def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Download Predictions CSV</a>'
    return href


# -------- Title --------
st.markdown(
"<h1 style='color:#1B5E3A; text-align:center;'>🫀 Heart Disease Risk Prediction</h1>",
unsafe_allow_html=True
)

tab1,tab2,tab3 = st.tabs(['Predict','Bulk Predict','Model Information'])

# ---------------- TAB 1 ----------------
with tab1:

    col1,col2 = st.columns(2)

    with col1:

        age = st.number_input("Age(years)", min_value=0,max_value=150)

        sex = st.selectbox("Sex",["Male","Female","Other"])

        chest_pain = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomtic"]
        )

        resting_bp = st.number_input("Resting Blood Pressure(mm Hg)",min_value=0,max_value=300)

        cholesterol = st.number_input("Serum Cholesterol(mm/dl)",min_value=0)

    with col2:

        fasting_bs = st.selectbox(
        "Fasting Blood Sugar",
        ["<= 120 mg/dl","> 120 mg/dl"]
        )

        resting_ecg = st.selectbox(
        "Resting ECG Results",
        ["Normal","ST-T Wave Abnormality","Left Ventricular Hypertrophy"]
        )

        max_hr = st.number_input("Maximum Heart Rate Achieved",min_value=60,max_value=202)

        exercise_angian = st.selectbox(
        "Exercise-Induced Angina",
        ["Yes","No"]
        )

        oldpeak = st.number_input("Oldpeak(ST Depression)",min_value=0.0,max_value=10.0)

        st_slope = st.selectbox(
        "Slope Of Peak Exercise ST Segment",
        ["Upsloping","Flat","Downsloping"]
        )

    # -------- Data Conversion --------
    sex = 0 if sex == "Male" else 1
    chest_pain = ["Atypical Angina","Non-Anginal Pain","Asymptomtic","Typical Angina"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg = ["Normal","ST-T Wave Abnormality","Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angian = 1 if exercise_angian == "Yes" else 0
    st_slope = ["Upsloping","Flat","Downsloping"].index(st_slope)

    input_data = pd.DataFrame({
        'Age':[age],
        'Sex':[sex],
        'ChestPainType':[chest_pain],
        'RestingBP':[resting_bp],
        'Cholesterol':[cholesterol],
        'FastingBS':[fasting_bs],
        'RestingECG':[resting_ecg],
        'MaxHR':[max_hr],
        'ExerciseAngina':[exercise_angian],
        'Oldpeak':[oldpeak],
        'ST_Slope':[st_slope]
    })

    algonames = [
    'Decision Trees',
    'Logistic Regression',
    'Random Forest',
    'Support Vector Machine'
    ]

    modelnames = [
    'tree.pkl',
    'LogisticRegression.pkl',
    'RandomForest.pkl',
    'svm.pkl'
    ]

    predictions = []

    def predict_Heart_Risk(data):

        for modelname in modelnames:

            model = pickle.load(open(modelname,'rb'))

            prediction = model.predict(data)

            predictions.append(prediction)

        return predictions

    # -------- Prediction --------
    if st.button("🔍 Predict Heart Risk"):

        st.markdown(
        "<h3 style='color:#0d6b43; font-weight:bold;'>🔎 Prediction Results</h3>",
        unsafe_allow_html=True
        )
        result = predict_Heart_Risk(input_data)

        for i in range(len(predictions)):

            st.markdown(f"<h3 style='color:black'>{algonames[i]}</h3>", unsafe_allow_html=True)
            
            if result[i][0] == 0:

                st.markdown(
                "<div class='result-good'>✅ No Heart Risk Detected</div>",
                unsafe_allow_html=True
                )

            else:

                st.markdown(
                "<div class='result-bad'>⚠️ Heart Risk Detected</div>",
                unsafe_allow_html=True
                )

# ---------------- TAB 2 ----------------
with tab2:
    
    st.markdown(
    "<h2 style='color:#1B5E3A;'>📂 Upload CSV File</h2>",
    unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Upload CSV",type=["csv"])

    if uploaded_file is not None:

        input_data = pd.read_csv(uploaded_file)

        model = pickle.load(open('LogisticRegression.pkl','rb'))

        excepted_columns = [
        'Age','Sex','ChestPainType','RestingBP','Cholesterol',
        'FastingBS','RestingECG','MaxHR','ExerciseAngina',
        'Oldpeak','ST_Slope'
        ]

        if set(excepted_columns).issubset(input_data.columns):

            input_data['Prediction LR'] = ''

            for i in range(len(input_data)):

                arr = input_data.iloc[i,:-1].values

                input_data['Prediction LR'][i] = model.predict([arr])[0]

            st.markdown(
            "<h3 style='color:#1B5E3A;'>📊 Predictions</h3>",
            unsafe_allow_html=True
            )

            st.write(input_data)

            st.markdown(
            get_binary_file_downloader_html(input_data),
            unsafe_allow_html=True
            )

        else:

            st.warning("Uploaded CSV columns incorrect.")

    else:

        st.info("Upload CSV to get predictions.")

# ---------------- TAB 3 ----------------
with tab3:

    st.markdown("<h2 style='color:#1B5E3A;'>📊 Heart Disease Data Dashboard</h2>", unsafe_allow_html=True)

    heart_df = pd.read_csv("heart.csv")

    # ---- Model Accuracy ----
    data = {
    'Decision Trees':80.97,
    'Logistic Regression':85.86,
    'Random Forest':84.23,
    'Support Vector Machine':84.22,
    'GridRF':89.75
    }

    df = pd.DataFrame(list(data.items()),columns=['Model','Accuracy'])

    fig1 = px.bar(
        df,
        x='Model',
        y='Accuracy',
        text='Accuracy',
        color='Accuracy',
        color_continuous_scale='Greens',
        title="Model Accuracy Comparison"
    )

    st.plotly_chart(fig1,use_container_width=True)

    