import streamlit as st
import numpy as np
import pickle as pkl


# Load label encoders
le_ST_Slope = pkl.load(open('ST_Slope_le.pkl','rb'))
le_ChestPainType = pkl.load(open('ChestPainType_le.pkl','rb'))
le_RestingECG = pkl.load(open('RestingECG_le.pkl','rb'))

# Load the scaler and models
scaler = pkl.load(open('scaler.pkl', 'rb'))
models = {
    'Decision Tree': pkl.load(open('Decision_Tree.pkl', 'rb')),
    'Logistic Regression': pkl.load(open('Logistic_Regression.pkl', 'rb')),
    'SVC':pkl.load(open('svc.pkl','rb')),
    'KNN': pkl.load(open('KNN.pkl', 'rb')),
    'Naive Bayes': pkl.load(open('Naive_bayes.pkl', 'rb')),
    'Random Forest': pkl.load(open('Random_forest.pkl', 'rb')),
    'Gradient Boosting': pkl.load(open('Gradient_boosting.pkl', 'rb')),
    'XGBoost' : pkl.load(open('xgboost.pkl','rb')),
    'AdaBoost': pkl.load(open('AdaBoost.pkl', 'rb')),
    'Stacking' : pkl.load(open('Stacking.pkl','rb'))
}

# Streamlit app
st.title('Heart Disease Prediction ❤️')
st.markdown('---')

# Input fields
c1, c2, c3 = st.columns(3)
with c1:
    Age = st.number_input("Age:", min_value=0, max_value=100)
    Sex = st.selectbox("Gender:", ['Male', 'Female'])
    Sex = 1 if Sex == 'Male' else 0
    ChestPainType = st.selectbox("Chest Pain Type:", ['ATA', 'NAP', 'ASY', 'TA'])
    RestingBP = st.number_input("RestingBP:", min_value=0)

with c2:
    Cholesterol = st.number_input("Cholesterol:", min_value=0)
    FastingBS = st.number_input("FastingBS:", min_value=0)
    FastingBS = 1 if FastingBS > 120 else 0
    RestingECG = st.selectbox("RestingECG:", ['Normal', 'ST', 'LVH'])

with c3:
    MaxHR = st.number_input("Maximum Heart Rate:", min_value=0)
    ExerciseAngina = st.selectbox("Exercise Angina:", ['Yes', 'No'])
    ExerciseAngina = 1 if ExerciseAngina == 'Yes' else 0
    Oldpeak = st.number_input("Oldpeak:")
    ST_Slope = st.selectbox("ST_Slope:", ['Up', 'Flat', 'Down'])

# Transform categorical inputs
ChestPainType_encoded = le_ChestPainType.transform([ChestPainType])[0]
RestingECG_encoded = le_RestingECG.transform([RestingECG])[0]
ST_Slope_encoded = le_ST_Slope.transform([ST_Slope])[0]

# Create feature array
features = np.array([Age, Sex, ChestPainType_encoded, RestingBP, Cholesterol,
                     FastingBS, RestingECG_encoded, MaxHR, ExerciseAngina, Oldpeak, ST_Slope_encoded]).reshape(1, -1)

# Scale features
features = scaler.transform(features)

# Model selection
model_name = st.sidebar.selectbox("Select model", models.keys())


# st.info("""
#         Note ✍️:\n
#         Best Recall Model: (xgboost) With Recall: 95.54%\n
#         Best Accuracy Model: (KNN) With accuracy: 88.00%\n
#         Best F1_Score Model: (KNN) With F1_Score: 89.72%
# """)

# Button for submit
submit = st.sidebar.button("Predict", use_container_width=True)


# Prediction one model
if submit:
    try:
        model = models[model_name]
        prediction = model.predict(features)[0]
        if prediction == 0:
            st.sidebar.success("No Heart Disease :thumbsup:")
        else:
            st.sidebar.error("May have heart disease :thumbsdown:")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")


# prediction all model and take mode
Mode = st.button("Most frequent result for 10 models", use_container_width=True)
if Mode:
   vals=[]
   for k,v in models.items():
       vals.append(v.predict(features)[0])
   

   sum=sum(vals)
   if sum > 5 : # 1
      st.error(f"May have heart disease :thumbsdown:")
   else: # 0
      st.success(f"No Heart Disease :thumbsup:")
