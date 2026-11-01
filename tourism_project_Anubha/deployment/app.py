
import os
import streamlit as st
import pandas as pd
import joblib

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Tourism Product Prediction",
    page_icon="✈️",
    layout="wide"
)

# =====================================================
# 2. LOAD TRAINED MODEL
# =====================================================

model_path = os.path.join(
    os.path.dirname(__file__),
    "best_tourism_product_model.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

# =====================================================
# 3. APPLICATION INTRODUCTION
# =====================================================

st.title("✈️ Tourism Product Purchase Prediction App")

st.write("""
This application predicts the likelihood of a customer
purchasing a tourism package based on customer
demographics, preferences and interaction details.

Enter the customer information below to obtain a prediction.
""")

st.success("Trained XGBoost model loaded successfully.")

# =====================================================
# 4. COLLECT CUSTOMER INPUTS
# =====================================================

st.header("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Customer Age", 18, 100, 35
    )

    city_tier = st.selectbox(
        "City Tier", [1, 2, 3]
    )

    duration = st.number_input(
        "Duration of Pitch",
        min_value=0.0,
        value=10.0
    )

    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business", "Large Business"]
    )

    gender = st.selectbox(
        "Gender", ["Male", "Female"]
    )

with col2:

    contact = st.selectbox(
        "Type of Contact",
        ["Self Enquiry", "Company Invited"]
    )

    persons = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        value=3
    )

    followups = st.number_input(
        "Number of Follow-ups",
        min_value=0,
        value=3
    )

    product = st.selectbox(
        "Product Pitched",
        ["Basic", "Deluxe", "Standard",
         "Super Deluxe", "King"]
    )

    property_star = st.selectbox(
        "Preferred Property Star",
        [3, 4, 5]
    )

with col3:

    marital = st.selectbox(
        "Marital Status",
        ["Married", "Single", "Divorced"]
    )

    trips = st.number_input(
        "Number of Trips",
        min_value=0,
        value=2
    )

    passport = st.selectbox(
        "Passport",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    satisfaction = st.selectbox(
        "Pitch Satisfaction Score",
        [1, 2, 3, 4, 5]
    )

    own_car = st.selectbox(
        "Own Car",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

children = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    value=1
)

income = st.number_input(
    "Monthly Income",
    min_value=0.0,
    value=25000.0
)

designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

# =====================================================
# 5. CREATE INPUT DATAFRAME
# =====================================================

input_data = pd.DataFrame([{

    "Age": age,
    "TypeofContact": contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": persons,
    "NumberOfFollowups": followups,
    "ProductPitched": product,
    "PreferredPropertyStar": property_star,
    "MaritalStatus": marital,
    "NumberOfTrips": trips,
    "Passport": passport,
    "PitchSatisfactionScore": satisfaction,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": children,
    "Designation": designation,
    "MonthlyIncome": income

}])

# =====================================================
# 6. MAKE PREDICTION
# =====================================================

if st.button("Predict Tourism Product Purchase"):

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    positive_index = list(model.classes_).index(1)

    purchase_probability = probabilities[positive_index]

    classification_threshold = 0.45

    st.subheader("Prediction Result")

    st.metric(
        "Purchase Probability",
        f"{purchase_probability * 100:.2f}%"
    )

    if purchase_probability >= classification_threshold:

        st.success(
            "Customer is likely to purchase the tourism package!"
        )

    else:

        st.warning(
            "Customer is not likely to purchase the tourism package."
        )

    st.write("Predicted Class:", int(prediction))

    st.progress(float(purchase_probability))

    st.caption(
        "Prediction generated using the trained XGBoost pipeline."
    )
