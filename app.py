# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set the color scheme to dark
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown(
    """
    <style>
    body {
        background-color: #1e1e1e; /* Dark background */
        color: white; /* White text */
    }
    .login-form {
        background-color: rgba(50, 50, 50, 0.9); /* Slightly lighter dark background for the form */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
    }
    .stButton>button {
        background-color: #007bff; /* Button color */
        color: white; /* Button text color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Login Functionality
def login():
    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<div class="login-form">', unsafe_allow_html=True)
        st.sidebar.title("Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type='password')
        
        if st.sidebar.button("Login"):
            if username == "user" and password == "password":  # Replace with your logic
                st.session_state['logged_in'] = True
                st.success("Logged in successfully!")
            else:
                st.error("Invalid username or password")
        st.markdown('</div></div>', unsafe_allow_html=True)

# Check if user is logged in
if 'logged_in' not in st.session_state:
    login()

if st.session_state.get('logged_in', False):
    # Load dataset
    df = pd.read_csv(r'C:\Users\Dell\OneDrive\Desktop\coding\diabetes predictor\diabetes.csv')

    # HEADINGS
    st.title('Diabetes Predictor')
    st.subheader('Training Data Overview')
    st.write(df.describe())

    # Visualization
    st.subheader('Data Visualization')
    st.bar_chart(df)

    # X AND Y DATA
    x = df.drop(['Outcome'], axis=1)
    y = df['Outcome']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Feature Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # FUNCTION FOR USER INPUT
    def user_report():
        pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
        glucose = st.sidebar.slider('Glucose', 0, 200, 120)
        bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
        skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
        insulin = st.sidebar.slider('Insulin', 0, 846, 79)
        bmi = st.sidebar.slider('BMI', 0, 67, 20)
        dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
        age = st.sidebar.slider('Age', 21, 88, 33)

        user_report = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': bp,
            'SkinThickness': skinthickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        report_data = pd.DataFrame(user_report, index=[0])
        return report_data

    # PATIENT DATA
    user_data = user_report()

    # Model Selection
    model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "Logistic Regression"])

    # Train the model based on user choice
    if model_choice == "Random Forest":
        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)
        model = rf
    else:
        lr = LogisticRegression(max_iter=200)
        lr.fit(x_train, y_train)
        model = lr

    # OUTPUT
    st.subheader('Model Accuracy: ')
    accuracy = accuracy_score(y_test, model.predict(x_test))
    st.write(f"Accuracy: {accuracy * 100:.2f}%")

    # Confusion Matrix
    st.subheader('Confusion Matrix')
    conf_matrix = confusion_matrix(y_test, model.predict(x_test))
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    # Classification Report
    st.subheader('Classification Report')
    report = classification_report(y_test, model.predict(x_test), output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write(report_df)

    # Feature Importance
    if model_choice == "Random Forest":
        st.subheader('Feature Importance')
        importance = model.feature_importances_
        feature_names = x.columns
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        fig = plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        st.pyplot(fig)

    # Prediction
    user_result = model.predict(user_data)
    output = 'You are healthy' if user_result[0] == 0 else 'You are Diabetic'
    st.title(output)

    # Personalized Recommendations
    if user_result[0] == 1:
        st.subheader('Recommendations:')
        st.write("""
        - Consult a healthcare provider for further testing.
        - Consider lifestyle changes such as a balanced diet and regular exercise.
        - Monitor your blood sugar levels regularly.
        - Drink plenty of water throughout the day and limit sugary drinks.
        - Aim for at least 150 minutes of moderate aerobic activity each week, such as walking, cycling, or swimming.
          Include strength training exercises at least twice a week.
        """)
    else:
        st.subheader('Keep up the good work!')
        st.write("Continue maintaining a healthy lifestyle.")

    # Downloadable Report
    output_csv = pd.DataFrame({
        'Feature': user_data.columns,
        'Value': user_data.values.flatten(),
        'Prediction': output
    })

    # Convert DataFrame to CSV
    csv = output_csv.to_csv(index=False)

    # Download Button
    st.download_button(
        label="Download Report",
        data=csv,
        file_name='health_analysis_report.csv',
        mime='text/csv'
    )
else:
    st.warning("Please log in to access the Health Analysis Tool.")
