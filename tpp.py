import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Student Placement Analytics", layout="wide")

st.title("🎓 Student Performance & Placement Analytics System")

# Upload Dataset
uploaded_file = st.file_uploader("Upload Student Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Information")
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    # Fill missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Encode categorical columns
    le = LabelEncoder()

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    st.subheader("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x='CGPA')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df, x='Attendance')
        st.plotly_chart(fig, use_container_width=True)

    if 'Placement_Status' in df.columns:

        placement_fig = px.pie(
            names=["Placed", "Not Placed"],
            values=[
                sum(df['Placement_Status'] == 1),
                sum(df['Placement_Status'] == 0)
            ],
            title="Placement Distribution"
        )

        st.plotly_chart(placement_fig, use_container_width=True)

        st.subheader("🤖 Machine Learning Model")

        X = df.drop('Placement_Status', axis=1)
        y = df['Placement_Status']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        st.success(f"Model Accuracy: {accuracy*100:.2f}%")

        st.subheader("📈 Feature Importance")

        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        fig = px.bar(
            importance,
            x='Importance',
            y='Feature',
            orientation='h'
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🎯 Placement Predictor")

        user_input = {}

        for col in X.columns:
            value = st.number_input(
                f"Enter {col}",
                value=float(X[col].mean())
            )
            user_input[col] = value

        if st.button("Predict Placement"):

            input_df = pd.DataFrame([user_input])

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            if prediction == 1:
                st.success(
                    f"Likely to be Placed ✅\n\nPlacement Probability: {probability*100:.2f}%"
                )
            else:
                st.error(
                    f"Placement Risk ⚠️\n\nPlacement Probability: {probability*100:.2f}%"
                )

            st.subheader("💡 Recommendations")

            if user_input.get("CGPA", 0) < 7:
                st.write("- Improve academic performance.")

            if user_input.get("Attendance", 0) < 75:
                st.write("- Increase attendance.")

            if user_input.get("Coding_Score", 0) < 60:
                st.write("- Practice DSA and coding regularly.")

            if user_input.get("Communication_Score", 0) < 60:
                st.write("- Improve communication and interview skills.")

else:
    st.info("Please upload a placement dataset CSV file.")