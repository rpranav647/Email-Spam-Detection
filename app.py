import streamlit as st
import joblib

# Title
st.title("📩 Spam Message Classifier")
st.write("Enter a message and choose a model + vectorizer to predict if it's *Spam* or *Ham*.")

# Load available model names
model_names = [
    "logistic_regression", "knn", "svm",
    "random_forest", "naive_bayes", "decision_tree"
]
vectorizer_names = ["bow", "tfidf"]

# Select vectorizer and model
vec_choice = st.selectbox("Select Vectorizer", vectorizer_names)
model_choice = st.selectbox("Select Model", model_names)

# Text input
user_input = st.text_area("Enter your message here:")

if st.button("Predict"):
    try:
        # Load vectorizer
        vec_path = f"{vec_choice}_vectorizer.pkl"
        vectorizer = joblib.load(vec_path)

        # Load model
        model_path = f"{model_choice}_{vec_choice}.pkl"
        model = joblib.load(model_path)

        # Vectorize input
        input_vec = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(input_vec)[0]
        label = "🟢 Ham" if prediction == 0 else "🔴 Spam"

        st.subheader("Prediction:")
        st.success(label) if prediction == 0 else st.error(label)

    except FileNotFoundError:
        st.error("Model or vectorizer not found. Please ensure all .pkl files are in the same directory.")
