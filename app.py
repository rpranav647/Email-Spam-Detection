# import streamlit as st
# import joblib
# import os

# # Load the CountVectorizer
# vectorizer = joblib.load("vectorizer.pkl")

# # Load all models
# model_paths = {
#     "Logistic Regression": "saved_models/logistic_regression_model.pkl",
#     "K-Nearest Neighbors": "saved_models/k-nearest_neighbors_model.pkl",
#     "Support Vector Machine": "saved_models/support_vector_machine_model.pkl",
#     "Random Forest": "saved_models/random_forest_model.pkl",
# }

# models = {name: joblib.load(path) for name, path in model_paths.items()}

# # Streamlit App
# st.title("📨 Spam Message Classifier")

# st.write("This app predicts whether a message is **Spam** or **Ham (Not Spam)** using different ML models.")

# # User input
# message = st.text_area("Enter your message here:")

# model_choice = st.selectbox("Choose a model:", list(models.keys()))

# if st.button("Predict"):
#     if message.strip() == "":
#         st.warning("Please enter a message to classify.")
#     else:
#         # Transform the input message
#         vect_msg = vectorizer.transform([message]).toarray()

#         # Predict
#         prediction = models[model_choice].predict(vect_msg)[0]
#         label = "📩 Ham (Not Spam)" if prediction == 0 else "🚫 Spam"

#         st.markdown(f"### Prediction: {label}")


import streamlit as st
import joblib

# Title
st.title("📩 Spam Message Classifier")
st.write("Enter a message and choose a model + vectorizer to predict if it's **Spam** or **Ham**.")

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


    except FileNotFoundError:
        st.error("Model or vectorizer not found. Please ensure all `.pkl` files are in the same directory.")

