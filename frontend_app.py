import streamlit as st
import requests
from PIL import Image
import time
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()  # loading all the environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# function to load Gemini Pro model and get responses
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])


def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response


# Backend URL
backend_url = 'http://127.0.0.1:5000'


def fetch_options(endpoint):
    try:
        response = requests.get(f'{backend_url}/{endpoint}')
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f'Failed to fetch {endpoint}: {e}')
        return []


def main():
    st.set_page_config(page_title="Mental Workload Predictor", layout="centered")

    # Display welcome image
    welcome_image = Image.open("static/welcome.jpg")
    st.image(welcome_image, use_column_width=True)
    time.sleep(2)  # Shorter delay to make the app more responsive
    st.empty()  # Clear placeholder after the initial load

    # Display background image
    background_image = Image.open("static/background.jpg")
    st.image(background_image, use_column_width=True)

    st.title('Mental Workload Prediction')

    # Fetch participants and conditions from the backend
    participants = fetch_options('participants')
    conditions = fetch_options('conditions')

    # Check if options are available
    if not participants:
        st.error('No participants available. Please check the backend.')
        return

    if not conditions:
        st.error('No conditions available. Please check the backend.')
        return

    # Create selectboxes for user input
    participant = st.selectbox('Participant ID', participants)
    condition = st.selectbox('Condition', conditions)

    if st.button('Predict Workload'):
        if participant and condition:
            payload = {'participant': participant, 'condition': condition}
            try:
                response = requests.post(f'{backend_url}/predict', json=payload)
                response.raise_for_status()
                prediction = response.json()
                st.success(f'Predicted Workload: {prediction}')
            except requests.RequestException as e:
                st.error(f'Error in prediction: {e}. Please try again.')
        else:
            st.warning('Please provide both Participant ID and Condition.')

    st.header("Gemini Chatbot")

    # Initialize session state for chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    input = st.text_input("Input: ", key="input")
    submit = st.button("Ask the question")

    if submit and input:
        response = get_gemini_response(input)
        # Add user query and response to session state chat history
        st.session_state['chat_history'].append(("You", input))
        st.subheader("The Response is")
        for chunk in response:
            st.write(chunk.text)
            st.session_state['chat_history'].append(("Bot", chunk.text))
    st.subheader("The Chat History is")

    for role, text in st.session_state['chat_history']:
        st.write(f"{role}: {text}")


if __name__ == '__main__':
    main()
