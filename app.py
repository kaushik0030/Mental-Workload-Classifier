from flask import Flask, request, jsonify
import pandas as pd
import joblib
import openai

app = Flask(__name__)

# Load the model, scaler, and feature columns
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Load the combined dataset with workload labels
df = pd.read_csv('Hb_HbO_with_workload.csv')

# OpenAI API key
openai.api_key = 'your_openai_api_key'

@app.route('/participants', methods=['GET'])
def get_participants():
    participants = df['Participant'].unique().tolist()
    return jsonify(participants)

@app.route('/conditions', methods=['GET'])
def get_conditions():
    conditions = df['Condition'].unique().tolist()
    return jsonify(conditions)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    participant = data['participant']
    condition = data['condition']
    
    try:
        features = get_features_for_prediction(participant, condition, df)
        prediction = model.predict(features)
        
        workload_mapping = {0: 'Low', 1: 'Mid', 2: 'High'}
        predicted_workload = [workload_mapping[pred] for pred in prediction]
        
        return jsonify(predicted_workload)
    except ValueError as e:
        return jsonify(str(e)), 400

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    
    if not message:
        return jsonify({'response': 'No message provided'}), 400
    
    try:
        response = openai.Completion.create(
            engine="davinci",
            prompt=message,
            max_tokens=150
        )
        return jsonify({'response': response.choices[0].text.strip()})
    except Exception as e:
        return jsonify({'response': str(e)}), 500

def get_features_for_prediction(participant, condition, df):
    filtered_data = df[(df['Participant'] == participant) & (df['Condition'] == condition)]
    
    if filtered_data.empty:
        raise ValueError("No data found for the given participant and condition")
    
    features = filtered_data[feature_columns]
    features_scaled = scaler.transform(features)
    
    return features_scaled

if __name__ == '__main__':
    app.run(debug=True)
