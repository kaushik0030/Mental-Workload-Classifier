import pandas as pd
import joblib

# Load the model, scaler, and feature columns
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Load the combined dataset with workload labels
df = pd.read_csv('Hb_HbO_with_workload.csv')

# Function to preprocess input data based on participant and condition
def get_features_for_prediction(participant, condition, df):
    # Filter the dataset based on participant and condition
    filtered_data = df[(df['Participant'] == participant) & (df['Condition'] == condition)]
    
    if filtered_data.empty:
        raise ValueError("No data found for the given participant and condition")
    
    # Select only the feature columns used in training
    features = filtered_data[feature_columns]
    
    # Standardize the features
    features_scaled = scaler.transform(features)
    
    return features_scaled

# Function to predict mental workload
def predict_workload(participant, condition):
    try:
        features = get_features_for_prediction(participant, condition, df)
        prediction = model.predict(features)
        
        # Assuming Workload mapping is same as used in training
        workload_mapping = {0: 'Low', 1: 'Mid', 2: 'High'}
        predicted_workload = [workload_mapping[pred] for pred in prediction]
        
        return predicted_workload
    except ValueError as e:
        return str(e)

# Example usage
if __name__ == "__main__":
    participant = 'Participant 1'  # Replace with actual participant ID
    condition = 'Neutral'  # Replace with actual condition

    predicted_workload = predict_workload(participant, condition)
    print(f'Predicted Workload for {participant} under {condition} condition: {predicted_workload}')
