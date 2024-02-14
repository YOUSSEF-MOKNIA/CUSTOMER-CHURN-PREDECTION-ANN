from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from flask_cors import CORS
import pandas as pd



app = Flask(__name__)
CORS(app)
# Load the trained model
model = load_model('D:\churn_model.h5')

# Define columns to encode as binary (Yes/No)
columns_yes_no = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 
                  'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                  'PaperlessBilling']



def min_max_scale(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def preprocess_data(df):
    # Convert 'Yes' and 'No' values to 1s and 0s
    label_encoder = LabelEncoder()
    for column in columns_yes_no:
        df[column] = label_encoder.fit_transform(df[column])

    # Convert 'Gender' values to 1s and 0s
    df['gender'].replace({'Female': 1, 'Male': 0}, inplace=True)
    
    # Convert 'tenure' column to numeric
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Define the expected range for each feature
    tenure_min = 0
    tenure_max = 72  # Assuming maximum tenure is 72 months

    monthly_charges_min = 18  # Example minimum monthly charge
    monthly_charges_max = 118  # Example maximum monthly charge

    total_charges_min = 0  # Example minimum total charge
    total_charges_max = 8684  # Example maximum total charge

    # Scale the input values
    df['tenure'] = df['tenure'].apply(lambda x: min_max_scale(x, tenure_min, tenure_max))
    df['MonthlyCharges'] = df['MonthlyCharges'].apply(lambda x: min_max_scale(x, monthly_charges_min, monthly_charges_max))
    df['TotalCharges'] = df['TotalCharges'].apply(lambda x: min_max_scale(x, total_charges_min, total_charges_max))
    
    return df



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get feature data from request
    feature_data = request.json['features']
    
    # Convert feature data to DataFrame
    feature_df = pd.DataFrame.from_dict([feature_data])

    # Preprocess feature data
    preprocessed_df = preprocess_data(feature_df)
    print('tenure',preprocessed_df['tenure'])
    print(preprocessed_df['MonthlyCharges'])
    # Convert DataFrame to float types
    preprocessed_df = preprocessed_df.astype(float)
    
    # Make prediction using the loaded model
    prediction = model.predict(preprocessed_df)
    
    # Convert prediction to JSON response
    response = {'prediction': prediction[0].tolist()}
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)