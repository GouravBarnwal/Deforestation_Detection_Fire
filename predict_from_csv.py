import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

def load_model():
    try:
        return joblib.load("best_fire_detection_model.pkl")
    except FileNotFoundError:
        print("Error: Model file not found. Please ensure 'best_fire_detection_model.pkl' exists.")
        return None

def load_scaler():
    try:
        return joblib.load("scaler.pkl")
    except FileNotFoundError:
        print("Error: Scaler file not found. Please ensure 'scaler.pkl' exists.")
        return None

def map_fire_type(pred):
    """Map numeric prediction to fire type label"""
    fire_types = {
        0: "🌳 Vegetation Fire",
        1: "🔥 Industrial/Urban Fire",
        2: "🏭 Other Static Land Source",
        3: "🌊 Offshore Fire"
    }
    return fire_types.get(pred, "Unknown")

def predict_fire_type(row, model, scaler):
    """Predict fire type for a single row of data"""
    try:
        # Prepare input features in the correct order expected by the model:
        # ['brightness', 'scan', 'track', 'confidence', 'bright_t31', 'frp']
        X = np.array([[
            row['brightness'],
            row['scan'],
            row['track'],
            row['confidence'],
            row['bright_t31'],
            row['frp']
        ]])
        
        # Scale the features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        pred = model.predict(X_scaled)[0]
        
        # Special case for vegetation fire near coastlines
        if pred == 0 and (row['latitude'] <= 10.0 or row['longitude'] >= 92.0):
            pred = 3  # Change to Offshore Fire
            
        return map_fire_type(pred)
    except Exception as e:
        print(f"Error predicting for row: {e}")
        return "Prediction Error"

def process_csv(input_file, output_file, model, scaler, sample_size=None):
    """Process a CSV file and add predictions"""
    try:
        print(f"Processing {input_file}...")
        
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # If sample_size is provided, take a random sample
        if sample_size and len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)
            print(f"  Using sample of {sample_size} rows")
        
        # Add prediction column
        print("  Making predictions...")
        df['predicted_fire_type'] = df.apply(
            lambda row: predict_fire_type(row, model, scaler), 
            axis=1
        )
        
        # Save results
        df.to_csv(output_file, index=False)
        print(f"  Results saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False

def main():
    # Load model and scaler
    print("Loading model and scaler...")
    model = load_model()
    scaler = load_scaler()
    
    if model is None or scaler is None:
        return
    
    # Create output directory if it doesn't exist
    os.makedirs("predictions", exist_ok=True)
    
    # Process each year's data
    years = [2021, 2022, 2023]
    for year in years:
        input_file = f"modis_{year}_India.csv"
        output_file = f"predictions/predictions_{year}.csv"
        
        if os.path.exists(input_file):
            # For demonstration, process only first 1000 rows from each file
            # Remove sample_size=None to process all rows
            process_csv(input_file, output_file, model, scaler, sample_size=1000)
        else:
            print(f"File not found: {input_file}")
    
    print("\nPrediction complete! Check the 'predictions' folder for results.")

if __name__ == "__main__":
    main()
