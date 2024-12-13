import requests
import pandas as pd
import numpy as np
import os
import time
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import timedelta, datetime
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import seaborn as sns

# Constants for wave speeds and default depth
P_WAVE_SPEED = 6  # km/s (approximate)
S_WAVE_SPEED = 3.5  # km/s (approximate)
DEFAULT_DEPTH = 10  # Default depth in km if depth data is missing

# Function to calculate P-wave and S-wave arrival times based on epicenter distance
def calculate_wave_arrival_times(depth, latitude, longitude):
    depth_km = depth if depth is not None else DEFAULT_DEPTH
    R = 6371  # Radius of the Earth in km
    lat1 = np.radians(0)  # Station latitude (adjustable)
    lon1 = np.radians(0)  # Station longitude (adjustable)
    lat2 = np.radians(latitude)
    lon2 = np.radians(longitude)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    epicenter_distance = R * c

    p_wave_time = epicenter_distance / P_WAVE_SPEED
    s_wave_time = epicenter_distance / S_WAVE_SPEED

    return p_wave_time, s_wave_time

# Function to collect historical earthquake data
def collect_historical_data(latitude, longitude, radius, start_time, end_time, output_directory):
    print("Collecting historical earthquake data...")
    url = f"https://earthquake.usgs.gov/fdsnws/event/1/query.geojson"
    params = {
        'starttime': start_time,
        'endtime': end_time,
        'latitude': latitude,
        'longitude': longitude,
        'maxradiuskm': radius,
        'minmagnitude': 1.0,
        'format': 'geojson'
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()

        try:
            data = response.json()
        except ValueError as e:
            print("Error parsing JSON response:", e)
            return None

        earthquake_data = []
        print(f"Number of features retrieved: {len(data.get('features', []))}")
        for feature in data.get('features', []):
            properties = feature['properties']
            depth = properties.get('depth', DEFAULT_DEPTH)
            p_wave_time, s_wave_time = calculate_wave_arrival_times(depth, feature['geometry']['coordinates'][1], feature['geometry']['coordinates'][0])

            earthquake_data.append({
                'magnitude': properties.get('mag', None),
                'place': properties.get('place', "Unknown"),
                'time': pd.to_datetime(properties['time'], unit='ms') if 'time' in properties else None,
                'latitude': feature['geometry']['coordinates'][1],
                'longitude': feature['geometry']['coordinates'][0],
                'depth': depth,
                'p_wave_time': p_wave_time,
                's_wave_time': s_wave_time
            })

        if earthquake_data:
            df = pd.DataFrame(earthquake_data)
            historical_data_file = os.path.join(output_directory, f"historical_data_{start_time}_{end_time}_{int(time.time())}.csv")
            df.to_csv(historical_data_file, index=False)
            print(f"Historical data saved to {historical_data_file}")
            return df
        else:
            print("No earthquake data available for the given parameters.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error in fetching data from API: {e}")
        return None

# Dataset class for BERT
class EarthquakeDataset(Dataset):
    def __init__(self, texts, targets, is_prediction=False):
        self.texts = texts
        self.targets = targets
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.is_prediction = is_prediction

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx] if not self.is_prediction else 0  # For prediction, no target is needed

        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'target': torch.tensor(target, dtype=torch.float)
        }

# Function to train the BERT model and evaluate it
def train_and_test_model(historical_df, num_epochs=20, batch_size=64):  # Increased batch size to 64
    print("Preparing data for training...")
    texts = historical_df['place'].astype(str).tolist()
    targets = historical_df['magnitude'].tolist()

    # Create binary labels based on a threshold magnitude
    threshold = 2.0  # Adjust threshold as needed
    binary_targets = [1 if mag >= threshold else 0 for mag in targets]

    X_train, X_val, y_train, y_val = train_test_split(texts, binary_targets, test_size=0.2, random_state=42)
    print(f"Training data size: {len(X_train)}, Validation data size: {len(X_val)}")

    train_dataset = EarthquakeDataset(X_train, y_train)
    val_dataset = EarthquakeDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
    model.to(device)
    print("Model loaded to device.")

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    scaler = GradScaler()

    print("Starting training...")
    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = 5  # Patience for early stopping

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)

            with autocast():
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = torch.nn.BCEWithLogitsLoss()(outputs.logits.flatten(), targets)  # Use BCE loss for binary classification

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['target'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                loss = torch.nn.BCEWithLogitsLoss()(outputs.logits.flatten(), targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping...")
            break

    print("Evaluating the model on the validation set...")
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.sigmoid(outputs.logits.flatten()).cpu().numpy()
            predictions.extend(preds)

    predictions = np.array(predictions)
    predictions_binary = [1 if pred >= 0.5 else 0 for pred in predictions]  # Convert probabilities to binary predictions

    accuracy = accuracy_score(y_val, predictions_binary)
    print(f"Validation Accuracy: {accuracy:.4f}")

    return model, predictions, y_val  # Return the trained model and predictions

# Function to predict future earthquakes
def predict_future_earthquakes(model, n_days, latitude=37.7749, longitude=-122.4194):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    print(f"Predicting future earthquake probabilities for the next {n_days} days...")
    future_dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n_days)]
    predictions = []

    for date in future_dates:
        # Create a dummy input based on the date, latitude, and longitude
        input_text = f"Predicted earthquake for {date} at coordinates ({latitude}, {longitude})"
        inputs = EarthquakeDataset([input_text], [0], is_prediction=True)
        input_loader = DataLoader(inputs, batch_size=1)

        with torch.no_grad():
            for batch in input_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.sigmoid(outputs.logits.flatten()).tolist()
                predictions.append(preds[0])  # Store the prediction

    return predictions

# Function to visualize predictions
def visualize_predictions(true_values, predictions, n_days, threshold):
    plt.figure(figsize=(16, 8))  # Increased figure size for better readability
    days = list(range(n_days))

    # Plotting the predicted probabilities
    plt.plot(days, predictions, label='Predicted Probabilities', color='blue', marker='o', linestyle='-')
    plt.axhline(y=threshold * 100, color='red', linestyle='--', label='Magnitude Threshold (2.0)')

    # Labels and Title
    plt.title('Future Earthquake Predictions', fontsize=18)
    plt.xlabel('Days Ahead', fontsize=14)
    plt.ylabel('Predicted Probability (%)', fontsize=14)
    plt.xticks(days)
    plt.yticks(np.arange(0, 101, 10))  # Setting y-ticks for better readability
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()  # Adjust layout to avoid clipping
    plt.show()

    # Plotting histogram for true vs predicted
    plt.figure(figsize=(16, 8))  # Increased figure size for better readability
    sns.histplot(true_values, color='green', label='True Magnitudes', kde=True, bins=30, stat='density', alpha=0.6)
    sns.histplot(predictions, color='blue', label='Predicted Probabilities', kde=True, bins=30, stat='density', alpha=0.6)

    # Labels and Title
    plt.title('Distribution of True and Predicted Values', fontsize=18)
    plt.xlabel('Magnitude / Probability', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()  # Adjust layout to avoid clipping
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Define parameters
    latitude = 37.7749  # Example latitude for San Francisco
    longitude = -122.4194  # Example longitude for San Francisco
    radius = 100  # Radius in km for historical data collection
    start_time = '2024-01-01'  # Start date for historical data
    end_time = '2024-10-08'  # End date for historical data
    output_directory = '.'  # Directory to save historical data

    # Collect historical earthquake data
    historical_df = collect_historical_data(latitude, longitude, radius, start_time, end_time, output_directory)

    if historical_df is not None:
        # Train and evaluate the model
        model, predictions, true_values = train_and_test_model(historical_df)  # Get both predictions and true values

        # Input for future predictions
        n_days = int(input("Enter the number of days for future predictions: "))
        future_predictions = predict_future_earthquakes(model, n_days, latitude=latitude, longitude=longitude)
        future_predictions_percentage = [pred * 100 for pred in future_predictions]
        print("Future Earthquake Predictions (in %):", future_predictions_percentage)

        # Visualize predictions
        visualize_predictions(true_values, future_predictions_percentage, n_days, threshold=2.0)

        # Optionally save the model
        model_path = 'earthquake_prediction_model.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
  # Function to calculate and plot P-wave and S-wave diagrams for n days
def plot_wave_diagrams(future_predictions, n_days, latitude, longitude):
    days = list(range(n_days))

    # Assume a constant depth (DEFAULT_DEPTH) for simplicity, modify if you have actual data
    depth = DEFAULT_DEPTH

    plt.figure(figsize=(16, 8))

    for i in days:
        # Calculate wave arrival times for each day
        p_wave_time, s_wave_time = calculate_wave_arrival_times(depth, latitude, longitude)

        # For plotting purposes, assume earthquake occurs at t = 0 for each day
        event_time = i  # Each day is marked by 'i' in this context

        # Plot P-Wave (faster arrival)
        plt.plot([event_time, event_time + p_wave_time], [0, 1], label=f"P-Wave Day {i+1}", color='blue', marker='o')

        # Plot S-Wave (slower arrival)
        plt.plot([event_time, event_time + s_wave_time], [0, 0.8], label=f"S-Wave Day {i+1}", color='green', marker='x')

    # Title and labels
    plt.title('P-Wave and S-Wave Arrival Times Over Days', fontsize=18)
    plt.xlabel('Time (days)', fontsize=14)
    plt.ylabel('Wave Amplitude', fontsize=14)
    plt.xticks(np.arange(0, n_days + 1, 1))
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage after future predictions are made
if __name__ == "__main__":
    # The code above for collecting data, training, and making predictions should run first.

    if future_predictions:
        # Plot P-wave and S-wave diagrams for the number of future days specified
        plot_wave_diagrams(future_predictions_percentage, n_days, latitude, longitude)

