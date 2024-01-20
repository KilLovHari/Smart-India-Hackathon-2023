import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Simulated dataset (replace with real-world data)
data = {
    'Temperature': [380, 385, 390, 395, 400, 405, 410, 415, 420, 425],
    'Vibration': [6.0, 5.8, 5.5, 5.7, 6.2, 6.5, 6.8, 7.0, 7.2, 7.5],
    'Eccentricity': [3.2, 3.5, 3.0, 3.8, 4.0, 3.7, 3.5, 3.2, 3.0, 3.3],
    'RotationalSpeed': [1600, 1610, 1620, 1630, 1640, 1650, 1660, 1670, 1680, 1690],
    'Cavitation': [55, 60, 50, 65, 70, 45, 55, 60, 62, 58],
    'YoungsModulus': 200e9,  # Example Young's Modulus in Pascals
    'TensileStrength': 400e6,  # Example Tensile Strength in Pascals
    'Elasticity': 0.3,  # Example Elasticity
    'RemainingLife': [15, 14, 13, 12, 11, 10, 9, 8, 7, 6]  # in years
}

df = pd.DataFrame(data)

# Split data into features (X) and labels (y)
X = df.drop('RemainingLife', axis=1)
y = df['RemainingLife']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# Save the trained model for future use
joblib.dump(model, 'generator_remaining_life_model.joblib')

# Use the trained model for prediction
# Assuming you have new sensor data for the generator
new_data = {
    'Temperature': [410],
    'Vibration': [6.5],
    'Eccentricity': [3.3],
    'RotationalSpeed': [1665],
    'Cavitation': [58],
    'YoungsModulus': 200e9,  # Constant Young's Modulus for the generator
    'TensileStrength': 400e6,  # Constant Tensile Strength for the generator
    'Elasticity': 0.3,  # Constant Elasticity for the generator
}

new_data_df = pd.DataFrame(new_data)

# Make predictions using the trained model
predicted_remaining_life = model.predict(new_data_df)[0]

print(f'Predicted Remaining Life of the Generator: {predicted_remaining_life:.2f} years')
