# Import necessary libraries
import pandas as pd


# Load dataset
df = pd.read_csv("maintenance_data_ai4i2020.csv")  # Ensure you have the correct path

# Show basic info
print("Dataset Head:\n", df.head())
print("\nDataset Summary:\n", df.info())
print("\nMissing Values:\n", df.isnull().sum())




