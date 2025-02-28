import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

try:
    crop = pd.read_csv("Crop_Recommendation.csv")
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Crop_Recommendation.csv not found!")
    exit()

if 'Crop' not in crop.columns:
    print("❌ Error: 'Crop' column not found in the dataset!")
    exit()

X = crop.drop('Crop', axis=1)
y = crop['Crop']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


nb_model = GaussianNB()
nb_model.fit(X_train, y_train)


filename = "nb_model.pkl"
with open(filename, "wb") as file:
    pickle.dump(nb_model, file)

print("✅ Model trained and saved successfully as 'nb_model.pkl'!")

