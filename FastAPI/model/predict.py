import pickle

# importing the ml model
with open('model/insurance.pkl', 'rb') as f:
    model = pickle.load(f)

# MLFlow
MODEL_VERSION = "1.0.0"
