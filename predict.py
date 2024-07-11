import joblib

# Load the saved model
loaded_classifier = joblib.load('eTc.sav')

# Example new data
new_data = [" Hi The test link will be active till 11:59 PM today. Pls complete the test before the deadline.regards CGPU_MITS"]

# Predict on new data
new_predictions = loaded_classifier.predict(new_data)

print(new_predictions)
