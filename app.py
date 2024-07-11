from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("eTc.sav")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email = request.form['email']
    prediction = model.predict([email])[0]  # Retrieve the first prediction from the array
    result = "Safe Email" if prediction == "Safe Email" else "Phishing"
    print(email)
    print(prediction)
    print(result)
    return render_template('result.html', email=email, result=result)

if __name__ == '__main__':
    app.run(debug=True)
