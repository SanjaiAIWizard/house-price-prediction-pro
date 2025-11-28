from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and label encoder
model = joblib.load('house_price_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get form data
        try:
            bedrooms = float(request.form['bedrooms'])
            bathrooms = float(request.form['bathrooms'])
            sqft_living = int(request.form['sqft_living'])
            
            floors = float(request.form['floors'])
            city = request.form['city']

            # Convert city to numerical value
            city_encoded = label_encoder.transform([city])[0]

            # Make prediction
            features = np.array([[bedrooms, bathrooms, sqft_living, floors, city_encoded]])
            prediction = model.predict(features)[0]
            prediction = round(prediction, 2)
        except Exception as e:
            prediction = "Invalid input"

    return render_template('index.html', prediction = prediction)

if __name__ == '__main__':
    app.run(debug=True)
