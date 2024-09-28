from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        present_price = float(request.form['present_price'])
        kms_driven = float(request.form['kms_driven'])
        fuel_type = int(request.form['fuel_type'])
        seller_type = int(request.form['seller_type'])
        transmission = int(request.form['transmission'])
        owner = int(request.form['owner'])
        age_of_the_car = int(request.form['age_of_the_car'])

        # Arrange the input data in the same order as training features
        input_features = np.array([[present_price, kms_driven, fuel_type, seller_type, transmission, owner, age_of_the_car]])

        # Make the prediction
        prediction = model.predict(input_features)

        # Return the result
        return f'The predicted selling price of the car is {prediction[0]:.2f} Lakhs'

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
