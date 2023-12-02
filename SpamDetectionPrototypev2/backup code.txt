from flask import Flask, render_template, request
from spam_detection_model import train_spam_detection_model
import os

app = Flask(__name__, static_url_path='/static')

script_dir = os.path.dirname(os.path.abspath(__file__))

sms_data_path = os.path.join(script_dir, 'sms_dataset.csv')
mail_data_path = os.path.join(script_dir, 'mail_dataset.csv')

print("Absolute path for SMS dataset:", sms_data_path)
print("Absolute path for Mail dataset:", mail_data_path)

# Train the model when the application starts
trained_model, feature_extraction = train_spam_detection_model(mail_data_path, sms_data_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sms_input = request.form['sms_input']
        input_data_features = feature_extraction.transform([sms_input])
        my_prediction = trained_model.predict(input_data_features)

        # Example of getting accuracy after prediction (replace with your actual calculation)
        accuracy_on_test_data = 0.85

        # Pass the necessary variables to the template
        return render_template('index.html',
                               prediction="Spam" if my_prediction[0] == 0 else "Not Spam",
                               accuracy_on_test_data=accuracy_on_test_data)

if __name__ == '__main__':
    app.run(debug=True)
