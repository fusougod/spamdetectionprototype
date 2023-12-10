# Import redirect and url_for
from flask import Flask, render_template, request, redirect, url_for
from spam_detection_model import train_spam_detection_model
import os
from imapclient import IMAPClient
from email.header import decode_header
import email
from email import policy
from email.parser import BytesParser

app = Flask(__name__, static_url_path='/static')

script_dir = os.path.dirname(os.path.abspath(__file__))

sms_data_path = os.path.join(script_dir, 'sms_dataset.csv')
mail_data_path = os.path.join(script_dir, 'mail_dataset.csv')

print("Absolute path for SMS dataset:", sms_data_path)
print("Absolute path for Mail dataset:", mail_data_path)

# Train the model when the application starts
trained_model, feature_extraction = train_spam_detection_model(mail_data_path, sms_data_path)

def get_imap_client(username, password):
    return IMAPClient('imap.gmail.com', use_uid=True, ssl=True, port=993)

# Add your mail password here
mail_password = 'tofl nccl vlrk hvke'

# List to store blocked senders
blocked_senders = []


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_sms', methods=['POST'])
def predict_sms():
    if request.method == 'POST':
        sms_input = request.form['sms_input']

        input_data_features = feature_extraction.transform([sms_input])
        my_prediction = trained_model.predict(input_data_features)

        # Example of getting accuracy after prediction (replace with your actual calculation)
        accuracy_on_test_data = 0.85

        # Pass the necessary variables to the template
        return render_template('sms.html',
                               prediction="Spam" if my_prediction[0] == 0 else "Not Spam",
                               accuracy_on_test_data=accuracy_on_test_data)


@app.route('/view-emails', methods=['GET', 'POST'])
def view_emails():
    if request.method == 'POST':
        # Get user inputs
        email = request.form['email']
        password = request.form['password']

        try:
            # Use IMAP to get emails
            with get_imap_client(email, password) as client:
                client.login(email, password)

                # Example: extracting the subject, sender, date, and snippet of the first 10 emails
                select_info = client.select_folder('INBOX')
                messages = client.search()
                emails = []
                if messages:
                    for message_id in messages[-10:]:  # Adjust the range to get more emails
                        raw_message = client.fetch([message_id], ['BODY[]'])[message_id][b'BODY[]']

                        # Parse the email message using the email library
                        msg = BytesParser(policy=policy.default).parsebytes(raw_message)
                        subject, encoding = decode_header(msg['Subject'])[0]
                        if isinstance(subject, bytes):
                            subject = subject.decode(encoding or 'utf-8')
                        sender = msg.get('From', 'N/A')

                        # Check if the sender is blocked
                        if {'email': email, 'sender': sender} in blocked_senders:
                            continue  # Skip this email

                        date = msg['Date']
                        body = msg.get_body(preferencelist=('plain', 'html')).get_content()

                        # Use the trained model to predict if the email is spam
                        input_data_features = feature_extraction.transform([body])
                        spam_prediction = trained_model.predict(input_data_features)[0]

                        emails.append({'subject': subject, 'sender': sender, 'date': date, 'body': body, 'spam_prediction': spam_prediction})

                return render_template('emails.html', emails=emails, blocked_senders=blocked_senders)

        except Exception as e:
            return render_template('emails.html', error=str(e))

    return render_template('emails.html', emails=[], blocked_senders=blocked_senders)

@app.route('/block-sender', methods=['POST'])
def block_sender():
    if request.method == 'POST':
        email = request.form['email']
        sender_to_block = request.form['sender']

        print(f"Blocking sender: {sender_to_block} for email: {email}")

        # Add logic to check if the sender is already blocked
        if {'email': email, 'sender': sender_to_block} not in blocked_senders:
            blocked_senders.append({'email': email, 'sender': sender_to_block})
            print(f"Sender blocked successfully. Blocked senders: {blocked_senders}")
        else:
            print(f"Sender is already blocked. Blocked senders: {blocked_senders}")

        # Redirect back to the view-emails page or any other appropriate page
        return redirect(url_for('view_emails'))

@app.route('/get-started')
def get_started():
    return render_template('get_started.html')

@app.route('/choose-redirect-html')
def choose_redirect_html():
    return render_template('choose_redirect.html')

# Route for SMS detection
@app.route('/sms')
def sms():
    return render_template('sms.html')

if __name__ == '__main__':
    app.run(debug=True)
