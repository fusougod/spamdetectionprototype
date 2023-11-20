import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the data from a CSV file into a pandas DataFrame
try:
    raw_sms_data = pd.read_csv('mail_data.csv', encoding='latin-1')
except UnicodeDecodeError:
    print("Error: Unable to decode using 'latin-1' encoding. Please check the file encoding.")
    exit(1)

# Replace the null values with an empty string
sms_data = raw_sms_data.fillna('')

# Clean the 'Category' column values and convert to integers
sms_data['Category'] = sms_data['Category'].apply(lambda x: 0 if x.strip() == 'spam' else 1)

# Separating the data into texts and labels
X = sms_data['Message']
Y = sms_data['Category']

# Split the data into training data and test data
model = SVC(class_weight='balanced', kernel='linear')  # Use a linear kernel for simplicity

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Transform the text data to feature vectors that can be used as input to SVM
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

# Convert X_train and X_test to feature vectors
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Train the model (SVM)
model = SVC(class_weight='balanced', kernel='linear')  # Use a linear kernel for simplicity

# Train the SVM model with the training data
model.fit(X_train_features, Y_train)

# Prediction on training data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

# Prediction on test data
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

# Ask for user input (SMS) to predict
sms_input = input('Enter SMS: ')

# Convert SMS text to feature vectors
input_data_features = feature_extraction.transform([sms_input])

# Make prediction
my_prediction = model.predict(input_data_features)

# Display the prediction
print("Prediction:", "Spam" if my_prediction[0] == 0 else "Not Spam")
print("Accuracy on test data after prediction:", accuracy_on_test_data * 100, '%')
