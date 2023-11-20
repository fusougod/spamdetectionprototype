import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_spam_detection_model(data_path1, data_path2):
    raw_sms_data1 = pd.read_csv('sms_dataset.csv', encoding='latin-1')
    raw_sms_data2 = pd.read_csv('mail_dataset.csv', encoding='latin-1')
    raw_sms_data = pd.concat([raw_sms_data1, raw_sms_data2], ignore_index=True)

    sms_data = raw_sms_data.fillna('')
    sms_data['Category'] = sms_data['Category'].apply(lambda x: 0 if x.strip() == 'spam' else 1)

    X = sms_data['Message']
    Y = sms_data['Category']

    model = SVC(class_weight='balanced', kernel='linear')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

    feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

    X_train_features = feature_extraction.fit_transform(X_train)
    X_test_features = feature_extraction.transform(X_test)

    model = SVC(class_weight='balanced', kernel='linear') 
    model.fit(X_train_features, Y_train)

    prediction_on_training_data = model.predict(X_train_features)
    accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
    precision_on_training_data = precision_score(Y_train, prediction_on_training_data)
    recall_on_training_data = recall_score(Y_train, prediction_on_training_data)
    f1_on_training_data = f1_score(Y_train, prediction_on_training_data)

    prediction_on_test_data = model.predict(X_test_features)
    accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
    precision_on_test_data = precision_score(Y_test, prediction_on_test_data)
    recall_on_test_data = recall_score(Y_test, prediction_on_test_data)
    f1_on_test_data = f1_score(Y_test, prediction_on_test_data)

    return model, feature_extraction
