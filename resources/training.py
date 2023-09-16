from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


def train_LogisticRegression(df):
    # Separate the input features (X) and the output target (y)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the input features (optional but often improves performance)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    print("Model Accuracy:", accuracy)
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(confusion)

    model_filename = 'logistic_regression_model.joblib'
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}")


import xgboost as xgb

def train_XGBClassifier(df):
    # Separate the input features (X) and the output target (y)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the input features (optional but often improves performance)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train an XGBoost model
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    print("Model Accuracy:", accuracy)
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(confusion)


from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam

def train_Keras(df):

    # Separate the input features (X) and the output target (y)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the input features (optional but often improves performance)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build a simple neural network with Keras
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred_binary)
    report = classification_report(y_test, y_pred_binary)
    confusion = confusion_matrix(y_test, y_pred_binary)

    print("Model Accuracy:", accuracy)
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(confusion)


def train_lstm(df, window_size):

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape the data for LSTM input (samples, timesteps, features)
    window_size = 6  # Adjust this based on the lag features used
    number_of_features = 11
    X_train = X_train.reshape(X_train.shape[0], window_size+1, number_of_features)
    X_test = X_test.reshape(X_test.shape[0], window_size+1, number_of_features)

    # Define and compile the LSTM model
    model = Sequential()
    model.add(LSTM(128, input_shape=(window_size+1, X_train.shape[2]), return_sequences=True))
    model.add(LSTM(64, input_shape=(window_size+1, X_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    # Train the model and save the training history
    history = model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred_binary)
    report = classification_report(y_test, y_pred_binary)
    confusion = confusion_matrix(y_test, y_pred_binary)

    print("Model Accuracy:", accuracy)
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(confusion)

    model.save("lstm.keras")

    '''
    # Plot loss over epochs
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # If you have monitored accuracy during training
    if 'accuracy' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'])
        plt.title('Training Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

    plt.show()
    '''