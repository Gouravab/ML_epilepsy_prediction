import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.io import savemat

# Load dataset
ESR = pd.read_csv('data.csv')

# Target variable
tgt = ESR.y
tgt[tgt > 1] = 0

# Plot class distribution
ax = sn.countplot(tgt, label="Count")
non_seizure, seizure = tgt.value_counts()
print('The number of trials for the non-seizure class is:', non_seizure)
print('The number of trials for the seizure class is:', seizure)

# Features and target
X = ESR.iloc[:, 1:179].values
y = ESR.iloc[:, 179].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Reshape data for LSTM [samples, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Build LSTM model
model = Sequential()

# Adding the LSTM layers and some Dropout regularization
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(units=1, activation='sigmoid'))

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Making predictions and evaluating the model
y_pred_lstm = (model.predict(X_test) > 0.5).astype("int32")
accuracy = model.evaluate(X_test, y_test)[1] * 100
print(f'Accuracy of LSTM model: {accuracy:.2f} %')

# Saving results
savemat('eeg_results_improved_lstm.mat', {'y_test': y_test, 'y_pred': y_pred_lstm})
