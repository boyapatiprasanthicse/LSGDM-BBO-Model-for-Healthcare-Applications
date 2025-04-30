# LSTM feature extraction module
# Build and train LSTM model for feature extraction
from keras.models import Sequential
from keras.layers import LSTM, Dense

def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
