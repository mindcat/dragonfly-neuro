# FILE: training.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Constants
FOV_SIZE = 21
NUM_SAMPLES = 10000

# Generate synthetic dataset
def generate_synthetic_data(num_samples, fov_size):
    X = np.random.rand(num_samples, fov_size, fov_size, 1)  # Random FOVs
    y = np.random.randint(-1, 2, size=(num_samples, 2))  # Random pitch and yaw adjustments (-1, 0, 1)
    return X, y

# Generate dataset
# X, y = generate_synthetic_data(NUM_SAMPLES, FOV_SIZE)

X = np.load('data/X.npy')
y = np.load('data/y.npy')

# Define the model
model = Sequential([
    Input(shape=(FOV_SIZE, FOV_SIZE, 1)),
    Flatten(),
    Dense(441, activation='relu'),
    Dense(194481, activation='relu'),
    Dense(441, activation='relu'),
    Dense(2, activation='linear'),  # Output layer for pitch and yaw adjustments
    Reshape((2,))
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['accuracy'])

# Define a checkpoint callback to save the best model
checkpoint = ModelCheckpoint('models/algo_trained.keras', monitor='val_loss', save_best_only=True, mode='min')

# Train the model
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, callbacks=[checkpoint])

# Save the final model
model.save('models/algo_trained_final.keras')

print("Model training complete and saved as 'models/dragonfly_model.keras' and 'models/dragonfly_model_final.keras'")