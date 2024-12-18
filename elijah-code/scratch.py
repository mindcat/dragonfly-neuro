from tensorflow.keras.models import load_model, save
import tensorflow.keras

model_keras = load_model('models/algo_trained.keras')

model_keras.save('models/algo_trained.h5')