# import akida
import numpy as np
import keras
from quantizeml.models import quantize, QuantizationParams
from cnn2snn import convert

model_keras = keras.models.load_model('models/algo_trained.keras')

qparams = QuantizationParams(input_weight_bits=8, weight_bits=8, activation_bits=8)
model_quantized = quantize(model_keras, qparams=qparams)
model_quantized.summary()

model_akida = convert(model_quantized)
model_akida.summary()

model_akida.save('models/algo_trained_akida.fbz')