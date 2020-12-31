#!/usr/bin/python3
import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import tensorflow.keras as keras

class SimpleLayer(keras.Model):
    def __init__(self, size_x, size_y, size_z):
        super().__init__()

        self.weight = self.add_weight(
            shape=(size_x, size_y, size_z),
            initializer='random_normal',
            trainable=False
        )

    def call(self, to):
      # input is just a index
      return self.weight[:to]

sizex = 128
sizey = 8
sizez = 32

model = SimpleLayer(sizex, sizey, sizez)


@tf.function(input_signature=[tf.TensorSpec([], dtype=tf.int32)])
def func(to):
    return model(to)


## TEST at TensorFlow
for i in range(10):
  to = random.randint(0, sizex - 1)

  out = func(to)

  print(f'TEST: Input: [{sizex}, {sizey}, {sizez}] SiliceTo: {to} Output: {out.shape}')


## Convert to tflite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
]
converter.experimental_new_converter = True

tflite_model = converter.convert()

with tf.io.gfile.GFile('model/strided_slice.tflite', 'wb') as f:
    f.write(tflite_model)
