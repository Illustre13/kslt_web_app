import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.layers import Layer

class Attention(Layer):
    def __init__(self, attn_units, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.attn_units = attn_units

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attn_W",
            shape=(input_shape[-1], self.attn_units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attn_b",
            shape=(self.attn_units,),
            initializer="zeros",
            trainable=True,
        )
        self.v = self.add_weight(
            name="attn_v",
            shape=(self.attn_units, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.v, axes=1), axis=1)
        attention_weights = tf.squeeze(attention_weights, axis=-1)
        context_vector = tf.reduce_sum(
            attention_weights[..., tf.newaxis] * inputs, axis=1
        )
        return context_vector  # (Optionally return weights)
