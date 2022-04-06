# module import 
import tensorflow as tf

IMG_WIDTH = 256
IMG_HIGHT = 256
IMG_CHANNEL = 3

# model building
inputs = tf.keras.layers.Input( (IMG_WIDTH, IMG_HIGHT, IMG_CHANNEL) )

s = tf.keras.layers.Lambda(lambda x : x / 255)(inputs)
ß