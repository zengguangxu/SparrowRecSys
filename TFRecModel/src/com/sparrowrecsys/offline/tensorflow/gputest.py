import tensorflow as tf
import numpy as np

gpus = tf.config.experimental.list_logical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus)
print(cpus)


x = np.arange(10).reshape(1, 5, 2)
print(x)
print(x[0][0][1])
# [[[0 1]
#   [2 3]
#  [4 5]
# [6 7]
# [8 9]]]
y = np.arange(10, 20).reshape(1, 2, 5)
print(y)
# [[[10 11 12 13 14]
#   [15 16 17 18 19]]]
tf.keras.layers.Dot(axes=(1, 2))([x, y])