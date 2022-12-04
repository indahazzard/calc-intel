import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

sess = tf.Session()

# Assuming prediction model
pred = np.asarray([0.2, 0.3, 0.5, 10.0, 12.0, 13.0, 3.5, 7.4, 3.9, 2.3])

# convert ndarray into tensor
x_val = tf.convert_to_tensor(pred)

# Assuming actual values
actual = np.asarray([0.1, 0.4, 0.6, 9.0, 11.0, 12.0, 3.4, 7.1, 3.8, 2.0])

# L2 loss:L1=(pred-actual)^2
l2 = tf.square(pred-actual)
l2_out = sess.run(tf.round(l2))
print(l2_out)

# L2 loss:L1=abs(pred-actual)
l1 = tf.abs(pred-actual)
l1_out = sess.run(l1)
print(l1_out)

# cross entropy loss
softmax_xentropy_variable = tf.nn.sigmoid_cross_entropy_with_logits(logits=l1_out,labels=l2_out)
print(sess.run(softmax_xentropy_variable))
