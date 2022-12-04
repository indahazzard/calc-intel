import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
x = tf.constant(12, dtype='float32')
sess = tf.Session()
print(sess.run(x))

