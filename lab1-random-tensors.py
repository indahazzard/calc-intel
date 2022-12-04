import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

sess = tf.Session()
R1 = tf.random_uniform([2, 3], minval=0, maxval=4)
print(sess.run(R1))

R2 = tf.random_normal([2,3], mean=5, stddev=4)
print(sess.run(R2))
