import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

sess = tf.Session()
G = tf.range(start=6, limit=45, delta=3)
print(sess.run(G))

H = tf.linspace(10.0, 92.0, 5)
print(sess.run(H))
