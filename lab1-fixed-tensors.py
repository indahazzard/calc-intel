import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

sess = tf.Session()
A = tf.zeros([2, 3])
print(sess.run(A))

B = tf.ones([4, 3])
print(sess.run(B))

C = tf.fill([2,3], 13)
print(sess.run(C))

D = tf.diag([4, -3, 2, 1])
print(sess.run(D))
