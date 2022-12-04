import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

sess = tf.Session()
E = tf.nn.tanh([10,2,1,0.5,0,-0.5,1.,2.,5.,10.])
print(sess.run(E))

J = tf.nn.sigmoid([10,2,1,0.5,0,-0.5,1.,2.,5.,10.])
print(sess.run(J))
