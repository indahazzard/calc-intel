import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

sess = tf.Session()
# ReLU can't have values under zero.
# Uses max value for activation.
A = tf.nn.relu([-2.,1.,-3.,13.])
print(sess.run(A))

# ELU can have values under zero and non integer values
# Uses values which closes to zero for activation 
B = tf.nn.elu([-2.,1.,-3.,13.])
print(sess.run(B))
