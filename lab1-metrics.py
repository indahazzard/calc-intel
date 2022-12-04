import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

x = tf.placeholder(tf.int32, [5])
y = tf.placeholder(tf.int32, [5])

acc, acc_op = tf.metrics.accuracy(labels=x, predictions=y)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

val = sess.run([acc, acc_op], feed_dict={x: [1, 1, 0, 1, 0], y: [0, 1, 0, 0, 1]})

val_acc = sess.run(acc)
print(val_acc)
