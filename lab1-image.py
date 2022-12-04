import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

image = tf.image.decode_jpeg(tf.read_file("./assets/test.jpg"), channels=3)
sess = tf.InteractiveSession()
print(sess.run(tf.shape(image)))
print(sess.run(image[10:15, 0:4, 1]))
