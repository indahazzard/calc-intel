import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

x = tf.Variable(3, name='x', dtype=tf.float32)
log_x = tf.log(x)
log_x_squared = tf.square(log_x)

optimizer = tf.train.GradientDescentOptimizer(0.7)
train = optimizer.minimize(log_x_squared)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    print("starting at", "x: ", session.run(x), "log(x)^2: ", session.run(log_x_squared))
    for step in range(10):
        session.run(train)
        print("step", step, "x: ", session.run(x), "log(x)^2: ", session.run(log_x_squared))
