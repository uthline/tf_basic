"""
https://www.edwith.org/others26/lecture/10732/
"""
import tensorflow.compat.v1 as tf
with tf.compat.v1.Session() as sess:
    W = tf.Variable(tf.random_normal([1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    # Our hypothesis XW + b
    hypothesis = X * W + b

    # cost/loss function
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    # minimize
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(cost)

    # initial global variables in the graph
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                                             feed_dict={X: [1, 2, 3, 4, 5], Y: [2.1, 3.1, 4.1, 5.1, 6.1]})

        if step % 20 == 0:
            print(step,  cost_val, W_val, b_val)

    print(sess.run(hypothesis, feed_dict={X: [5]}))
    print(sess.run(hypothesis, feed_dict={X: [2.5]}))
    print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))
