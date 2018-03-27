from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
from os.path import exists
from encoder import encode
from decoder import decode
from input_data import input_data

logdir = "checkpoints/model.ckpt"

# network parameters
learning_rate = 0.01
num_steps = 1000
display_step = 10

X = tf.placeholder(tf.float32, [None, 224, 224, None])

with tf.name_scope("Encoding"):
    encoded_image = encode(X)

with tf.name_scope("Decoding"):
    decoded_image = decode(encoded_image)

with tf.name_scope("Loss"):
    y_pred = decoded_image
    y_true = X
    loss = tf.reduce_mean(tf.pow(y_pred, y_true), 2)

with tf.name_scope("Optimization"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss=loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    if exists(logdir):
        sess = saver.restore(sess, logdir)
    iterator = input_data()

    for i in range(num_steps + 1):
        next_items = iterator.get_next()
        batch_x =  sess.run(next_items)
        _ = sess.run(train_op, feed_dict={X: batch_x})
        if i % display_step == 0:
            loss_ = sess.run(loss, feed_dict={X: batch_x})
            print("Iteration number: ", str(i), " Loss: ", str(loss_))
            saver.save(sess, logdir)






