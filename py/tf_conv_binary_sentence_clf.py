import tensorflow as tf
import numpy as np

num_filters = 2

dim_of_word_vec = 4
num_of_words = 6

x = tf.placeholder(tf.float32, [None, num_of_words, dim_of_word_vec])
x_input = tf.reshape(x, [-1, num_of_words, dim_of_word_vec, 1])

W_conv = tf.Variable(tf.truncated_normal([2, 2, 1, num_filters], stddev=0.1))

h_conv = tf.nn.conv2d(x_input, W_conv, strides=[1, 1, 1, 1], padding='SAME')

h_pool = tf.nn.max_pool(h_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

h_pool_flat = tf.reshape(h_pool, [-1, int(dim_of_word_vec/2) * int(num_of_words/2) * num_filters])

num_units1 = int(dim_of_word_vec/2) * int(num_of_words/2) * num_filters
num_units2 = 1024

w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
b2 = tf.Variable(tf.zeros([num_units2]))
hidden2 = tf.nn.relu(tf.matmul(h_pool_flat, w2) + b2)
                 
w0 = tf.Variable(tf.zeros([num_units2, 2]))
b0 = tf.Variable(tf.zeros([2]))
p = tf.nn.softmax(tf.matmul(hidden2, w0) + b0)
                 
t = tf.placeholder(tf.float32, [None, 2])
loss = -tf.reduce_sum(t * tf.log(p))
train_step = tf.train.AdamOptimizer(0.0005).minimize(loss)
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                 
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

data = np.array([
       # Sentence1
       [[1, 2, 3, 4], # Word1
        [5, 6, 7, 8], # Word2
        [9, 10, 11, 12], # Word3
        [13, 14, 15, 16], # Word4
        [17, 18, 19, 20], # Word5
        [21, 22, 23, 24]], # Word6
       # Sentence 2
        [[1, 1, 1, 1], # Word1
        [1, 1, 1, 1], # Word2
        [1, 1, 1, 1], # Word3
        [1, 1, 1, 1], # Word4
        [1, 1, 1, 1], # Word5
        [1, 1, 1, 1]] # Word6
                ])

labels = np.array([
        [1, 0], # Sentence1のラベル
        [0, 1] # Sentence2のラベル
        ])

i = 0
for _ in range(4000):
    i += 1
    batch_xs = data
    batch_ts = labels
    sess.run(train_step, feed_dict={x: batch_xs, t: batch_ts})
    if i % 100 == 0:
         #loss_val, acc_val = sess.run([loss, accuracy], feed_dict={x:test_x, t:test_labels})
         #print('Step:{}, Loss:{}, Accuracy:{}'.format(i, loss_val, acc_val))
         saver.save(sess, 'mdc_session', global_step=i)
predicted = sess.run(p, feed_dict={x: batch_xs})
print("predicted : ", np.argmax(predicted, axis=1))
