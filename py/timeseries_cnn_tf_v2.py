import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime


class CNNMultiClassModel(object):
    """
    CNN多クラス分類モデル
    """

    def __init__(
            self, num_classes, row_size, col_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        self.x = tf.placeholder(
            tf.float32, [None, row_size, col_size], name="input_x")
        self.input_x = tf.reshape(self.x, [-1, row_size, col_size, 1])
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # 各フィルターサイズに対して畳み込み層とMaxPooling層を作成
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # 畳み込み層
                filter_shape = [filter_size, col_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(
                    filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(
                    0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.input_x,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # ReLU活性化関数適用
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, row_size - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # MaxPoolした各アウトプットを結合して1次元配列に変換
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(
                self.h_pool_flat, self.dropout_keep_prob)

        # スコアと予測値
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # 交差エントロピー損失関数
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # 精度
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(
                self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")


def create_timeseries_data(time_min, time_max, num_data):
    timeseries_data = []
    for time in np.linspace(time_min, time_max, num_data):
        x1 = np.sin(0.05*time) * np.exp(0.001*time)
        x2 = np.cos(0.01*time) * np.sin(0.003*time)
        x3 = np.sin(0.04*time) * np.sin(0.003*time)
        x4 = np.cos(0.04*time) * np.cos(0.001*time)
        timeseries_data.append([x1, x2, x3, x4])

    return np.array(timeseries_data)


def calc_label(timeseries_data, row_index, sample_interval):
    base = timeseries_data[row_index, 0]
    try:
        change_rate = sum(
            [(timeseries_data[row_index + sample_interval*(i+1), 0] - base) for i in range(10)]) / 10
    except Exception as e:
        print(e)
        change_rate = 0
    if change_rate > 0.4:
        label = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif change_rate > 0.3:
        label = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif change_rate > 0.2:
        label = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif change_rate > 0.1:
        label = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif change_rate > 0.05:
        label = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif change_rate > -0.05:
        label = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif change_rate > -0.1:
        label = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif change_rate > -0.2:
        label = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif change_rate > -0.3:
        label = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif change_rate > -0.4:
        label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    else:
        label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    return np.array(label)


def prepare_train_test_data(timeseries_data, block_row_size):
    """
    num_block = int(timeseries_data.shape[0] / block_row_size)
    series_data_blocks = []
    for i in range(num_block):
        try:
            block = timeseries_data[i*block_row_size:(i+1)*block_row_size,:]
            label = calc_label(timeseries_data = timeseries_data, row_index = (i+1)*block_row_size, sample_interval = 5)
            series_data_blocks.append([block, label])
        except Exception as e:
            print(e)
            continue
    """
    series_data_blocks = []
    skip_window = 50
    i = 0
    while True:
        try:
            block = timeseries_data[i*skip_window:i *
                                    skip_window+block_row_size, :]
            label = calc_label(timeseries_data=timeseries_data, row_index=i *
                               skip_interval+block_row_size, sample_interval=5)
            series_data_blocks.append([block, label])
            i += 1
        except Exception as e:
            print(e)
            break
    series_data_blocks = np.array(series_data_blocks)
    return (np.array([data for data in series_data_blocks[:int(0.7*len(series_data_blocks)), 0]]),
            np.array(
                [label for label in series_data_blocks[:int(0.7*len(series_data_blocks)), 1]]),
            np.array([data for data in series_data_blocks[int(
                0.7*len(series_data_blocks)):int(0.9*len(series_data_blocks)), 0]]),
            np.array([label for label in series_data_blocks[int(0.7*len(series_data_blocks)):int(0.9*len(series_data_blocks)), 1]]))


if __name__ == "__main__":
    timeseries_data = create_timeseries_data(
        time_min=0, time_max=1000, num_data=10000)
    #train_data = timeseries_data[:int(0.8*timeseries_data.shape[0]),:]
    #test_data = timeseries_data[int(0.8*timeseries_data.shape[0]):,:]
    block_row_size = 200
    train_data, train_label, test_data, test_label = prepare_train_test_data(
        timeseries_data=timeseries_data, block_row_size=block_row_size)

    print(timeseries_data.shape)
    print(train_data.shape)
    print(test_data.shape)

    #print(train_data[3:300, 1])
    print(np.argmax(np.array([data for data in test_data[3:50, 1]]), axis=1))

    plt.plot(range(len(timeseries_data[:, 0])), timeseries_data[:, 0])
    plt.plot(range(len(timeseries_data[:, 1])), timeseries_data[:, 1])
    plt.plot(range(len(timeseries_data[:, 2])), timeseries_data[:, 2])
    plt.plot(range(len(timeseries_data[:, 3])), timeseries_data[:, 3])
    plt.show()

    filter_sizes = [10, 30, 50, 100, 200]
    num_filters = 100

    sess = tf.InteractiveSession()
    with sess.as_default():
        model = CNNMultiClassModel(num_classes=11, row_size=block_row_size,
                                   col_size=4, filter_sizes=filter_sizes, num_filters=num_filters)

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        dropout_keep_prob = 0.5
        TRAIN_STEPS = 1000
        # 学習開始
        print("Started Training..")
        local_step = 0
        while True:
            local_step += 1

            x_batch = train_data
            y_batch = train_label
            feed_dict = {
                model.x: x_batch,
                model.input_y: y_batch,
                model.dropout_keep_prob: dropout_keep_prob
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, model.loss, model.accuracy],
                feed_dict)
            now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            print("{} : local_step/global_step : {}/{}".format(now, local_step, step))
            # 10エポックごとにテストデータに対して予測精度の検証を行う
            if local_step % 10 == 0:
                print("{} : local_step/global_step : {}/{} - accuracy : {}".format(now,
                                                                                   local_step, step, accuracy))
                print(
                    "{} : local_step/global_step : {}/{} - loss : {}".format(now, local_step, step, loss))
                for td, tl in zip(test_data, test_label):
                    predicted = sess.run(model.predictions, feed_dict={model.x: np.array(
                        [td]), model.dropout_keep_prob: dropout_keep_prob})
                    print("predicted label : {}, correct label : {}, predicted label - correct label : {}".format(
                        predicted, np.argmax(tl), predicted - np.argmax(tl)))

            if step >= TRAIN_STEPS:
                break
