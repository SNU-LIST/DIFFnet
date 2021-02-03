import tensorflow as tf
import numpy as np
from hyperparam import * 
import math

def avg_pool(x,quantization):
            avg_pool_size1 = math.ceil(math.ceil(math.ceil(quantization / 2) / 2) / 2)
            avg_pool_size2 = math.ceil(math.ceil(math.ceil(quantization / 2) / 2) / 2)
            return tf.nn.avg_pool(x, ksize=[1, avg_pool_size1, avg_pool_size2, 1],
                                  strides=[1, avg_pool_size1, avg_pool_size2, 1], padding='SAME')


def batch_norm(x, channel, isTrain, decay=0.99, name="bn"):

            with tf.variable_scope(name):
                beta = tf.get_variable(initializer=tf.constant(0.0, shape=[channel]), name='beta')
                gamma = tf.get_variable(initializer=tf.constant(1.0, shape=[channel]), name='gamma')
                batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                mean_sh = tf.get_variable(initializer=tf.zeros([channel]), name="mean_sh", trainable=False)
                var_sh = tf.get_variable(initializer=tf.ones([channel]), name="var_sh", trainable=False)

                def mean_var_with_update():
                    mean_assign_op = tf.assign(mean_sh, mean_sh * decay + (1 - decay) * batch_mean)
                    var_assign_op = tf.assign(var_sh, var_sh * decay + (1 - decay) * batch_var)
                    with tf.control_dependencies([mean_assign_op, var_assign_op]):
                        return tf.identity(batch_mean), tf.identity(batch_var)

                mean, var = tf.cond(tf.cast(isTrain, tf.bool), mean_var_with_update, lambda: (mean_sh, var_sh))
                # print(x)
                normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3, name="normed")

            return normed


def conv2d(x, w_shape, b_shape, s_shape, keep_prob_, train, isTrain, relu=True):
            weights = tf.get_variable("conv_weights", w_shape, initializer=tf.contrib.layers.xavier_initializer(),
                                      trainable=train)
            conv_2d = tf.nn.conv2d(x, weights, strides=s_shape, padding='SAME')
            biases = tf.get_variable("biases", b_shape, initializer=tf.random_normal_initializer(), trainable=train)

            conv_2d = tf.nn.bias_add(conv_2d, biases)
            channel = conv_2d.get_shape().as_list()[-1]
            bn_x = batch_norm(conv_2d, channel, isTrain)

            if relu == True:
                bn_x = tf.nn.leaky_relu(bn_x, alpha=0.01)

            return bn_x


def residual_block(x, keep_prob, train, isTrain, reuse, chan, filter_size):
            with tf.variable_scope("conv1", reuse=reuse) as scope:
                y = conv2d(x, [1, 1, chan, chan / 2], [chan / 2], [1, 1, 1, 1], keep_prob, train, isTrain)
                scope.reuse_variables()

            with tf.variable_scope("conv2", reuse=reuse) as scope:
                y = conv2d(y, [filter_size, filter_size, chan / 2, chan / 2], [chan / 2], [1, 1, 1, 1], keep_prob,
                           train, isTrain)
                scope.reuse_variables()

            with tf.variable_scope("conv3", reuse=reuse) as scope:
                y = conv2d(y, [1, 1, chan / 2, chan], [chan], [1, 1, 1, 1], keep_prob, train, isTrain, relu=False)
                scope.reuse_variables()

            return tf.nn.relu(x + y)


def identity_block(x, keep_prob, train, isTrain, reuse, chan, filter_size, first=False):
            if first == True:
                with tf.variable_scope("conv1", reuse=reuse) as scope:
                    y = conv2d(x, [1, 1, default_chan, chan / 2], [chan / 2], [1, 1, 1, 1], keep_prob, train, isTrain)
                    scope.reuse_variables()

                with tf.variable_scope("conv2", reuse=reuse) as scope:
                    y = conv2d(y, [filter_size, filter_size, chan / 2, chan / 2], [chan / 2], [1, 1, 1, 1], keep_prob,
                               train, isTrain)
                    scope.reuse_variables()

                with tf.variable_scope("conv3", reuse=reuse) as scope:
                    y = conv2d(y, [1, 1, chan / 2, chan * 2], [chan * 2], [1, 1, 1, 1], keep_prob, train, isTrain)
                    scope.reuse_variables()

                with tf.variable_scope("conv4", reuse=reuse) as scope:
                    x = conv2d(x, [1, 1, default_chan, chan * 2], [chan * 2], [1, 1, 1, 1], keep_prob, train, isTrain,
                               relu=False)
                    scope.reuse_variables()

            else:
                with tf.variable_scope("conv1", reuse=reuse) as scope:
                    y = conv2d(x, [1, 1, chan, chan / 2], [chan / 2], [1, 2, 2, 1], keep_prob, train, isTrain)
                    scope.reuse_variables()

                with tf.variable_scope("conv2", reuse=reuse) as scope:
                    y = conv2d(y, [filter_size, filter_size, chan / 2, chan / 2], [chan / 2], [1, 1, 1, 1], keep_prob,
                               train, isTrain)
                    scope.reuse_variables()

                with tf.variable_scope("conv3", reuse=reuse) as scope:
                    y = conv2d(y, [1, 1, chan / 2, chan * 2], [chan * 2], [1, 1, 1, 1], keep_prob, train, isTrain,
                               relu=False)
                    scope.reuse_variables()

                with tf.variable_scope("conv4", reuse=reuse) as scope:
                    x = conv2d(x, [1, 1, chan, chan * 2], [chan * 2], [1, 2, 2, 1], keep_prob, train, isTrain,
                               relu=False)
                    scope.reuse_variables()

            return tf.nn.relu(x + y)


def normal_stage(x, keep_prob, train, isTrain, reuse, chan, filter_size, first=False):
            with tf.variable_scope("block1", reuse=reuse) as scope:
                x = identity_block(x, keep_prob, train, isTrain, reuse, chan, filter_size, first=first)
                scope.reuse_variables()

            with tf.variable_scope("block2", reuse=reuse) as scope:
                x = residual_block(x, keep_prob, train, isTrain, reuse, chan * 2, filter_size)
                scope.reuse_variables()

            with tf.variable_scope("block3", reuse=reuse) as scope:
                x = residual_block(x, keep_prob, train, isTrain, reuse, chan * 2, filter_size)
                scope.reuse_variables()

            with tf.variable_scope("block4", reuse=reuse) as scope:
                x = residual_block(x, keep_prob, train, isTrain, reuse, chan * 2, filter_size)
                scope.reuse_variables()

            return x