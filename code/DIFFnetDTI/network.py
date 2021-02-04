import tensorflow as tf
import numpy as np
import math
from hyperparam import * 
from network_util import *

class network:
            def __init__(self, input_shape, output_size, reuse, isTrain, keep_prob, learning_rate, factor, chan, quantization,train_data_num, name='DIFFnet'):
                self.input_shape = input_shape
                self.output_size = output_size
                self.learning_rate = learning_rate
                self.reuse = reuse
                self.isTrain = isTrain
                self.keep_prob = keep_prob
                self.global_step = tf.Variable(0, trainable=False)
                self.factor = factor
                self.chan = chan
                self.qn = quantization
                self.dn = train_data_num
                self.rate = tf.train.exponential_decay(self.learning_rate,self.global_step,self.dn/train_batch_size,0.87,staircase= True)

                with tf.variable_scope(name, reuse=reuse) as scope:
                    with tf.variable_scope("input", reuse=reuse) as scope:
                        self.inputs = tf.placeholder(tf.float32, [None,self.input_shape,self.input_shape,default_chan], name='inputs')
                        scope.reuse_variables()


                    with tf.variable_scope("normal_stage1", reuse=reuse) as scope:
                        self.normal_stage1 = normal_stage(self.inputs, self.keep_prob, True, isTrain, self.reuse,
                                                          self.chan, 7, first=True)
                        scope.reuse_variables()

                    with tf.variable_scope("normal_stage2", reuse=reuse) as scope:
                        self.normal_stage2 = normal_stage(self.normal_stage1, self.keep_prob, True, isTrain, self.reuse,
                                                          self.chan * 2, 5)
                        scope.reuse_variables()

                    with tf.variable_scope("normal_stage3", reuse=reuse) as scope:
                        self.normal_stage3 = normal_stage(self.normal_stage2, self.keep_prob, True, isTrain, self.reuse,
                                                          self.chan * 4, 3)
                        scope.reuse_variables()

                    with tf.variable_scope("normal_stage4", reuse=reuse) as scope:
                        self.normal_stage4 = normal_stage(self.normal_stage3, self.keep_prob, True, isTrain, self.reuse,
                                                          self.chan * 8, 3)
                        scope.reuse_variables()
                        
                    self.pool = avg_pool(self.normal_stage4,self.qn)

                    self.flat = tf.layers.flatten(self.pool)


                    self.linear_layer = tf.layers.dense(inputs=self.flat,
                                                        units=self.output_size*2,
                                                        activation=tf.nn.elu,
                                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                        name='linear_layer')


                    self.output = tf.layers.dense(inputs=self.linear_layer,
                                                  units=self.output_size,
                                                  activation=None,
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                  name='output')

                    self.label = tf.placeholder(tf.float32, [None, self.output_size], name='label')

                    self.loss = tf.reduce_mean(tf.square(self.label - self.output))

                    self.optimizer = tf.train.AdamOptimizer(self.rate).minimize(self.loss,global_step = self.global_step)