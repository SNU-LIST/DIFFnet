import numpy as np
from collections import deque
import os
from scipy import io
from datetime import datetime
import time
import math
import argparse
import random
from hyperparam import * 
from memory import *
from network import *
from projection2D import *

#
# Description:
#  Training code of DIFFnetDTI
#
#  Copyright @ Juhyung Park
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : jack0878@snu.ac.kr
#


def main(arg):
    
    epochs = arg.ep
    quantization = arg.qn
    train_data_num = arg.dn
    
    np.random.seed(random.randrange(0, int(1e4)))
    tf.set_random_seed(random.randrange(0, int(1e4)))
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(arg.GPU)

    input_location =  "specify input file"
    keep_prob = tf.placeholder("float")
    data_count = 0
    trainset = Memory(train_data_num, train_batch_size)
    train_batch_num = math.ceil(train_data_num/train_batch_size)
    for dirName, subdirList, fileList in sorted(os.walk(input_location)):
        for filename in fileList:
            if ".mat" in filename.lower():
                temp_s_matrix = np.array(io.loadmat(input_location  + filename)['S'], dtype='float')
                temp_data_gt_matrix = np.array(io.loadmat(input_location + filename)['info'], dtype='float')
                gradient_matrix = np.array(io.loadmat(input_location + filename)['gradient'], dtype='float')
                snum_matrix = np.array(io.loadmat(input_location + filename)['snum'], dtype='int')
                temp = temp_data_gt_matrix.shape[0]
                for ii in range(temp):
                    data_count = data_count + 1
                    temp_s = temp_s_matrix[ii,:]
                    temp_data_gt = temp_data_gt_matrix[ii,:]
                    gradient = gradient_matrix[ii,:]
                    snum = int(snum_matrix[ii,:])
                    scheme_vector = projection2D(gradient,quantization,snum)
                    temp_s = (temp_s-mean)/std                  
                    temp_data = np.zeros([quantization, quantization, default_chan])
                    temp_data = np.expand_dims(temp_data, 4)
                    temp_data_gt = np.expand_dims(temp_data_gt, 2)

                    for i in range(snum):
                                temp_data[ scheme_vector[3 * i, 0], scheme_vector[3 * i, 1],scheme_vector[3 * i, 2],0] = temp_s[i]
                                temp_data[ scheme_vector[3 * i + 1, 0], scheme_vector[3 * i + 1, 1],scheme_vector[3 * i + 1, 2],0] = temp_s[i]
                                temp_data[ scheme_vector[3 * i + 2, 0], scheme_vector[3 * i + 2, 1],scheme_vector[3 * i + 2, 2],0] = temp_s[i]
                                temp_data[ scheme_vector[3 * i + snum*3, 0], scheme_vector[3 * i + snum*3, 1],scheme_vector[3 * i + snum*3, 2],0] = temp_s[i]
                                temp_data[ scheme_vector[3 * i + snum*3+1, 0], scheme_vector[3 * i + snum*3+1, 1],scheme_vector[3 * i + snum*3+1, 2],0] = temp_s[i]
                                temp_data[ scheme_vector[3 * i + snum*3+2, 0], scheme_vector[3 * i + snum*3+2, 1],scheme_vector[3 * i + snum*3+2, 2],0] = temp_s[i]

                    
                    trainset.inputadd(temp_data)
                    trainset.groundadd(temp_data_gt)

                    if data_count % 1000 == 0:
                        print('Training data loaded : %d'%data_count)

                    if data_count == train_data_num:
                        break
                        
                        
                        
    trainable = True
    reuse = False
    Dnet = network(quantization, output_shape, reuse, trainable, keep_prob, lr,  factor, chan,quantization ,train_data_num)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for epoch in range(epochs):
                    running_loss = 0
                    val_loss = 0
                    print('Epoch %d' % epoch, end=' ')
                    startTime = time.time()
                    for i in range(train_batch_num):
                        data = trainset.sampleinput(i)
                        label = trainset.sampleground(i)
                        data = np.array([each[0] for each in data])
                        label = np.array([each[0] for each in label])

                        loss, output, rate, _ = sess.run([Dnet.loss, Dnet.output, Dnet.rate, Dnet.optimizer],
                                                         feed_dict={Dnet.inputs: data, Dnet.label: label})
                        running_loss = (running_loss * i + loss) / (i + 1)



                    trainset.shuffle()
                    endTime = time.time()

                    print('total loss: %.5f' % (running_loss), end=' ')
                    print("Time taken:", endTime - startTime, end='  ')
                    
                    if epoch % 10 == 9 or epoch == 0:
                        model_path = "model path+" + ".ckpt"
                        save_path = saver.save(sess, model_path)
                        print("Model Saved")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Varying hyper parameter")
    parser.add_argument("--GPU" , type = int, default = Hyper_parameters['GPU'])
    parser.add_argument("--ep" , type = int, default = Hyper_parameters['epoch'])
    parser.add_argument("--qn" , type = int, default = Hyper_parameters['qn'])
    parser.add_argument("--dn" , type = int, default = Hyper_parameters['dn'])
    arg = parser.parse_args()

    main(arg)







