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
#  Training code of DIFFnetNODDI
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
                    snum1 = snum_matrix[ii,0]
                    snum2 = snum_matrix[ii,1]
                    snum3 = snum_matrix[ii,2]
                    temp_s = temp_s_matrix[ii,0:snum1+snum2+snum3]
                    gradientb1 = gradient_matrix[ii,0:snum1,:]
                    gradientb2 = gradient_matrix[ii,snum1:snum1+snum2,:]
                    gradientb3 = gradient_matrix[ii,snum1+snum2:snum1+snum2+snum3,:]
                    temp_data_gt = temp_data_gt_matrix[ii,:]
                    scheme_vector = projection2D(gradientb1,gradientb2,gradientb3,quantization,snum1,snum2,snum3)
                    temp_data = np.zeros([quantization, quantization, default_chan])
                    temp_data = np.expand_dims(temp_data, 4)
                    temp_data_gt = np.expand_dims(temp_data_gt, 2)
                    tempb1 = (temp_s[0:snum1]- 0.7022 )/0.1159
                    tempb2 = (temp_s[snum1:snum1+snum2] - 0.4944)/0.1601
                    tempb3 = (temp_s[snum1+snum2:snum1+snum2+snum3] - 0.2676)/0.1516
                    for i in range(snum1):
                        temp_data[ scheme_vector[3 * i, 0], scheme_vector[3 * i, 1], scheme_vector[3 * i, 2],0] = tempb1[i]
                        temp_data[ scheme_vector[3 * i + 1, 0],scheme_vector[3 * i + 1, 1],scheme_vector[3 * i + 1, 2],0] = tempb1[i]
                        temp_data[ scheme_vector[3 * i + 2, 0],scheme_vector[3 * i + 2, 1],scheme_vector[3 * i + 2, 2],0] = tempb1[i]
                        temp_data[ scheme_vector[3 * i + 3*(snum1+snum2+snum3), 0],scheme_vector[3 * i + 3*(snum1+snum2+snum3), 1],scheme_vector[3 * i + 3*(snum1+snum2+snum3), 2],0] = tempb1[i]
                        temp_data[ scheme_vector[3 * i + 3*(snum1+snum2+snum3)+1, 0],scheme_vector[3 * i + 3*(snum1+snum2+snum3)+1, 1],scheme_vector[3 * i + 3*(snum1+snum2+snum3)+1, 2],0] = tempb1[i]
                        temp_data[ scheme_vector[3 * i + 3*(snum1+snum2+snum3)+2, 0],scheme_vector[3 * i + 3*(snum1+snum2+snum3)+2, 1],scheme_vector[3 * i + 3*(snum1+snum2+snum3)+2, 2],0] = tempb1[i]

                    for i in range(snum2):
                        temp_data[ scheme_vector[3 * i + 3*snum1, 0],scheme_vector[3 * i + 3*snum1, 1],scheme_vector[3 * i + 3*snum1, 2],0] = tempb2[i]
                        temp_data[ scheme_vector[3 * i + 3*snum1+1, 0],scheme_vector[3 * i + 3*snum1+1, 1],scheme_vector[3 * i + 3*snum1+1, 2],0] = tempb2[i]
                        temp_data[ scheme_vector[3 * i + 3*snum1+2, 0],scheme_vector[3 * i + 3*snum1+2, 1],scheme_vector[3 * i + 3*snum1+2, 2],0] = tempb2[i]
                        temp_data[ scheme_vector[3 * i + 3*(2*snum1+snum2+snum3), 0],scheme_vector[3 * i + 3*(2*snum1+snum2+snum3), 1],scheme_vector[3 * i + 3*(2*snum1+snum2+snum3), 2],0] = tempb2[i]
                        temp_data[ scheme_vector[3 * i + 3*(2*snum1+snum2+snum3)+1, 0],scheme_vector[3 * i + 3*(2*snum1+snum2+snum3)+1, 1],scheme_vector[3 * i + 3*(2*snum1+snum2+snum3)+1, 2],0] = tempb2[i]
                        temp_data[ scheme_vector[3 * i + 3*(2*snum1+snum2+snum3)+2, 0],scheme_vector[3 * i + 3*(2*snum1+snum2+snum3)+2, 1],scheme_vector[3 * i + 3*(2*snum1+snum2+snum3)+2, 2],0] = tempb2[i]

                    for i in range(snum3):
                        temp_data[ scheme_vector[3 * i + 3*(snum1+snum2), 0],scheme_vector[3 * i + 3*(snum1+snum2), 1],scheme_vector[3 * i + 3*(snum1+snum2), 2],0] = tempb3[i]
                        temp_data[ scheme_vector[3 * i + 3*(snum1+snum2)+1, 0],scheme_vector[3 * i + 3*(snum1+snum2)+1, 1],scheme_vector[3 * i + 3*(snum1+snum2)+1, 2],0] = tempb3[i]
                        temp_data[ scheme_vector[3 * i + 3*(snum1+snum2)+2, 0],scheme_vector[3 * i + 3*(snum1+snum2)+2, 1],scheme_vector[3 * i + 3*(snum1+snum2)+2, 2],0] = tempb3[i]
                        temp_data[ scheme_vector[3 * i + 3*(2*snum1+2*snum2+snum3), 0],scheme_vector[3 * i + 3*(2*snum1+2*snum2+snum3), 1],scheme_vector[3 * i + 3*(2*snum1+2*snum2+snum3), 2],0] = tempb3[i]
                        temp_data[ scheme_vector[3 * i + 3*(2*snum1+2*snum2+snum3)+1, 0],scheme_vector[3 * i + 3*(2*snum1+2*snum2+snum3)+1, 1],scheme_vector[3 * i + 3*(2*snum1+2*snum2+snum3)+1, 2],0] = tempb3[i]
                        temp_data[ scheme_vector[3 * i + 3*(2*snum1+2*snum2+snum3)+2, 0],scheme_vector[3 * i + 3*(2*snum1+2*snum2+snum3)+2, 1],scheme_vector[3 * i + 3*(2*snum1+2*snum2+snum3)+2, 2],0] = tempb3[i]


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
                prev_loss = 0
                for epoch in range(epochs):
                    running_loss = 0
                    val_loss = 0

                    print('epoch %d' % epoch, end=' ')

                    startTime = datetime.now()
                        
                    for i in range(train_batch_num):
                        data = trainset.sampleinput(i)
                        label = trainset.sampleground(i)
                        data = np.array([each[0] for each in data])
                        label = np.array([each[0] for each in label])

                        loss, output, rate, _ = sess.run([Dnet.loss, Dnet.output, Dnet.rate, Dnet.optimizer],
                                                         feed_dict={Dnet.inputs: data, Dnet.label: label})
                        running_loss = (running_loss * i + loss) / (i + 1)



                    trainset.shuffle()
                    endTime = datetime.now()

                    print('total loss: %.5f' % (running_loss), end=' ')
                    print("Time taken:", endTime - startTime, end='  ')
                    print("Learning rate: %e" % (rate * factor))

                    prev_loss = running_loss
                    
                    if epoch % 10 == 9 or epoch == 0:

                        print(' \n')
                        print('sample ground : ', end = ' ')
                        for k in range(output_shape):
                            print('%.4f, ' % label[0, k], end=' ')
                        print(' ')
                        
                        print('sample output : ', end = ' ')
                        for k in range(output_shape):
                            print('%.4f, ' % output[0, k], end=' ')
                        print(' ')
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







