import os
from scipy import io
from datetime import datetime
import math
import time
import argparse
import random
from hyperparam import * 
from memory import *
from network import *
from projection2D import *

#
# Description:
#  Evaluation code of DIFFnetNODDI
#
#  Copyright @ Juhyung Park
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : jack0878@snu.ac.kr
#

def main(arg):
    
    quantization = arg.qn
    epoch = arg.ep
    
    np.random.seed(random.randrange(0, int(1e4)))
    tf.set_random_seed(random.randrange(0, int(1e4)))
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(arg.GPU)
    trainable = False
    reuse = False
    keep_prob = tf.placeholder("float")
    model_path = "model path"+".ckpt"
    data_location = "data location"
    scheme_filename = "scheme file"
    schemedata = np.array(io.loadmat(data_location + scheme_filename)['temp'], dtype='float')
    b1 = schemedata[110:118, 0:3]
    b2 = schemedata[77:109, 0:3]
    b3 = schemedata[9:73,0:3]
    snum1 = 8
    snum2 = 32
    snum3 = 64
    scheme_vector = projection2D(b1,b2,b3,quantization,snum1,snum2,snum3)    
    batch_num = int(test_memory_size/test_batch_size)
    memory_size = test_memory_size
    batch_size = test_batch_size
    Dnetinvivo = network(quantization, output_shape, reuse, trainable, keep_prob, lr,  factor, chan,quantization ,train_data_num)

    
    print("Start In-vivo Test")


    for subject_num in range(subjectnum):
        startTime = datetime.now()
        subject_num = subject_num+1
        dwi_filename = 'dwi_file'
        mask_filename = 'mask_file'
        dwi_data = np.array(io.loadmat(data_location + dwi_filename)['temp'], dtype='float')
        mask_data = np.array(io.loadmat(data_location + mask_filename)['temp'], dtype='float')
        map_dimension = mask_data.shape
        total_voxel_num = int(np.sum(mask_data))
        save_name = 'save name'

        print(" =############################################################# ")
        print("  Subject number : ", subject_num)
        print("  Data dim : ", dwi_data.shape)
        print("  Mask dim : ", map_dimension)
        print("  Voxels : ", total_voxel_num)

        temp_result = np.zeros([3,map_dimension[0],map_dimension[1],map_dimension[2]])
        
        temp_processed_data = np.zeros([memory_size,119])
        temp_index = np.zeros([memory_size, 3])
        temp_data = np.zeros([memory_size, quantization, quantization, default_chan])

        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_path)

            cnt = 0

            for z_i in range(map_dimension[2]):
                for y_i in range(map_dimension[1]):
                    for x_i in range(map_dimension[0]):
                        if mask_data[x_i,y_i,z_i] == 1:
                            tcnt = cnt % 10000
                            temp_array = dwi_data[x_i,y_i,z_i,:]
                            nomalize_factor = np.mean(temp_array[0:9])
                            temp_processed_data[tcnt,:] = temp_array / nomalize_factor 
                            temp_index[tcnt,:] = [x_i,y_i,z_i]
                            cnt = cnt + 1

                            if (cnt % memory_size == 0) + (cnt == total_voxel_num):
                                tempb1 = (temp_processed_data[:,110:118] - 0.7022)/0.1159
                                tempb2 = (temp_processed_data[:,77:109] - 0.4944)/0.1601
                                tempb3 = (temp_processed_data[:,9:73] - 0.2676)/0.1516
                                for i in range(snum1):
                                    temp_data[:, scheme_vector[3 * i, 0], scheme_vector[3 * i, 1], scheme_vector[3 * i, 2]] = tempb1[:,i]
                                    temp_data[:, scheme_vector[3 * i + 1, 0],scheme_vector[3 * i + 1, 1],scheme_vector[3 * i + 1, 2]] = tempb1[:,i]
                                    temp_data[:, scheme_vector[3 * i + 2, 0],scheme_vector[3 * i + 2, 1],scheme_vector[3 * i + 2, 2]] = tempb1[:,i]
                                    temp_data[:, scheme_vector[3 * i + 3*(snum1+snum2+snum3), 0],scheme_vector[3 * i + 3*(snum1+snum2+snum3), 1],scheme_vector[3 * i + 3*(snum1+snum2+snum3), 2]] = tempb1[:,i]
                                    temp_data[:, scheme_vector[3 * i + 3*(snum1+snum2+snum3)+1, 0],scheme_vector[3 * i + 3*(snum1+snum2+snum3)+1, 1],scheme_vector[3 * i + 3*(snum1+snum2+snum3)+1, 2]] = tempb1[:,i]
                                    temp_data[:, scheme_vector[3 * i + 3*(snum1+snum2+snum3)+2, 0],scheme_vector[3 * i + 3*(snum1+snum2+snum3)+2, 1],scheme_vector[3 * i + 3*(snum1+snum2+snum3)+2, 2]] = tempb1[:,i]

                                for i in range(snum2):
                                    temp_data[:, scheme_vector[3 * i + 3*snum1, 0],scheme_vector[3 * i + 3*snum1, 1],scheme_vector[3 * i + 3*snum1, 2]] = tempb2[:,i]
                                    temp_data[:, scheme_vector[3 * i + 3*snum1+1, 0],scheme_vector[3 * i + 3*snum1+1, 1],scheme_vector[3 * i + 3*snum1+1, 2]] = tempb2[:,i]
                                    temp_data[:, scheme_vector[3 * i + 3*snum1+2, 0],scheme_vector[3 * i + 3*snum1+2, 1],scheme_vector[3 * i + 3*snum1+2, 2]] = tempb2[:,i]
                                    temp_data[:, scheme_vector[3 * i + 3*(2*snum1+snum2+snum3), 0],scheme_vector[3 * i + 3*(2*snum1+snum2+snum3), 1],scheme_vector[3 * i + 3*(2*snum1+snum2+snum3), 2]] = tempb2[:,i]
                                    temp_data[:, scheme_vector[3 * i + 3*(2*snum1+snum2+snum3)+1, 0],scheme_vector[3 * i + 3*(2*snum1+snum2+snum3)+1, 1],scheme_vector[3 * i + 3*(2*snum1+snum2+snum3)+1, 2]] = tempb2[:,i]
                                    temp_data[:, scheme_vector[3 * i + 3*(2*snum1+snum2+snum3)+2, 0],scheme_vector[3 * i + 3*(2*snum1+snum2+snum3)+2, 1],scheme_vector[3 * i + 3*(2*snum1+snum2+snum3)+2, 2]] = tempb2[:,i]

                                for i in range(snum3):
                                    temp_data[:, scheme_vector[3 * i + 3*(snum1+snum2), 0],scheme_vector[3 * i + 3*(snum1+snum2), 1],scheme_vector[3 * i + 3*(snum1+snum2), 2]] = tempb3[:,i]
                                    temp_data[:, scheme_vector[3 * i + 3*(snum1+snum2)+1, 0],scheme_vector[3 * i + 3*(snum1+snum2)+1, 1],scheme_vector[3 * i + 3*(snum1+snum2)+1, 2]] = tempb3[:,i]
                                    temp_data[:, scheme_vector[3 * i + 3*(snum1+snum2)+2, 0],scheme_vector[3 * i + 3*(snum1+snum2)+2, 1],scheme_vector[3 * i + 3*(snum1+snum2)+2, 2]] = tempb3[:,i]
                                    temp_data[:, scheme_vector[3 * i + 3*(2*snum1+2*snum2+snum3), 0],scheme_vector[3 * i + 3*(2*snum1+2*snum2+snum3), 1],scheme_vector[3 * i + 3*(2*snum1+2*snum2+snum3), 2]] = tempb3[:,i]
                                    temp_data[:, scheme_vector[3* i + 3*(2*snum1+2*snum2+snum3)+1, 0],scheme_vector[3 * i + 3*(2*snum1+2*snum2+snum3)+1, 1],scheme_vector[3 * i + 3*(2*snum1+2*snum2+snum3)+1, 2]] = tempb3[:,i]
                                    temp_data[:, scheme_vector[3* i + 3*(2*snum1+2*snum2+snum3)+2, 0],scheme_vector[3 * i + 3*(2*snum1+2*snum2+snum3)+2, 1],scheme_vector[3 * i + 3*(2*snum1+2*snum2+snum3)+2, 2]] = tempb3[:,i]
                                
                                if cnt % memory_size == 0:

                                    for i in range(batch_num):
                                        data = temp_data[i * batch_size: (i + 1) * batch_size, :]
                                        index = np.array(temp_index[i * batch_size: (i + 1) * batch_size, :], dtype='int')
                                        temp_data_shape = index.shape
                                        temp_data_num = temp_data_shape[0]
                                        output = sess.run(Dnetinvivo.output, feed_dict={Dnetinvivo.inputs: data})

                                        for p in range(temp_data_num):
                                            temp_result[:,index[p,0],index[p,1],index[p,2]]  = output[p,:]


                                else:
                                    remain_voxel = (cnt%memory_size)
                                    temp_batch_num = math.ceil(remain_voxel/batch_size)

                                    for i in range(temp_batch_num):
                                        data = temp_data[i * batch_size: min((i + 1) * batch_size, remain_voxel), :]
                                        index = np.array(temp_index[i * batch_size: min((i + 1) * batch_size, remain_voxel), :], dtype='int')
                                        temp_data_shape = index.shape
                                        temp_data_num = temp_data_shape[0]
                                        output = sess.run(Dnetinvivo.output,feed_dict={Dnetinvivo.inputs: data})

                                        for p in range(temp_data_num):
                                            temp_result[:,index[p,0],index[p,1],index[p,2]]        = output[p,:]

                
            ICVF = temp_result[0,:,:,:]
            ISOVF = temp_result[1,:,:,:]
            ODI = temp_result[2,:,:,:]
            data={ 'ICVF' : ICVF , 'ISOVF' : ISOVF, "ODI" : ODI }

            io.savemat(save_name, data)
        print("\nTime taken:", datetime.now() - startTime, "sec\n\n #############################################################\n")
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Varying hyper parameter")
    parser.add_argument("--GPU" , type = int, default = Hyper_parameters['GPU'])
    parser.add_argument("--qn" , type = int, default = Hyper_parameters['qn'])
    parser.add_argument("--ep" , type = int, default = Hyper_parameters['epoch'])
    arg = parser.parse_args()

    main(arg)
