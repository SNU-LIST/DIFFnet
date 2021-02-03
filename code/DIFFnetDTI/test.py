import os
from scipy import io
from datetime import datetime
import math
import time
import argparse
from hyperparam import * 
from memory import *
from network import *
from projection2D import *

def main(arg):
    
    quantization = arg.qn
    epoch = arg.ep

    np.random.seed(678)
    tf.set_random_seed(5678)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(arg.GPU)
    trainable = False
    reuse = False
    keep_prob = tf.placeholder("float")
    snum=arg.sig
    model_path = "model path"+".ckpt"
    data_location = "data location"
    scheme_filename = "scheme file"
    scheme_data = np.array(io.loadmat(data_location + scheme_filename)['scheme'], dtype='float')
    batch_num = int(test_memory_size/test_batch_size)
    memory_size = test_memory_size
    batch_size = test_batch_size
    Dnetinvivo = network(quantization, output_shape, reuse, trainable, keep_prob, lr,  factor, chan,quantization ,train_data_num)

    print("Start In-vivo Test")

    for subject_num in range(subjectnum):
        startTime = datetime.now()
        subject_num = subject_num+1
        dwi_filename = 'dwi+file'
        mask_filename = 'mask_file'
        dwi_data = np.array(io.loadmat(data_location + dwi_filename)['temp'], dtype='float')
        mask_data = np.array(io.loadmat(data_location + mask_filename)['mask'], dtype='float')
        map_dimension = mask_data.shape
        total_voxel_num = int(np.sum(mask_data))
        save_name = 'save name'
        
        print(" ############################################################# ")
        print("  Subject number : ", subject_num)
        print("  Data dim : ", dwi_data.shape)
        print("  Mask dim : ", map_dimension)
        print("  Voxels : ", total_voxel_num)

        temp_result = np.zeros([4,map_dimension[0],map_dimension[1],map_dimension[2]])
        temp_processed_data = np.zeros([memory_size,36])
        temp_index = np.zeros([memory_size, 3])
        temp_data = np.zeros([memory_size, quantization, quantization, default_chan])
        scheme_vector = projection2D(scheme_data,quantization,snum)
        
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_path)
            cnt = 0

            for z_i in range(map_dimension[2]):
                temp_time = time.time()
                for y_i in range(map_dimension[1]):
                    for x_i in range(map_dimension[0]):
                        if mask_data[x_i,y_i,z_i] == 1:
                            tcnt = cnt % 10000
                            temp_array = dwi_data[x_i,y_i,z_i,:]
                            nomalize_factor = np.mean(temp_array[0:4])
                            temp_processed_data[tcnt,:] = (temp_array / nomalize_factor - 0.6647)/0.2721
                            temp_index[tcnt,:] = [x_i,y_i,z_i]
                            cnt = cnt + 1

                            if (cnt % memory_size == 0) + (cnt == total_voxel_num):
                                for i in range(snum):
                                    temp_data[:, scheme_vector[3 * i , 0], scheme_vector[3 * i , 1], scheme_vector[3 * i , 2]] = temp_processed_data[:,i + 4]
                                    temp_data[:, scheme_vector[3 * i +1, 0], scheme_vector[3 * i+1, 1], scheme_vector[3 * i +1, 2]] = temp_processed_data[:,i + 4]
                                    temp_data[:, scheme_vector[3 * i +2, 0], scheme_vector[3 * i + 2, 1], scheme_vector[3 * i + 2, 2]] = temp_processed_data[:,i + 4]
                                    temp_data[:, scheme_vector[3 * i + snum*3, 0], scheme_vector[3 * i + snum*3, 1], scheme_vector[3 * i + snum*3, 2]] = temp_processed_data[:,i + 4]
                                    temp_data[:, scheme_vector[3 * i + snum*3+1, 0], scheme_vector[3 * i + snum*3+1, 1], scheme_vector[3 * i + snum*3+1, 2]] = temp_processed_data[:,i + 4]
                                    temp_data[:, scheme_vector[3 * i + snum*3+2, 0], scheme_vector[3 * i + snum*3+2, 1], scheme_vector[3 * i + snum*3+2, 2]] = temp_processed_data[:,i + 4]

                                if cnt % memory_size == 0:

                                    for i in range(batch_num):
                                        data = temp_data[i * batch_size: (i + 1) * batch_size, :]
                                        index = np.array(temp_index[i * batch_size: (i + 1) * batch_size, :], dtype='int')
                                        temp_data_shape = index.shape
                                        temp_data_num = temp_data_shape[0]
                                        output = sess.run(Dnetinvivo.output, feed_dict={Dnetinvivo.inputs: data})

                                        for p in range(temp_data_num):
                                            temp_result[:,index[p,0],index[p,1],index[p,2]]        = output[p,:]


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



                                print("Time taken for slice :", time.time() - temp_time,' Sec')
            FA = temp_result[0,:,:,:]
            MD = temp_result[1,:,:,:]
            AD = temp_result[2,:,:,:]
            RD = temp_result[3,:,:,:]
            data={ 'FA' : FA , 'MD' : MD, "AD" : AD, "RD" : RD }

            io.savemat(save_name, data)

        print("\nTime taken:", datetime.now() - startTime, "sec\n\n #############################################################\n")
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Varying hyper parameter")
    parser.add_argument("--GPU" , type = int, default = Hyper_parameters['GPU'])
    parser.add_argument("--qn" , type = int, default = Hyper_parameters['qn'])
    parser.add_argument("--ep" , type = int, default = Hyper_parameters['epoch'])
    parser.add_argument("--sig", type = int, default = Hyper_parameters['signum'])
    arg = parser.parse_args()

    main(arg)
