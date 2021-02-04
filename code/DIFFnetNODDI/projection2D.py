import numpy as np
from scipy import io
from hyperparam import * 

def projection2D(b1_vector, b2_vector,b3_vector,quantization,snum1,snum2,snum3):     
    
        index_matrix = np.zeros([(snum1+snum2+snum3)*6, 3], dtype='int')
        b3_vector_inv = -1 * b3_vector
        b2_vector_inv = -1 * b2_vector
        b1_vector_inv = -1 * b1_vector
        for i in range(snum1):
            for j in range(3):
                x = j % 3
                y = (j + 1) % 3
                z = (j + 2) % 3
                x_projection = b1_vector[i, x]
                y_projection = b1_vector[i, y]
                z_projection = b1_vector[i, z]
                index_matrix[i * 3 + j, :] = [round((x_projection + 1) / 2 * quantization + 0.5) - 1, round((y_projection + 1) / 2 * quantization + 0.5) - 1, z]

        for i in range(snum2):
            for j in range(3):
                x = j % 3
                y = (j + 1) % 3
                z = (j + 2) % 3
                x_projection = b2_vector[i, x]
                y_projection = b2_vector[i, y]
                z_projection = b2_vector[i, z]
                index_matrix[i * 3 + j + 3*snum1, :] = [round((x_projection + 1) / 2 * quantization + 0.5) - 1, round((y_projection + 1) / 2 * quantization + 0.5) - 1, z +3]

        for i in range(snum3):
            for j in range(3):
                x = j % 3
                y = (j + 1) % 3
                z = (j + 2) % 3
                x_projection = b3_vector[i, x]
                y_projection = b3_vector[i, y]
                z_projection = b3_vector[i, z]
                index_matrix[i * 3 + j + 3*(snum1+snum2), :] = [round((x_projection + 1) / 2 * quantization + 0.5) - 1, round((y_projection + 1) / 2 * quantization + 0.5) - 1, z + 6]

        for i in range(snum1):
            for j in range(3):
                x = j % 3
                y = (j + 1) % 3
                z = (j + 2) % 3
                x_projection = b1_vector_inv[i, x]
                y_projection = b1_vector_inv[i, y]
                index_matrix[i * 3 + j + 3*(snum1+snum2+snum3), :] = [round((x_projection + 1) / 2 * quantization + 0.5) - 1,
                                                    round((y_projection + 1) / 2 * quantization + 0.5) - 1, z]

        for i in range(snum2):
            for j in range(3):
                x = j % 3
                y = (j + 1) % 3
                z = (j + 2) % 3
                x_projection = b2_vector_inv[i, x]
                y_projection = b2_vector_inv[i, y]
                index_matrix[i * 3 + j + 3*(2*snum1+snum2+snum3), :] = [round((x_projection + 1) / 2 * quantization + 0.5) - 1,
                                                         round((y_projection + 1) / 2 * quantization + 0.5) - 1, z +3]

        for i in range(snum3):
            for j in range(3):
                x = j % 3
                y = (j + 1) % 3
                z = (j + 2) % 3
                x_projection = b3_vector_inv[i, x]
                y_projection = b3_vector_inv[i, y]
                index_matrix[i * 3 + j + 3*(2*snum1+2*snum2+snum3), :] = [round((x_projection + 1) / 2 * quantization + 0.5) - 1,
                                                          round((y_projection + 1) / 2 * quantization + 0.5) - 1, z + 6]
                
        return index_matrix