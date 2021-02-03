import numpy as np
from scipy import io
from hyperparam import * 

def projection2D(scheme_data,quantization,snum):     
        index_matrix = np.zeros([snum*6, 3], dtype='int')
        vector = scheme_data[:, 0:3]
        vector_inv = -1 * scheme_data[:, 0:3]

        for i in range(snum):
            for j in range(3):
                x = j % 3
                y = (j + 1) % 3
                z = (j + 2) % 3
                x_projection = vector[i, x]
                y_projection = vector[i, y]
                index_matrix[i * 3 + j , :] = [round((x_projection + 1) / 2 * quantization + 0.5) - 1,round((y_projection + 1) / 2 * quantization + 0.5) - 1, z ]


        for i in range(snum):
            for j in range(3):
                x = j % 3
                y = (j + 1) % 3
                z = (j + 2) % 3
                x_projection = vector_inv[i, x]
                y_projection = vector_inv[i, y]
                index_matrix[i * 3 + j + snum*3, :] = [round((x_projection + 1) / 2 * quantization + 0.5) - 1, round((y_projection + 1) / 2 * quantization + 0.5) - 1, z ]

        return index_matrix