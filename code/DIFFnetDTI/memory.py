from collections import deque
import numpy as np
import random

class Memory():
        def __init__(self, data_num, batch_size=100):
                self.data_num = data_num
                self.inputset = deque(maxlen=data_num)
                self.groundset = deque(maxlen=data_num)
                self.index = np.linspace(0, data_num - 1, data_num, dtype=int)
                self.batch_size = batch_size

        def inputadd(self, data):
                data = data.transpose((3,0,1,2))
                self.inputset.append(data)

        def groundadd(self, data):
                data = data.transpose((1, 0))
                self.groundset.append(data)

        def sampleinput(self, sequence):
                index = self.index[sequence * self.batch_size: min((sequence + 1) * self.batch_size, self.data_num)]
                # print(index)
                return np.array([self.inputset[i] for i in index])

        def sampleground(self, sequence):
                index = self.index[sequence * self.batch_size: min((sequence + 1) * self.batch_size, self.data_num)]
                # print(index)
                return np.array([self.groundset[i] for i in index])

        def shuffle(self):
                random.shuffle(self.index)