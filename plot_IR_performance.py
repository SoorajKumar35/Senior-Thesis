import numpy as np
import matplotlib.pyplot as plt

path = '/Users/soorajkumar/Desktop/ECE496/performance_stats.txt'

num_channels = 8

all_performance = np.genfromtxt(path)
num_sets = int(all_performance.shape[0]//num_channels)

error_mat = np.zeros((num_channels, num_sets))

for i in range(num_sets):
    error_mat[:, i] = all_performance[i*num_channels:(i+1)*num_channels]

plt.figure(1)
plt.plot(error_mat[:, 0])
plt.plot(error_mat[:, 1])
plt.plot(error_mat[:, 2])
# plt.plot(error_mat[:, 3])
# plt.plot(error_mat[:, 4])
# plt.plot(error_mat[:, 5])
plt.legend(['100ms', '1s', '10s'])
# plt.plot(error_mat[:, 3])
plt.show()

pass

