from matplotlib import pyplot as plt
import seaborn as sns
import sys
sys.path.append("..")
from data_gen import open_file

units, times, labels = open_file("dataset/train.h5")

# data_idx = 7000     #0
# data_idx = 13000    #1
# data_idx = 43000    #2
data_idx = 0        #3
# data_idx = 31000    #4
# data_idx = 25000    #5
# data_idx = 55000    #6
# data_idx = 37000    #7
# data_idx = 19000    #8
# data_idx = 49000    #9

print(labels)
print("Label:", labels[data_idx])

x_max = 34
y_max = 34
xaddr = units[data_idx] % x_max
yaddr = (units[data_idx] // x_max) % y_max
pol = units[data_idx] // (x_max*y_max)
timestamps = times[data_idx]


fig = plt.figure(figsize=(15, 6))
ax = fig.add_subplot(111, projection='3d')

for i, color in enumerate([[1, 0, 0], [0, 0, 1]]):
    idx = (pol == i)
    ax.scatter(timestamps[idx], xaddr[idx], yaddr[idx], color=color, marker='.')

ax.set_xlabel('Time (s)')
ax.set_ylabel('X')
ax.set_zlabel('Y')

ax.elev = -175
ax.azim = -80

plt.title("N-MNIST: Digit {:d}".format(labels[data_idx]))

plt.tight_layout()
plt.show()
