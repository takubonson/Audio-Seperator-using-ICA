import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from scipy.io.wavfile import read, write

rate = read("data/speechA1.wav")[0]
A1 = read("data/speechA1.wav")[1].astype("float64")
A2 = read("data/speechA2.wav")[1].astype("float64")
A1 -= np.mean(A1)
A2 -= np.mean(A2)
sample_num = len(A1)
X = np.matrix([A1, A2])
sum = np.matrix([[0., 0.], [0., 0.]])
for i in range(sample_num):
    x = X[:, i]
    sum += x@x.T
sigma = sum/sample_num
D, E = LA.eig(sigma)
root_D = np.diag(D**(-1/2))
V = E@root_D@(E.T)

w = np.matrix([[1.],[0.]])
size_w = LA.norm(np.asarray(w).flatten())
w /= size_w
ratio_a = 1000
ratio_b = 1000
while ((abs(abs(ratio_a)-1)>0.001) or (abs(abs(ratio_b)-1)>0.001)):
    sum = np.matrix([[0.],[0.]])
    for j in range(sample_num):
        x = X[:, j]
        z = V@x
        sum += z @ (LA.matrix_power((w.T @ z), 3))
    E__ = sum/sample_num
    new_w = E__ - 3*w
    new_size_w = LA.norm(np.asarray(new_w).flatten())
    new_w /= new_size_w
    ratio_a = float(new_w[0]/w[0])
    ratio_b = float(new_w[1]/w[1])
    w = new_w
w1 = w

w = np.matrix([[0.],[1.]])
size_w = LA.norm(np.asarray(w).flatten())
w /= size_w
ratio_a = 1000
ratio_b = 1000
while ((abs(abs(ratio_a)-1)>0.001) or (abs(abs(ratio_b)-1)>0.001)):
    sum = np.matrix([[0.],[0.]])
    for j in range(sample_num):
        x = X[:, j]
        z = V@x
        sum += z @ (LA.matrix_power((w.T @ z), 3))
    E__ = sum/sample_num
    new_w = E__ - 3*w
    new_size_w = LA.norm(np.asarray(new_w).flatten())
    new_w /= new_size_w
    ratio_a = float(new_w[0]/w[0])
    ratio_b = float(new_w[1]/w[1])
    w = new_w
w2 = w

y1 = []
y2 = []
for m in range(sample_num):
    x = X[:, m]
    z = V@x
    y1.append(float(w1.T@z))
    y2.append(float(w2.T@z))

y1 = np.array(y1)
y2 = np.array(y2)

write("person1.wav", rate, y1)
write("person2.wav", rate, y2)
print(w1)
print(w2)