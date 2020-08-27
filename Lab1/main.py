import numpy as np
import matplotlib.pyplot as plt
n = 100
mA = [0.0,0.5]
mB = [-0.5, 0.0]

sigmaA = 0.5
sigmaB = 0.5
#set1
set_a1 = np.random.rand(1,n) * sigmaA + mA[0]
set_a2 = np.random.rand(1,n) * sigmaA + mA[1]
plt.scatter(set_a1,set_a2)

#set2
set_b1 = np.random.rand(1,n) * sigmaB + mB[0]
set_b2 = np.random.rand(1,n) * sigmaB + mB[1]
plt.scatter(set_b1,set_b2)

plt.show()
