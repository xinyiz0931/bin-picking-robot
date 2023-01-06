import numpy as np
F = np.loadtxt("./out.txt")
print(F.shape)
print(np.mean(F, axis=0)[1:].tolist())