import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("dos.dat")[::-1,1:]


plt.imshow(data, aspect="auto", cmap="bwr", vmax=1, vmin=-1)
plt.colorbar()
plt.savefig("dos.png")