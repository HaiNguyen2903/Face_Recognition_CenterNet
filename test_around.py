import matplotlib.pyplot as plt
import numpy as np

if '__name__' == '__main__':
    ar = np.random.rand(50, 1)
    plt.plot(ar, 'r')
    plt.show()
    plt.savefig('fig_saved.png')