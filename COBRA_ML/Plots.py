import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

for n in range (1,11,1):
    plt.close('all')
    data=np.genfromtxt('CW32_Test{}_impactenergy.csv'.format(n), delimeter=',')
    plt.pcolormesh(data)
    plt.colorbar()
    plt.savefig('CW32_Test{}_impactenergy.png'.format(n))
