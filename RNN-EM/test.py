import numpy as np

x = np.array([[1, 1, 1]])
with open('test.txt', 'w') as handle:
    handle.write('t: ')
    np.savetxt(handle, x, fmt='%i')
    handle.write('p: ')
    np.savetxt(handle, x, fmt='%i')