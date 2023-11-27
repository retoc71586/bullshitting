import numpy as np

a = np.array([2, 3, 4])
b = np.array([5, 6, 7])
c = a

print(np.allclose(a, b))    
print(np.allclose(a, c))    