import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(np.matmul(A, B))
print(A@B)

print(np.linalg.norm(np.matmul(A, B) - A@B))