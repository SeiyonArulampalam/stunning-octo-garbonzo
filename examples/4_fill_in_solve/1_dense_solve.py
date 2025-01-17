# do a dense matrix solve so we know what the right answer is
# and then I will try and solve the same problem in cusparse after suite sparse fill-in
import numpy as np
import matplotlib.pyplot as plt

A = np.array([
    [3, 1, 0, 1],
    [1, 2, -1, 0],
    [0, -1, 4, 0],
    [1, 0, 0, 3],
]).astype(np.double)
# plt.imshow(A)
# plt.show()

b = np.array([1, 2, 3, 4]).reshape((4,1)).astype(np.double)

x = np.linalg.solve(A, b)
print(f"{x=}")

# also what is the cholesky factorization
L = np.linalg.cholesky(A)
print(f"{L=}")