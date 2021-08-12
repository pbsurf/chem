import numpy as np
from time import time

if 1:
  np.random.seed(0)

  size = 1024
  A, B = np.random.random((size, size)), np.random.random((size, size))
  C, D = np.random.random((size * 128,)), np.random.random((size * 128,))
  E = np.random.random((int(size / 2), int(size / 4)))
  F = np.random.random((int(size / 2), int(size / 2)))
  F = np.dot(F, F.T)
  G = np.random.random((int(size / 2), int(size / 2)))

  # Matrix multiplication
  N = 20
  t = time()
  for i in range(N):
      np.dot(A, B)
  delta = time() - t
  print('Dotted two %dx%d matrices in %0.6f s.' % (size, size, delta / N))
  del A, B

  # Vector multiplication
  N = 5000
  t = time()
  for i in range(N):
      np.dot(C, D)
  delta = time() - t
  print('Dotted two vectors of length %d in %0.6f ms.' % (size * 128, 1e3 * delta / N))
  del C, D

  # Singular Value Decomposition (SVD)
  N = 3
  t = time()
  for i in range(N):
      np.linalg.svd(E, full_matrices = False)
  delta = time() - t
  print("SVD of a %dx%d matrix in %0.6f s." % (size / 2, size / 4, delta / N))
  del E

  # Cholesky Decomposition
  N = 3
  t = time()
  for i in range(N):
      np.linalg.cholesky(F)
  delta = time() - t
  print("Cholesky decomposition of a %dx%d matrix in %0.6f s." % (size / 2, size / 2, delta / N))

  # Eigendecomposition
  t = time()
  for i in range(N):
      np.linalg.eig(G)
  delta = time() - t
  print("Eigendecomposition of a %dx%d matrix in %0.6f s." % (size / 2, size / 2, delta / N))


if 1:
  M = 500;
  c = np.empty((M, M));
  for i in range(1,M+1):
      for j in range(1,M+1):
          c[i-1][j-1] = np.sin(i + np.power(j, 2.))
  t0 = time();
  g = np.linalg.eigvals(c);
  t1 = time();
  for ii in range(100):
    gg = np.sqrt(np.exp(c*c + c - 1.0))
  t2 = time()
  print("eigvals: %f; mult: %f" %(t1 - t0, t2 - t1))
  #m = np.max(np.real(g));
  #print("m = ", m)

  # np config
  #np.__config__.show()
