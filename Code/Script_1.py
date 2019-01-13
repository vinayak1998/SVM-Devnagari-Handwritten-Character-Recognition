import sys
import os
import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import pandas as pd
from cvxopt import matrix
from numpy import array
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

part = sys.argv[1]

out = sys.argv[4]

Train = pd.read_csv(sys.argv[2], header=None)
Test = pd.read_csv(sys.argv[3], header=None)
y = Train[0].values
del Train[0]
X = Train.values
del Test[0]
X_test = Test.values

def gaussian_kernel(x, y, gamma):
    return np.exp((-linalg.norm(x-y)**2)*gamma)

X = X.astype(float)
X_test = X_test.astype(float)
y = y.astype(float)


C = sys.argv[5]
C = float(C)
m,n = X.shape
y = y.reshape(-1,1) * 1.
X_dash = y * X
if part == 'a':
  H = np.dot(X_dash , X_dash.T) * 1.
elif part == 'b':
  gamma = sys.arv[6]
  gamma = float(gamma)
  H = gaussian_kernel(x, y, gamma)
elif part == 'c':
  gamma = sys.argv[6]
  gamma = float(gamma)
  H = gaussian_kernel(x, y, gamma)

P = cvxopt_matrix(H)
q = cvxopt_matrix(-np.ones((m, 1)))
G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
A = cvxopt_matrix(y.reshape(1, -1))
b = cvxopt_matrix(np.zeros(1))

#Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])

w = ((y * alphas).T @ X).reshape(-1,1)
S = (alphas > 1e-4).flatten()
b = y[S] - np.dot(X[S], w)

y_pred=np.dot(X_test,w)+b[0]

np.savetxt(out,y_pred)



