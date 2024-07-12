import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

def DQM(a, b, N):
    i = np.arange(1, N + 1)
    x = a + 0.5 * (b - a) * (1 - np.cos((i - 1) * np.pi / (N - 1)))  # Chebyshev-Gauss-Lobatto distribution
    R = np.tile(x, (N, 1))
    Q = R - R.T
    P = Q + np.eye(N)
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                A[i, j] = np.prod(P[i, :]) / np.prod(P[j, :]) / (x[i] - x[j])
            else:
                A[i, j] = np.sum(1.0 / P[i, :]) - 1.0
    return x, A

def vibration_Timoshenko(v, y, BC, N, nf, mode):
    alpha = 2 * (1.2 + v)
    zeta, A = DQM(0, 1, N)
    B = np.dot(A, A)
    I = np.eye(N)
    Z = np.zeros((N, N))
    
    k11 = B
    k12 = -A
    m11 = -alpha * (y ** 2) / 12 * I
    m12 = Z
    
    k21 = A
    k22 = alpha * (y ** 2) / 12 * B - I
    m21 = Z
    m22 = -alpha * (y ** 4) / 144 * I
    
    K = np.block([[k11, k12], [k21, k22]])
    M = np.block([[m11, m12], [m21, m22]])
    
    if BC[0] == 'c':
        T1 = np.hstack([I[0, :], Z[0, :]])
        T2 = np.hstack([Z[0, :], I[0, :]])
    elif BC[0] == 's':
        T1 = np.hstack([I[0, :], Z[0, :]])
        T2 = np.hstack([Z[0, :], A[0, :]])
    else:  # free
        T1 = np.hstack([Z[0, :], A[0, :]])
        T2 = np.hstack([A[0, :], -I[0, :]])
    
    if BC[1] == 'c':
        T3 = np.hstack([I[-1, :], Z[-1, :]])
        T4 = np.hstack([Z[-1, :], I[-1, :]])
    elif BC[1] == 's':
        T3 = np.hstack([I[-1, :], Z[-1, :]])
        T4 = np.hstack([Z[-1, :], A[-1, :]])
    else:  # free
        T3 = np.hstack([Z[-1, :], A[-1, :]])
        T4 = np.hstack([A[-1, :], -I[-1, :]])
    
    T = np.vstack([T1, T2, T3, T4])
    
    b = np.array([0, N - 1, N, 2 * N - 1])
    d = np.setdiff1d(np.arange(2 * N), b)
    
    K = np.delete(K, b, axis=0)
    Kb = K[:, b]
    Kd = K[:, d]
    
    M = np.delete(M, b, axis=0)
    Mb = M[:, b]
    Md = M[:, d]
    
    Tb = T[:, b]
    Td = T[:, d]
    
    p = -np.linalg.inv(Tb).dot(Td)
    
    Ks = Kd + Kb.dot(p)
    Ms = Md + Mb.dot(p)
    
    _, Lm4 = eig(Ks, Ms)
    Lm4 = np.diag(Lm4)
    
    Lm = np.power(Lm4, 0.25)
    
    e = np.imag(Lm) != 0
    Lm = Lm[~e]
    
    e = Lm < 0
    Lm = Lm[~e]
    
    Lm = np.sort(Lm)
    Lm = Lm[:nf]
    
    return Lm

def main():
    v = 0.3
    y = 0.2
    BC = ['cc', 'cs', 'ss', 'cf']
    lbc = len(BC)
    N = 50
    nf = 15
    mode = 0
    Lm = np.zeros((nf, lbc))
    
    for i in range(lbc):
        Lm[:, i] = vibration_Timoshenko(v, y, BC[i], N, nf, mode)
    
    print('  cc')
    print(Lm[:, 0])
    print('ss ')
    print(Lm[:, 2])
    print('CF ')
    print(Lm[:, 3])
if __name__ == "__main__":
    main()
