import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def main():
    ccc1 = ["#c1272d", "#0000a7", "#eecc16", "#008176", "b3b3b3"]
    
    E = 10e7
    nu = 0.30
    rho = 1

    L = 1
    b = 1
    h = 0.001
    I = b * h**3 / 12
    kapa = 5 / 6
    A = b * h

    G = E / 2 / (1 + nu)
    C = np.array([[E * I, 0], [0, kapa * h * G]])
    
    N_elements_range = range(5, 102)
    N_modes = 5

    D = np.zeros((len(N_elements_range), N_modes, 3))

    for j, N_nodes in enumerate(N_elements_range):
        x_col = np.linspace(0, L, N_nodes)
        N_elements = N_nodes - 1
        elementNodes = np.vstack((np.arange(1, N_nodes), np.arange(2, N_nodes + 1))).T

        P = -1
        GDof = 2 * N_nodes

        K_Assembly, F_equiv, M_Assembly = formStiffnessMassTimoshenkoBeam(GDof, elementNodes, x_col, C, P, rho, I, h)

        prescribedDof = [
            np.array([0, N_nodes - 1, N_nodes, 2 * N_nodes - 1]), 
            np.array([0, N_nodes - 1]),                            
            np.array([0, N_nodes])                                
        ]

        for ii in range(len(prescribedDof)):
            D_vec = np.zeros(GDof)
            D_vec[prescribedDof[ii]] = 0
            F_vec = np.zeros(GDof)

            D_vec, F_vec = solution(prescribedDof[ii], K_Assembly, D_vec, F_vec, F_equiv)
            print("Max displacement")
            print(np.min(D_vec[:N_nodes]))

            D_modeShapes, w_n = solutionModal(prescribedDof[ii], D_vec[prescribedDof[ii]], K_Assembly, M_Assembly, N_modes)
            D1 = w_n * L * L * np.sqrt(rho * A / (E * I))
            D[j, :, ii] = np.sort(D1)

    plt.figure()
    plt.plot(N_elements_range, D[:, 0, 0], '-o', markevery=10, color=ccc1[0], linewidth=1.2, label='Clamped-Clamped')
    plt.plot(N_elements_range, D[:, 0, 1], '-.v', markevery=10, color=ccc1[2], linewidth=1.2, label='Simply supported-Simply supported')
    plt.plot(N_elements_range, D[:, 0, 2], '--s', markevery=10, color=ccc1[3], linewidth=1.2, label='Clamped-Free')
    plt.legend(loc='best')
    plt.grid(which='minor')
    plt.grid(which='major')
    plt.box(on=True)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Frequency, [-]')
    plt.title('Frequency Convergence')
    plt.xlim([0, 100])
    plt.show()

def formStiffnessMassTimoshenkoBeam(GDof, elementNodes, x_col, C, P, rho, I, h):
    N_elements = elementNodes.shape[0]
    N_Nodes = len(x_col)

    K_Assembly = np.zeros((GDof, GDof))
    M_Assembly = np.zeros((GDof, GDof))
    F_equiv_local = np.zeros(GDof)

    gaussLocations = np.array([0.577350269189626, -0.577350269189626])
    gaussWeights = np.ones(2)

    for iElement in range(N_elements):
        i_nodes = elementNodes[iElement] - 1
        elementDof = np.hstack([i_nodes, i_nodes + N_Nodes])
        indiceMass = i_nodes + N_Nodes
        ndof = len(i_nodes)
        Le = x_col[i_nodes[1]] - x_col[i_nodes[0]]
        detJacobian = Le / 2
        invJacobian = 1 / detJacobian
        for q in range(len(gaussWeights)):
            pt = gaussLocations[q]
            shape, naturalDerivatives = shapeFunctionL2(pt)
            Xderivatives = naturalDerivatives * invJacobian
            
            B = np.zeros((2, 2 * ndof))
            B[0, ndof:2 * ndof] = Xderivatives
            
            K_Assembly[np.ix_(elementDof, elementDof)] += B.T @ B * gaussWeights[q] * detJacobian * C[0, 0]
            F_equiv_local[i_nodes] += shape * P * detJacobian * gaussWeights[q]

            M_Assembly[np.ix_(indiceMass, indiceMass)] += shape[:, np.newaxis] @ shape[np.newaxis, :] * gaussWeights[q] * I * rho * detJacobian
            M_Assembly[np.ix_(i_nodes, i_nodes)] += shape[:, np.newaxis] @ shape[np.newaxis, :] * gaussWeights[q] * h * rho * detJacobian

    gaussLocations = np.array([0.0])
    gaussWeights = np.array([2.0])
    for iElement in range(N_elements):
        i_nodes = elementNodes[iElement] - 1
        elementDof = np.hstack([i_nodes, i_nodes + N_Nodes])
        ndof = len(i_nodes)
        Le = x_col[i_nodes[1]] - x_col[i_nodes[0]]
        detJacobian = Le / 2
        invJacobian = 1 / detJacobian
        for q in range(len(gaussWeights)):
            pt = gaussLocations[q]
            shape, naturalDerivatives = shapeFunctionL2(pt)
            Xderivatives = naturalDerivatives * invJacobian

            B = np.zeros((2, 2 * ndof))
            B[1, :ndof] = Xderivatives
            B[1, ndof:2 * ndof] = shape
            
            K_Assembly[np.ix_(elementDof, elementDof)] += B.T @ B * gaussWeights[q] * detJacobian * C[1, 1]
    
    return K_Assembly, F_equiv_local, M_Assembly

def shapeFunctionL2(xi):
    shape = np.array([(1 - xi) / 2, (1 + xi) / 2])
    naturalDerivatives = np.array([-1, 1]) / 2
    return shape, naturalDerivatives

def solution(prescribedDof, K_assembly, D_vec, F_vec, F_eq_vec=None):
    if F_eq_vec is None:
        F_eq_vec = np.zeros_like(F_vec)

    GDof = len(D_vec)
    freeDof = np.setdiff1d(np.arange(GDof), prescribedDof)

    D_vec[freeDof] = np.linalg.solve(K_assembly[np.ix_(freeDof, freeDof)], F_vec[freeDof] + F_eq_vec[freeDof])

    nonZeroDof = np.union1d(freeDof, np.array([d for d in prescribedDof if D_vec[d] != 0], dtype=int))

    F_vec[prescribedDof] = K_assembly[np.ix_(prescribedDof, nonZeroDof)] @ D_vec[nonZeroDof] - F_eq_vec[prescribedDof]
    return D_vec, F_vec

def solutionModal(prescribedDofs, D_prescribed, K_assembly, M_assembly, N_modes):
    GDof = K_assembly.shape[0]
    freeDof = np.setdiff1d(np.arange(GDof), prescribedDofs)

    D_modeShape_cols = np.zeros((GDof, N_modes))
    D_modeShape_cols[prescribedDofs, :] = D_prescribed[:, np.newaxis]

    eigvals, eigvecs = scipy.linalg.eigh(K_assembly[np.ix_(freeDof, freeDof)], M_assembly[np.ix_(freeDof, freeDof)])
    w_n_vec = np.sqrt(eigvals[:N_modes])
    D_modeShape_cols[freeDof, :] = eigvecs[:, :N_modes]
    
    return D_modeShape_cols, w_n_vec

if __name__ == "__main__":
    main()
