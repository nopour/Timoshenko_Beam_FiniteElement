
pip install numpy matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

def form_stiffness_mass_timoshenko_beam(GDof, numberElements, elementNodes, numberNodes, xx, C, P, rho, I, thickness):
    stiffness = np.zeros((GDof, GDof))
    mass = np.zeros((GDof, GDof))
    force = np.zeros(GDof)

    gauss_locations_bending = [0.577350269189626, -0.577350269189626]
    gauss_weights_bending = [1, 1]

    for e in range(numberElements):
        indices = elementNodes[e, :]
        elementDof = np.concatenate([indices, indices + numberNodes])
        length_element = xx[indices[1]] - xx[indices[0]]
        detJacobian = length_element / 2
        invJacobian = 1.0 / detJacobian

        for q in range(len(gauss_weights_bending)):
            pt = gauss_locations_bending[q]
            shape, naturalDerivatives = shape_function_L2(pt)
            Xderivatives = naturalDerivatives * invJacobian

            B = np.zeros((2, 4))
            B[0, 2:] = Xderivatives

            stiffness[np.ix_(elementDof, elementDof)] += B.T @ (C[0, 0] * B) * gauss_weights_bending[q] * detJacobian
            force[indices] += shape * P * gauss_weights_bending[q] * detJacobian

            # Mass matrix contributions
            N = np.zeros((4, 4))
            N[:2, :2] = np.outer(shape, shape)
            N[2:, 2:] = np.outer(shape, shape)
            mass[np.ix_(elementDof, elementDof)] += rho * A * N * gauss_weights_bending[q] * detJacobian

    gauss_location_shear = 0.0
    gauss_weight_shear = 2.0

    for e in range(numberElements):
        indices = elementNodes[e, :]
        elementDof = np.concatenate([indices, indices + numberNodes])
        length_element = xx[indices[1]] - xx[indices[0]]
        detJacobian = length_element / 2

        pt = gauss_location_shear
        shape, naturalDerivatives = shape_function_L2(pt)
        Xderivatives = naturalDerivatives * invJacobian

        B = np.zeros((2, 4))
        B[1, :2] = Xderivatives
        B[1, 2:] = shape

        stiffness[np.ix_(elementDof, elementDof)] += B.T @ (C[1, 1] * B) * gauss_weight_shear * detJacobian

    return stiffness, force, mass

def shape_function_L2(xi):
    shape = np.array([(1 - xi) / 2, (1 + xi) / 2])
    naturalDerivatives = np.array([-0.5, 0.5])
    return shape, naturalDerivatives

def solution(GDof, prescribedDof, stiffness, force):
    activeDof = np.setdiff1d(np.arange(GDof), prescribedDof)

    K_active = stiffness[np.ix_(activeDof, activeDof)]
    F_active = force[activeDof]

    U_active = np.linalg.solve(K_active, F_active)
    displacements = np.zeros(GDof)
    displacements[activeDof] = U_active

    return displacements

def solution_modal(prescribedDof, K, M, num_modes):
    activeDof = np.setdiff1d(np.arange(len(K)), prescribedDof)

    K_active = K[np.ix_(activeDof, activeDof)]
    M_active = M[np.ix_(activeDof, activeDof)]

    eigvals, eigvecs = eigh(K_active, M_active, subset_by_index=[0, num_modes-1])

    eigvals = np.sqrt(np.real(eigvals))
    eigvecs_full = np.zeros((len(K), num_modes))
    eigvecs_full[activeDof, :] = eigvecs[:, :num_modes]

    return eigvals[:num_modes], eigvecs_full

def output_displacements_reactions(displacements, stiffness, GDof, prescribedDof):
    print("Displacements:")
    for i in range(GDof):
        print(f"{i + 1}: {displacements[i]}")

    F = stiffness @ displacements
    reactions = F[prescribedDof]
    print("Reactions:")
    for i, r in zip(prescribedDof, reactions):
        print(f"{i + 1}: {r}")
def plot_displacements(nodeCoordinates, displacements):
    plt.figure()
    plt.plot(nodeCoordinates, displacements)
    plt.xlabel('Node')
    plt.ylabel('Displacement')
    plt.title('Displacement of nodes')
    plt.grid(True)
    plt.show()

def plot_forces(nodeCoordinates, forces):
    plt.figure()
    plt.plot(nodeCoordinates, forces)
    plt.xlabel('Node')
    plt.ylabel('Force')
    plt.title('Forces at nodes')
    plt.grid(True)
    plt.show()

def plot_mode_shapes(nodeCoordinates, mode_shapes, num_modes):
    plt.figure()
    for i in range(num_modes):
        plt.subplot(num_modes, 1, i + 1)
        plt.plot(nodeCoordinates, mode_shapes[:, i])
        plt.grid(True)
        plt.ylabel(f'Mode {i + 1}')
    plt.xlabel('Node')
    plt.suptitle('Mode Shapes')
    plt.show()

def get_boundary_conditions(boundary_type, numberNodes):
    if boundary_type == 'c-c':
        fixedNodeW = [0, numberNodes - 1]
        fixedNodeTX = fixedNodeW
    elif boundary_type == 'c-s':
        fixedNodeW = [0]
        fixedNodeTX = [0, numberNodes - 1]
    elif boundary_type == 's-s':
        fixedNodeW = [0, numberNodes - 1]
        fixedNodeTX = []
    elif boundary_type == 'c-f':
        fixedNodeW = [0]
        fixedNodeTX = [0]
    else:
        raise ValueError("Invalid boundary condition type")
    prescribedDof = fixedNodeW + [node + numberNodes for node in fixedNodeTX]
    return prescribedDof

# Constants
E = 2.11e11
poisson = 0.30
rho = 7850
L = 1
b = 1
h = 0.1
I = b * h**3 / 12
kapa = 5 / 6
A = b * h
P = -1  # Uniform pressure
G = E / (2 * (1 + poisson))
# Constitutive matrix
C = np.array([[E * I, 0], [0, kapa * h * G]])
# Mesh
numberElements = 100
nodeCoordinates = np.linspace(0, L, numberElements + 1)
elementNodes = np.vstack([np.arange(numberElements), np.arange(1, numberElements + 1)]).T

# Generation of coordinates and connectivities
numberNodes = len(nodeCoordinates)
GDof = 2 * numberNodes

# Compute stiffness matrix and force vector
stiffness, force, mass = form_stiffness_mass_timoshenko_beam(GDof, numberElements, elementNodes, numberNodes, nodeCoordinates, C, P, rho, I, h)

# Choose boundary conditions
boundary_type = 'c-c'  # Change this to 'c-c', 'c-s', 's-s', or 'c-f' for different boundary conditions
prescribedDof = get_boundary_conditions(boundary_type, numberNodes)

# Solution for static analysis
displacements = solution(GDof, prescribedDof, stiffness, force)

# Output displacements/reactions
output_displacements_reactions(displacements, stiffness, GDof, prescribedDof)

# Max displacement
U = displacements[:numberNodes]
max_displacement = np.min(U)
print(f"Max displacement: {max_displacement}")
# Plot displacements
plot_displacements(nodeCoordinates, displacements[:numberNodes])
# Plot forces
F = stiffness @ displacements
plot_forces(nodeCoordinates, F[:numberNodes])
# Normal modes analysis
num_modes = 8
eigenvalues, mode_shapes = solution_modal(prescribedDof, stiffness, mass, num_modes)
# Print natural frequencies
frequencies = eigenvalues / (2 * np.pi)
print("Natural frequencies (Hz):")
print(frequencies)
# Plot mode shapes
plot_mode_shapes(nodeCoordinates, mode_shapes[:numberNodes, :], num_modes)

