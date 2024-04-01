import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

def make_pure_Neumann_matrix(Nx, Ny, A, dx=1, dy=1):
    """
    (A P_x)_x + (A P_y)_y = f
    
    matrix shape Nx, Ny - включая "бонусные" +2
    Nx-2 + Nx*(Ny-2) + Nx-2 = Nx*Ny-4
    top                bottom
    +1 line
    Сохдание матрицы коэффициентов для решения уравнения Пуассона
    """
    Matrix = np.zeros(shape = (Nx*Ny-4, Nx*Ny-4))
    
    
    # BC top and bottom
    for i in range(Nx-2):
        # top
        Matrix[i, i] = 1/dy
        Matrix[i, i + Nx-1] = -1/dy
        # bottom
        Matrix[-1 - i, -1 - i] = 1/dy
        Matrix[-1 - i, -1 - (i + Nx-1)] = -1/dy
    
    # BC left and right
    for i in range(Ny-2):
        # left
        Matrix[(Nx-2) + i*Nx, (Nx-2) + i*Nx] = 1/dx 
        Matrix[(Nx-2) + i*Nx, (Nx-2) + i*Nx + 1] = -1/dx
        # right
        Matrix[-1 - ((Nx-2) + i*Nx), -1 - ((Nx-2) + i*Nx)] = 1/dx 
        Matrix[-1 - ((Nx-2) + i*Nx), -1 - ((Nx-2) + i*Nx + 1)] = -1/dx

    # Poisson_top 
    # i=1, j=1, ..., Nx-2
    for j in range(1, Nx-1):
        Matrix[ Nx-2+j, Nx-2+j] = -(A[1,j] + A[1,j+1])/(dx**2) - 2*A[1,j]/(dy**2) # i, j ячейка матрицы давления
        Matrix[ Nx-2+j, Nx-2 + j +1] = A[1,j+1]/(dx**2) # i, j+1
        Matrix[ Nx-2+j, Nx-2 + j -1] = A[1,j]/(dx**2) # i, j-1
        Matrix[ Nx-2+j, Nx-2 + j + (Nx)] = A[1, j]/(dy**2) # i+1, j
        Matrix[ Nx-2+j, Nx-2 + j - (Nx-1)] = A[1,j]/(dy**2) # i-1, j

    # Poisson_bottom
    # i=Ny-2, j =1, ..., Nx-2 
    for j in range(1, Nx-1):
        Matrix[ -1 -(Nx-2+j), -1 -(Nx-2+j)] = -(A[Ny-2,j] + A[Ny-2,j+1])/(dx**2) - 2*A[Ny-2,j]/(dy**2) # i, j ячейка матрицы давления
        Matrix[ -1 -(Nx-2+j), -1 -(Nx-2+j) +1] = (2*A[Ny-2,j+1]-A[Ny-2,j])/(dx**2) # i, j+1
        Matrix[ -1 -(Nx-2+j), -1 -(Nx-2+j) -1] = A[Ny-2,j]/(dx**2) # i, j-1
        Matrix[ -1 -(Nx-2+j), -1 -(Nx-2+j) + (Nx-1)] = A[Ny-2, j]/(dy**2) # i+1, j
        Matrix[ -1 -(Nx-2+j), -1 -(Nx-2+j) - (Nx)] = A[Ny-2,j]/(dy**2) # i-1, j

    # Poisson central
    for i in range(2, Ny - 2): # номер блока
        for j in range(1, Nx - 1): # номер строки в блоке      
            Matrix[ i*Nx + j -2, i*Nx + j -2] = -(A[i-1,j] + A[i-1,j+1])/(dx**2) - (A[i-1,j] + A[i-1,j+1])/(dy**2) # i, j ячейка матрицы давления
            Matrix[ i*Nx + j -2, i*Nx + j + 1 -2] = A[i-1,j+1]/(dx**2) # i, j+1
            Matrix[ i*Nx + j -2, i*Nx + j - 1 -2] = A[i-1,j]/(dx**2) # i, j-1
            Matrix[ i*Nx + j -2, i*Nx + j + Nx -2 ] = A[i+1-1, j]/(dy**2) # i+1, j
            Matrix[ i*Nx + j -2, i*Nx + j - Nx -2] = A[i-1,j]/(dy**2) # i-1, j
    return Matrix

def decomposition (vector, basis_matrix, values):
    """
    A*x_vec = y_vec
    x_vec = summ_k a_k*eigen_vec_k
    => summ_k a_k*eigen_value_k*vec_k = y_vec
    y_vec = summ_k b_k*vec_k
    => a_k = b_k/eigen_value_k
    """
    matrix = np.zeros(shape=basis_matrix.shape, dtype='complex128')
    for i in range(matrix.shape[1]):
        matrix.T[i] = basis_matrix.T[i]*values[i]
    coefficients = np.linalg.pinv(matrix)@vector
    return coefficients


def pure_Neumann_Poisson_solver(A,  f, BC_top, BC_bottom, BC_left, BC_right, Nx=128, Ny=128, L=1, H=1, Plot=False):
    """
    (A P_x)_x + (A P_y)_y = f

    P_x|x=0 = -Q0
    P_x|x=L = Q0
    P_y|y=0 = 0
    P_y|y=H = 0

    ->
    matrix @ pressure_vec = [f + BC + pressure[1,1]=0]_vec 


    Parameters
    ---------
    BC_top:
        P_y|y=H = 0
    BC_bottom:
        P_y|y=0 = 0
    BC_left:
        P_x|x=0 = -Q
    BC_right:
        P_x|x=L = Q
    """
    DX = L/Nx
    DY = H/Ny
    
    Pressure_matrix = np.zeros(shape=(Ny+2, Nx+2))
    matrix = make_pure_Neumann_matrix(Nx+2, Ny+2, A, DX, DY) # shape (Nx+2)*(Ny+2) - 4 , Nx+2)*(Ny+2) - 4 
     
    right_side_matrix = np.zeros(shape=(Ny+2, Nx+2))
    right_side_matrix[1:-1, 1:-1] = f
    right_side_matrix[0, 1:-1] = BC_top
    right_side_matrix[-1, 1:-1] = BC_bottom
    right_side_matrix.T[0, 1:-1] = BC_left
    right_side_matrix.T[-1, 1:-1] = BC_right

    right_side_vec = right_side_matrix.reshape(-1) # shape (Nx+2)*(Ny+2)
    new_right_side_vec = np.delete(right_side_vec, [1, Nx-1, -Nx, -1]) # f+BC    \ shape (Nx+2)*(Ny+2) - 4 +1
    print('start decomposition')
    evals, evecs = np.linalg.eig(matrix)
    coeffs = decomposition(new_right_side_vec, evecs, evals)
    print('end decomposition')
    print('start matmul')
    pressure_vec = jnp.matmul(evecs, coeffs)
    error_MSP = np.mean(np.square(np.abs(1-matrix@pressure_vec/np.where(new_right_side_vec==0, 1, new_right_side_vec))))
    Pressure_matrix[0, 1:-1] = pressure_vec[:Nx]
    Pressure_matrix[1:-1] = pressure_vec[Nx: -(Nx)].reshape(Ny, Nx+2)
    Pressure_matrix[-1, 1:-1] = pressure_vec[-(Nx):]

    Pressure_matrix[0,0] = Pressure_matrix[0,1] + Pressure_matrix[1,0] -Pressure_matrix[1,1] 
    Pressure_matrix[0,-1] = Pressure_matrix[0,-2] + Pressure_matrix[1,-1] -Pressure_matrix[1,-2]
    Pressure_matrix[-1,0] = Pressure_matrix[-1,1] + Pressure_matrix[-2,0] -Pressure_matrix[-1,1] 
    Pressure_matrix[-1,-1] = Pressure_matrix[-2,-1] + Pressure_matrix[-1,-2] -Pressure_matrix[-2,-2]  
    if Plot or (error_MSP>1.2):
        fig = plt.figure(figsize=(12,4))
        plt.subplot(121)
        plt.plot(np.real(matrix@(pressure_vec)), label='Расчетная правая часть')
        plt.plot(new_right_side_vec, label='Изначальная правая часть')
        plt.legend()
        plt.subplot(122)
        plt.semilogy(np.abs(1-matrix@pressure_vec/np.where(new_right_side_vec==0, 1, new_right_side_vec)))
        plt.ylabel("MSPE %")
        plt.title('error')
        plt.savefig('maybe something went wrong.png')
        plt.show()
    return Pressure_matrix, error_MSP
    



