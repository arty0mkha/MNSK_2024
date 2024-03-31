import numpy as np
from Neumann_Pressure import pure_Neumann_Poisson_solver

def get_dx(M, dx):
    """
    d_dx = (M[i,j+1] - M[i,j])/dx
    """
    return (np.roll(M, -1, axis=1) - M)/(2*dx)

def get_dy(M, dy):
    """
    d_dx = (M[i+1,j] - M[i,j])/dx
    """
    return (np.roll(M, -1, axis=0) - M)/(2*dy)

def get_velocities(Pressure, Mu, Width, dx, dy, u0):
    """
    (u,v) = W**2/(12*Mu)*grad(Pressure)
    """ 
    u = -Width**2/(12*Mu)*get_dx(Pressure, dx)
    v = -Width**2/(12*Mu)*get_dy(Pressure, dy)
    u[:, -1] = u0 # скорость втекающего потока?
    v[0] = 0 # непротекание?
    v[-1] = 0 # непротекание?
    return u, v

def get_viscosity(concentration, Mu1, Mu2):
    return (1 - concentration)*Mu1 + concentration*Mu2

def STI_solver(Nt,
               initial_concentrarion,
               Flux,
               initial_pressure,
               Nx, Ny,
               H, L,
               W0, v_in,
               mu1=0.1, mu2=0.001):
    """
    solver for: 

        1) d_dt (cW) + div (cW*velocity_vector) = inlet_flux_on_the boundary

        2) (u,v) = W**2/(12*Mu)*grad(Pressure)

        3) (W**3/(12*Mu) P_x)_x + (W**3/(12*Mu) P_y)_y = f

        4) Mu = mu1*c + mu2*(1 - c)

    Parameters:
    --------
    Nt:
        number of steps in time
    Nx, Ny :
        numbers mesh points
    H:
        height of space
    L:
        lenght of space
    W0:
        average width of space
    Flux:
        inlet flow on the boundary
    initial_pressure:

    initial_concentration:


    """
    
    dx = L/Nx
    dy = H/Ny
    times = np.zeros(Nt)

    W = W0 # np.random.uniform(low=W0*0.9, high=W0*1.1, size=(Ny, Nx))
    CFL = 0.2
    MU1 = mu1
    MU2 = mu2   
    Poisson_error = np.zeros(Nt)
    Poisson_error[0] = initial_pressure[1]
    Pressure_field = np.zeros(shape=(Nt, Ny+2, Nx+2))
    Pressure_field[0] = initial_pressure[0]

    concentration_field = np.zeros(shape=(Nt+1, Ny+2, Nx+2))
    concentration_field[0] = initial_concentrarion
    
    Vx = np.zeros(shape=(Nt, Ny+2, Nx+2))
    Vy = np.zeros(shape=(Nt, Ny+2, Nx+2))
    Vx[0], Vy[0] = get_velocities(Pressure_field[0],
                                  get_viscosity(concentration_field[0], MU1, MU2),
                                  W, dx, dy, v_in)

    

    for i in range(Nt-1):
        print(f"tick: {i+1}")
        dt = CFL*dx/max(np.max(np.abs(Vx[i])), np.max(np.abs(Vy[i])))
        times[i+1] = times[i] + dt
        
        viscosity_field = get_viscosity(concentration_field[i], MU1, MU2)

        divergence = get_dx(concentration_field[i]*W*Vx[i], dx) - get_dy(concentration_field[i]*W*Vy[i], dy)
        concentration_field[i+1] = concentration_field[i] + dt/W*(Flux - divergence) 
        concentration_field[i+1] = np.where(concentration_field[i+1]<0, 0, concentration_field[i+1])
        concentration_field[i+1] = np.where(concentration_field[i+1]>1, 1, concentration_field[i+1])
        
        Vx[i+1], Vy[i+1] = get_velocities(Pressure_field[i], viscosity_field, W, dx, dy, v_in)

        Q = v_in*viscosity_field.T[-1, 1:-1]*12/(W**2)
        A = W**3/(viscosity_field*12)
        f = np.zeros(shape=(Ny,Nx))
        top = np.zeros(Nx) 
        bottom= np.zeros(Nx) 
        left = Q*np.ones(Ny) 
        right = -Q*np.ones(Ny) 
        Pressure_field[i+1], Poisson_error[i+1] = pure_Neumann_Poisson_solver(A,
                                                          f,
                                                          top,
                                                          bottom,
                                                          left,
                                                          right,
                                                          Nx, Ny, L, H)

    return times, Pressure_field, Vx, Vy, concentration_field, Poisson_error

