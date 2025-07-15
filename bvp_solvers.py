#! /usr/bin/env python3


# -----------------------------------------------------------------------------
# File: bvp_solvers.py
# Description: Planetary regolith thermal model, including Hapke's 1996 radiative transfer 2-stream approximation. .
# Author: Andrew J. Ryan, 2025
#
# This code is free to use, modify, and distribute for any purpose.
# Please contact the author (ajryan4@arizona.edu) to discuss applications or if you
# use this code in your research or projects.
# -----------------------------------------------------------------------------

#Note that these boundary problems are specifically designed for the Hapke RTE model.
# They are not general-purpose solvers.

def solve_bvp_vis(x, Fun, u, J, A,h0,hN,Q,T,tol=1e-8, max_iter=50):
    # Boundary value solver specifically for the visible RTE
    # Also could be used for thermal RTE in single-layer scenario. 
    # Boundary conditions are baked in! 
    # Solve u'' = Fun(u), 0<=x<=L, with bundary conditions:
    #   u(0) = 0.5 u'(0),   u(L) = -0.5 u'(L)
    # via finite differences + Newton's method,
    # Specifically designed for Hapke RTE, single-layer scenario or vis RTE. 
    # x  : node positions, normally x_RTE in main code. 
    # Fun: function F(u)
    # u  : initial guess for u
    # J  : Jacobian
    # A  : central difference stencil (3 elements)
    # h0 : Thickness of top layer
    # hN : Thickness of bottom layer. 

    import numpy as np
    import scipy.linalg as la
    N = len(x)-1
    A1,A2,A3 = A
    # --- build residual vector R ---
    R = np.zeros(N+1)
    for it in range(max_iter):
        # Downards stream = 0
        R[0] = (2*h0 + 1)*u[0] - u[1] - 2*h0*Q
        # interior i=1..N-1
        R[1:N] =  A1*u[0:N-1] + A2*u[1:N] + A3*u[2:N+1] - Fun(x[1:-1],u[1:-1],T[2:N+1])
        # Upwards stream = 0
        R[N] = -u[N-1] + (2*hN + 1)*u[N]

        normR = np.max(np.abs(R))
        if normR < tol:
            #print(f"Converged in {it} iterations, ||R||∞={normR:.2e}")
            break

        # --- solve for update du ---
        #Matrix version
        #du = np.linalg.solve(J, -R)
        #Banded version
        du = la.solve_banded((1,1),J,-R)

        # --- update solution ---
        u += du
    else:
        print("Warning: solve_bvp_1 exceeded max_iter")

    return u

def solve_bvp_therm(x, Fun, u, J, A,h0,hN,D,Q,T,single_layer = True,tol=1e-8, max_iter=50):
    # Boundary value solver specifically for the thermal RTE in the two-layer scenario. 
    # Boundary conditions are baked in!   
    # Solve u'' = F(u), 0<=x<=L, with boundary conditions:
    #   u(0) = 0.5 u'(0),   u(L) = u(N) + D 
    # via finite differences + Newton's method,
    # Specifically designed for two layer thermal RTE. 
    # x  : node positions, normally x_RTE in main code. 
    # Fun: function F(u)
    # J  : Jacobian
    # A  : central difference stencil (3 elements)
    # h0 : Thickness of top layer
    # hN : Thickness of bottom layer. 
    # D  : Constant term for lower boundary condition. 
    
    import numpy as np
    import scipy.linalg as la
    N = len(x)-1
    A1,A2,A3 = A
    # --- build residual vector R ---    
    R = np.zeros(N+1)

    for it in range(max_iter):        
        # boundary at i=0:  (2h+1) u0 - u1 = 0
        R[0] = (2*h0 + 1)*u[0] - u[1] - 2*h0*Q
        # interior i=1..N-1
        R[1:N] =  A1*u[0:N-1] + A2*u[1:N] + A3*u[2:N+1] - Fun(x[1:-1],u[1:-1],T[2:N+1])
        # boundary at i=N: -u[N-1] + (2h+1) u[N] = 0
        if single_layer:
            # phi_therm = sigma/np.pi * T[-1]**4
            # Equivalent to both streams being equal at the bottom of an optically thick medium
            R[N] = u[N] - D
        else:
            # Upwards stream is equal to sigma/np.pi * T[-1]**4. Downward stream is solved for.
            R[N] = -u[N-1] + (2*hN + 1)*u[N] - 2*hN*D

        normR = np.max(np.abs(R))
        if normR < tol:
            #print(f"Converged in {it} iterations, ||R||∞={normR:.2e}")
            break

        # --- solve for update du ---
        #Matrix version
        #du = np.linalg.solve(J, -R)
        #Banded version
        du = la.solve_banded((1,1),J,-R)

        # --- update solution ---
        u += du
    else:
        print("Warning: solve_bvp_2layer exceeded max_iter")

    return u
