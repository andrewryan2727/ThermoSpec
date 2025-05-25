import numpy as np

# -----------------------------------------------------------------------------
# File: stencils.py
# Description: Planetary regolith thermal model, including Hapke's 1996 radiative transfer 2-stream approximation. .
# Author: Andrew J. Ryan, 2025
#
# This code is free to use, modify, and distribute for any purpose.
# Please contact the author (ajryan4@arizona.edu) to discuss applications or if you
# use this code in your research or projects.
# -----------------------------------------------------------------------------

def fd1d_heat_implicit_matrix_nonuniform_kieffer(x_num, A1, A2, A3):
    """
    Build full finite-difference matrix for implicit 1D heat equation
    with non-uniform grid following Kieffer (2013) KRC method.
    """
    a = np.zeros((x_num, x_num))
    # Dirichlet boundaries: top and bottom
    a[0, 0] = 1.0
    for i in range(1, x_num - 1):
        a[i, i-1] = -A1[i-1]
        a[i, i]   = -(A1[i-1]*A2[i-1] - 1.0)
        a[i, i+1] = -A1[i-1]*A3[i-1]
    a[-1, -1] = 1.0
    return a


def fd1d_heat_implicit_diagonal_nonuniform_kieffer(x_num, A1, A2, A3):
    """
    Build banded (3-diagonal) matrix for implicit 1D heat equation
    with non-uniform grid following Kieffer (2013) KRC method.
    Returns `ab` array for scipy.linalg.solve_banded((1,1), ab).
    """
    ab = np.zeros((3, x_num))
    # upper diagonal
    ab[0, 2:]   = -A1 * A3
    # main diagonal
    ab[1, 0]    = 1.0
    ab[1, 1:-1] = -(A1 * A2 - 1.0)
    ab[1, -1]   = 1.0
    # lower diagonal
    ab[2, :-2]  = -A1
    return ab

#-----------------------------------------------------------------------------#
# Boundary-value problem stencils and banded Jacobians for radiative-transfer
#-----------------------------------------------------------------------------#

def build_bvp_stencil(x, nlay_dust):
    """
    Compute finite-difference coefficients for a non-uniform BVP stencil
    across the dust layer (excluding virtual boundary nodes).

    Returns A_im1, A_i, A_ip1, h0, hN for banded and full Jacobians.
    """
    h0 = x[1] - x[0] #first layer thickness
    hN = x[nlay_dust] - x[nlay_dust - 1] #last layer thickness
    hx = x[1:nlay_dust+1] - x[:nlay_dust]
    h = hx[1:] #thickness of all other layers
    h_im1 = h[:-1]
    h_ip1 = h[1:]
    A_im1 = 2.0 / (h_im1 * (h_im1 + h_ip1))
    A_i   = -2.0 / (h_im1 * h_ip1)
    A_ip1 = 2.0 / (h_ip1 * (h_im1 + h_ip1))
    return A_im1, A_i, A_ip1, h0, hN


def build_jacobian_vis_banded(A_bvp_params, gamma_vis):
    """
    Assemble the banded Jacobian ab matrix for visible RTE BVP.
    """
    A_im1, A_i, A_ip1, h0, hN = A_bvp_params
    N = len(A_im1)+1
    dF = 4*gamma_vis**2.
    ab = np.zeros((3, N+1))
    sub  = np.zeros(N+1)
    main = np.zeros(N+1)
    sup  = np.zeros(N+1)

    # row 0
    main[0] = 2*h0 + 1
    sup[0]  = -1.0

    # rows 1..N-1
    sub[1:N]  = A_im1
    main[1:N] = A_i - dF
    sup[1:N]  = A_ip1

    # row N
    sub[N]   = -1.0
    main[N]  = 2*hN + 1
    sup[N]   = 0.0

    # pack into ab for solve_banded((1,1), ...)
    ab[0, 1:]  = sup[:-1]
    ab[1, :]   = main
    ab[2, :-1] = sub[1:]
    return ab


def build_jacobian_therm_banded(A_bvp_params, gamma_therm):
    """
    Assemble the banded Jacobian ab matrix for thermal RTE BVP.
    """
    A_im1, A_i, A_ip1, h0, hN = A_bvp_params
    N = len(A_im1)+1
    dF = 4*gamma_therm**2.
    ab = np.zeros((3, N+1))
    main = np.zeros(N+1)
    sup  = np.zeros(N+1)
    sub  = np.zeros(N+1)
    # row 0 (Robin)
    main[0] = 2*h0 + 1
    sup[0]  = -1.0

    # rows 1..N-1
    sub[1:N]  = A_im1
    main[1:N] = A_i - dF
    sup[1:N]  = A_ip1

    # row N (Dirichlet)
    main[N] = 1.0

    # pack into ab for solve_banded((1,1), ...)
    ab[0, 1:]   = sup[:-1]
    ab[1, :]    = main
    ab[2, :-1]  = sub[1:]
    return ab
