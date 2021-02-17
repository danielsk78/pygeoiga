import numpy as np
from pygeoiga.nurb import NURB

def bezier_extraction_operator(knot_vector, degree):
    """
    Function to compute the univariate extraction operator as described in Borden et al. 2010
    The Bézier extraction operator maps a piecewise Bernstein polynomial basis onto a B-spline basis.
    This transformation makes it possible to use piecewise C 0 Bézier elements as the finite element
    representation of an NURBS or T-spline.

    To compute the Bézier elements of an NURBS, we use Bézier decomposition.

    Args:
        XI:
        degree:

    Returns:
        C: Extractor operator
    """
    size = len(knot_vector)-(degree)*2#-1
    m = len(knot_vector)
    a = degree
    b = a + 1
    nb = 0
    C = np.zeros((degree+1, degree+1, size))
    C[...,0] = np.eye(degree+1)

    while b < m:
        i = b
        C[..., nb + 1] = np.eye(degree + 1)
        #Count multiplicity of the knot at location b
        while b < m and knot_vector[b+1] == knot_vector[b]:
            b += 1
            if b == m-1:
                return C

        mult = b-i
        alphas = np.zeros(degree+mult)
        if mult < degree-1:
            numer = knot_vector[b]-knot_vector[a]
            for j in range(degree, mult+1, -1):
                alphas[j-mult-2] = numer/(knot_vector[a+j]-knot_vector[a])

            r = degree-mult

            # update the matrix coefficients for r new knots
            for j in range(r-1):
                save = r-j-2
                s = mult+j
                for k in range(degree, s+1, -1):
                    alpha = alphas[k-s-2]
                    C[:, k, nb] = alpha * C[:, k, nb]+(1-alpha)*C[:, k-1, nb]
                if b < m:
                    C[save:j+save+2, save, nb+1] = C[degree-j-1:degree+1, degree, nb]
            # Finished with the current operator
            nb += 1
            if b < m:
                # update indices for next operator
                a = b
                b += 1

        elif mult == degree:
            nb += 1
            if b<m:
                a=b
                b +=1

    return C

def bezier_extraction_operator_bivariate(C_xi, C_eta, nxi, neta, degree):
    """
    Computes the localized bivariate extraction operators by using Kronecker tensor product of two arrays
    Args:
        C_xi: univariate extraction operator in direction xi
        C_eta: univariate extraction operator in direction eta
        nxi: number of elements in xi direction
        neta: number of elements in eta direction
        degree: polynomial order
    Returns:
        C_biv: bivariate extraction operator
    """
    C_biv = np.zeros(((degree+1)**2, (degree+1)**2, nxi*neta))
    k = 0
    for j in range(neta):
        for i in range(nxi):
            C_biv[..., k] = np.kron(C_eta[..., j], C_xi[..., i])
            k += 1
    return C_biv


def bivariate_bernstein_basis(ber_xi, ber_eta, db_x1, db_eta, degree):
    """
    Form bivariate basis functions and derivatives
    Args:
        ber_xi: Bernstein basis function in xi direction
        ber_eta: Bernstein basis function in eta direction
        db_x1: xi derivatives of Bernstein basis functions
        dd_eta: eta derivatives of Bernstein basis functions
        degree: polinomial order

    Returns:
        Bb: Bernstein basis functions
        dB: derivatives of Bernstein basis functions
    """
    Bb = np.zeros((1,(degree+1)**2))
    dB = np.zeros((2,(degree+1)**2))
    k=0
    for i in range(degree):
        for j in range(degree):
            Bb[k] = ber_xi[i]*ber_eta[j]
            dB[0,k] = db_x1[i]*ber_eta[j]
            dB[1,k] = ber_xi[i]*db_eta[j]
            k += 1

    return Bb, dB

def neumann_bc_bezier(D, IEN, P, nx, ny, ncp, degree, W, C):
    """
    Apply neumann boundary conditions to a side
    Args:
        D: Displacement vector
        IEN: element topology: numbering of control points
        P: coordinates of NURBS control points
        nx: number of elements in xi direction
        ny: number of elements in eta direction
        ncp: number of control points
        degree: degree of the polynomial
        W: weights of NURBS control points
        C: bézier extraction operators

    Returns:

    """
    from pygeoiga.analysis.common import gauss_points
    G, W = gauss_points(degree)

    for e in range(nel):  # elements
        # e = 0
        # Element topology of current element
        IEN_e = IEN[e]
        # element degree of freedom
        eDOF = IEN_e  # np.append(IEN_e, IEN_e + ncp)
        kappa_e = kappa_element[e]
        k = 0

        for g in range(len(G)):
            xi = G[g, 0]
            eta = G[g, 1]

            _, dR = nurbs_basis(xi, eta, degree, e, IEN_e, weight, C)
            J, dxy = jacobian(dR, P, IEN_e)
            k = k + (dxy.T @ dxy) * np.linalg.det(J) * W[g] * kappa_e

        # print(k)
        # temp = K_glb[eDOF][:,eDOF] + k
        K_glb[np.ix_(eDOF, eDOF)] = K_glb[np.ix_(eDOF, eDOF)] + k
    return K_glb