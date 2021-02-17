import numpy as np
from pygeoiga.analysis.common import gauss_points, kappa_domain, map_nodes


def analyze_bezier(knots, B, refine = 3, _refine=True):
    """
    Args:
        knots: U and V
        B: control point of surface

    Returns:
        [degree = degree
        n_xi and n_eta are the number of elements
        nel = n_xi * n_eta  # total number of elements
        n = n_xi + degree_u  # Number of basis functions/ control points in xi direction
        m = n_eta + degree_v  # Number of basis functions/ control points in eta direction
        ncp = n * m  # Total number of control points
        gDof = ncp  # Global degrees of freedom - Temperature
        C = bezier extractor
        P = control points
        W = Weights
        IEN = Elemental topology of nodes
        K = Empty stiffness matrix
        b = load vector empty
        D = Displacement vector empty]
    """
    from pygeoiga.nurb.refinement import knot_insertion
    from pygeoiga.analysis.common import transform_matrix, IEN_element_topology
    from pygeoiga.analysis.bezier_extraction import bezier_extraction_operator_bivariate, bezier_extraction_operator

    U, V = knots
    degree_u = len(np.where(U == 0.)[0]) - 1
    n_xi = len(U) - (degree_u + 1) * 2

    degree_v = len(np.where(V == 0.)[0]) - 1
    n_eta = len(V) - (degree_v + 1) * 2

    ### Knot refinement
    if _refine:
        direction = 0
        knots_ins = np.linspace(0.1, 0.9, refine)
        knots = [U, V]
        B, knots = knot_insertion(B, [degree_v, degree_u], knots, knots_ins, direction=direction)
        direction = 1
        B, knots = knot_insertion(B, [degree_v, degree_u], knots, knots_ins, direction=direction)
        U, V = knots

    degree_u = len(np.where(U == 0.)[0]) - 1
    degree_v = len(np.where(V == 0.)[0]) - 1
    # same number of control points as basis functions
    n_xi = B.shape[1] - degree_u
    n_eta = B.shape[0] - degree_v
    # n_xi = len(U) - degree_u - 3
    # n_eta = len(V) - degree_u - 3

    # n_xi and n_eta are the number of elements
    nel = n_xi * n_eta  # total number of elements
    n = n_xi + degree_u  # Number of basis functions/ control points in xi direction
    m = n_eta + degree_v  # Number of basis functions/ control points in eta direction
    ncp = n * m  # Total number of control points
    gDof = ncp  # Global degrees of freedom - Temperature

    K = np.zeros((gDof, gDof))  # stiffness matrix size

    C_xi = bezier_extraction_operator(U, degree_u)

    C_eta = bezier_extraction_operator(V, degree_v)

    assert degree_u == degree_v

    degree = degree_u

    C = bezier_extraction_operator_bivariate(C_xi, C_eta, n_xi, n_eta, degree)

    control_points = B[..., :-1]
    ref_weights = B[..., -1, np.newaxis]
    P, W = transform_matrix(control_points, ref_weights)

    # Element topology
    IEN = IEN_element_topology(n_xi, n_eta, degree)

    b = np.zeros(gDof)  # load vector size
    D = np.zeros(gDof) # Displacement vector

    return degree, n_xi, n_eta, nel, n, m, ncp, gDof, C, P, W, IEN, K, b, D, B, knots


def form_k(K_glb, IEN, P, kappa, nel, degree, weight, C):
    """
    Function to form the global stiffness matrix
    Args:
        K_glb: empty stiffness matrix
        IEN: element topology = numbering of control points
        P: coordinates of NURBS control points x, y, z in single matrix (:,3)
        kappa: thermal conductivity
        nel: number of elements
        degree: polynomial order
        weight: weights of NURBS control points as array size(:,1)
        C: Bezier extraction operators shape(nx,ny, nel)
    Returns:
        K filled
    """
    # Gauss points and weights for performing the gaussian quadrature rule [-1 1]

    G, W = gauss_points(degree)
    #kappa_element = kappa_domain(kappa, IEN)
    for e in range(nel): # elements
        #Element topology of current element
        IEN_e = IEN[e]
        #element degree of freedom
        eDOF = IEN_e #np.append(IEN_e, IEN_e + ncp)
        #kappa_e = kappa_element[e]
        k = 0

        for g in range(len(G)):
            xi = G[g, 0]
            eta = G[g, 1]

            _, dR = nurbs_basis(xi, eta, degree, e, IEN_e, weight, C)
            J, dxy = jacobian(dR, P, IEN_e)
            k = k + (dxy.T @ dxy) * np.linalg.det(J) * W[g] * kappa #kappa_e

        #print(k)
        #temp = K_glb[eDOF][:,eDOF] + k
        K_glb[np.ix_(eDOF,eDOF)] = K_glb[np.ix_(eDOF,eDOF)] + k
    return K_glb


def bernstein_basis(xi, degree):
    """
    Function that forms univariate bernstein basis functions and derivatives
    Args:
        xi: parametric coordinate
        degree: polynomial order
    Returns:
        B_b: bernstein basis functions over interval [-1, 1]
        dB_b: derivative bernstein basis over interval [-1, 1]
    """
    B_b = np.zeros(degree + 1)
    dB_b = np.zeros(degree + 1)

    # Degree = 0
    B_b[0] = 1
    # Degree = 1,2,3..
    for p in range(1, degree + 1, 1):
        for i in range(p, 0, -1):
            dB_b[i] = 0.5 * p * (B_b[i - 1] - B_b[i])
            B_b[i] = 0.5 * (1 - xi) * B_b[i] + 0.5 * (1 + xi) * B_b[i - 1]
        dB_b[0] = -0.5 * p * B_b[0]
        B_b[0] = 0.5 * (1 - xi) * B_b[0]

    return B_b, dB_b


def bernstein_basis_bivariate(B_b_xi, dB_b_xi, B_b_eta, dB_b_eta, degree):
    """
    Forms the bivariate bernstein basis functions and derivatives
    Args:
        B_b_xi: bernstein basis function in xi direction
        dB_b_xi: derivative bernstein basis function in xi direction
        B_b_eta: bernstein basis function in eta direction
        dB_b_eta: derivative bernstein basis function in eta direction
        degree: polynomial order

    Returns:
        B_b: Bernstein basis functions over the interval [-1,1]x[-1,1]
        dB_b: derivatives of Bernstein basis functions
    """
    B_b = np.asarray([x * y for y in B_b_eta for x in B_b_xi])
    dB_b = np.asarray([(dB_b_xi[i] * B_b_eta[j],
                        B_b_xi[i] * dB_b_eta[j])
                       for j in range(degree + 1)
                       for i in range(degree + 1)
                       ])

    return B_b, dB_b


def nurbs_basis(xi, eta, degree, e, IEN_e, weight, C):
    """
    Function to form the basis function and derivatives
    Args:
        xi: coordinate od xi parametric direction
        eta: coordinate od eta parametric direction
        degree: polynomial order
        e: current element number
        IEN_e: element topology of current element
        weight: NURBS weights
        C: bezier extraction operators

    Returns:
        Rb: NURBS basis functions
        dR: derivatives of NURBS basis functions
    """
    B_b_xi, dB_b_xi = bernstein_basis(xi, degree)
    B_b_eta, dB_b_eta = bernstein_basis(eta, degree)
    B_b, dB_b = bernstein_basis_bivariate(B_b_xi, dB_b_xi, B_b_eta, dB_b_eta, degree)

    # Bezier weights
    wb = C[:len(IEN_e), :, e].T @ weight[IEN_e]
    Wb = B_b @ wb
    dWB_b_xi = dB_b[:, 0] @ wb
    dWB_b_eta = dB_b[:, 1] @ wb

    # NURBS basis functions and derivatives
    temp = np.diag(weight[IEN_e].T[0]) @ C[:len(IEN_e), :, e]

    Rb =  temp @ np.divide(B_b, Wb, out=np.zeros_like(B_b), where=Wb != 0)
    dR_1 = temp @ (np.divide(dB_b[:, 0], Wb, out=np.zeros_like(dB_b[:, 0]), where=Wb != 0) -
                   dWB_b_xi * np.divide(B_b, Wb**2, out=np.zeros_like(B_b), where=Wb**2 != 0))

    dR_2 = temp @ (np.divide(dB_b[:, 1], Wb, out=np.zeros_like(dB_b[:, 1]), where=Wb != 0) -
                   dWB_b_eta * np.divide(B_b, Wb**2, out=np.zeros_like(B_b), where=Wb**2 != 0))

    dR = np.asarray([dR_1, dR_2]).T

    return Rb, dR

def jacobian(dR, P, IEN_e):
    """
    Calculate the jacobian matrix and its derivative. Is not yet compatible with 3d structures
    Args:
        dR: Derivatives of the nurbs basis
        P: coordinates of the control points
        IEN_e: element topology of the current element

    Returns:
        J: jacobian matrix
        dxy: derivative
    """
    J = np.zeros((dR.shape[1], dR.shape[1]))
    for i in range(dR.shape[1]):
        for j in range(dR.shape[1]):
            J[i, j] = dR[:, i] @ P[IEN_e, j]

    dxy = np.linalg.solve(J, dR.T)
    #J = np.linalg.det(J)

    return np.asarray(J), np.asarray(dxy)

def form_load_vector(b, bc):#, IEN, P, nx, ny, ncp, degree, weight, C):
    """
    Forms the global nodal vector
    Args:
        b: empty global nodal vector
        IEN: element topology: numbering of control points
        P: coordinates of NURBS control points
        nx: number of elements in xi direction
        ny: number of elements in eta direction
        ncp: number of control points
        degree: polynomial order
        weight: weights of NURBS control points
        C: Bézier extraction operators

    Returns:
    G, W = gauss_points(degree)
    xi = 1

    s = 0

    e= nx*s
    # Element topology of current element
    IEN_e = IEN[e]
    IEN_ey = IEN_e + ncp

    # element degree of freedom
    eDOF = np.append(IEN_e, IEN_e + ncp)
    k = 0
    """
    prDOF = bc.get("prDOF")
    T_t = bc.get("T_t")
    T_b = bc.get("T_t")
    b[prDOF[0,:]]= T_b
    b[prDOF[1, :]] = T_t

    return b


def map_bezier_elements(D, degree, P, n, m, ncp, nel, IEN, weight, C):
    """

    Args:
        D: global displacement vector
        degree: degree of the nurb object
        P: control points
        n: number of control points in xi direction
        m: number of control points in eta direction
        ncp: number of control points
        nel: number of elements
        IEN: element topology: numbering of control points
        weight: weight of the control points
        C: bezier extraction operator
        ax: matplotlib axes

    Returns:

    """
    # call parametric coordinates which equals nodes in FEA
    nodes = map_nodes(degree)
    t_temp = np.zeros(ncp)
    x_temp = np.zeros(ncp)
    y_temp = np.zeros(ncp)

    for e in range(nel):
        IEN_e = IEN[e]
        for g in range(len(nodes)):
            xi = nodes[g, 0]
            eta = nodes[g, 1]

            Rb, _ = nurbs_basis(xi, eta, degree, e, IEN_e, weight, C)
            t_temp[IEN_e[g]] = Rb @ D[IEN_e]
            x_temp[IEN_e[g]] = Rb @ P[IEN_e, 0]
            y_temp[IEN_e[g]] = Rb @ P[IEN_e, 1]

    t = np.zeros((n, m))
    x = np.zeros((n, m))
    y = np.zeros((n, m))

    k = 0
    for j in range(m):
        for i in range(n):
            t[i, j] = t_temp[k]
            x[i, j] = x_temp[k]
            y[i, j] = y_temp[k]
            k += 1

    return x, y, t