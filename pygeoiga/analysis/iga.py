import numpy as np
from pygeoiga.analysis.common import gauss_points, map_nodes
from pygeoiga.engine.NURB_engine import basis_function_point


def analyze_IGA(knots, B, refine = 3, _refine=True):
    """
        Args:
            XY: edges of the square
            degree: degree of the nurbs
            refine: number of how many more points to refine

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

    n_xi = len(U) - degree_u - 3
    n_eta = len(V) - degree_v - 3
    #n_xi = len(U) - (degree_u + 1) * 2
    #n_eta = len(V) - (degree_v + 1) * 2

    # n_xi and n_eta are the number of elements
    nel = n_xi * n_eta  # total number of elements
    n = n_xi + degree_u  # Number of basis functions/ control points in xi direction
    m = n_eta + degree_v  # Number of basis functions/ control points in eta direction
    ncp = n * m  # Total number of control points
    gDof = ncp  # Global degrees of freedom - Temperature

    K = np.zeros((gDof, gDof))  # stiffness matrix size
    degree = degree_u
    control_points = B[..., :-1]
    ref_weights = B[..., -1, np.newaxis]
    P, W = transform_matrix(control_points, ref_weights)

    # Element topology
    IEN = IEN_element_topology(n_xi, n_eta, degree)

    b = np.zeros(gDof)  # load vector size
    D = np.zeros(gDof)

    return degree, n_xi, n_eta, nel, n, m, ncp, gDof, P, W, IEN, K, b, D, knots, B


def form_k_IGA(K_glb, IEN, P, kappa, nx, ny, degree, knots=None):
    """
       Function to form the global stiffness matrix
       Args:
           K_glb: empty stiffness matrix
           IEN: element topology = numbering of control points
           P: coordinates of NURBS control points x, y, z in single matrix (:,3)
           kappa: thermal conductivity
           nel: number of elements
           ncp: number of control points
           degree: polynomial order
           weight: weights of NURBS control points as array size(:,1)

       Returns:
           K filled
       """
    # Gauss points and weights for performing the gaussian quadrature rule [-1 1]

    G, W = gauss_points(degree)
    #kappa_element = kappa_domain(kappa, IEN)
    #Vectors defining the parameter space
    xi_ve = np.arange(0, nx+1)
    eta_ve = np.arange(0, ny+1)
    #global knot_save
    #xi_ve = knots[0][degree:-degree]
    #eta_ve = knots[1][degree:-degree]
    #xi_ve = knot_save[0][degree:-degree]#*nx
    #eta_ve = knot_save[1][degree:-degree]#*ny
    e = 0
    for i in range(ny):
        for j in range(nx):
            IEN_e = IEN[e] # Element topology of current element
            eDOF = IEN_e
            #kappa_e = kappa_element[e]
            k = 0
            for g in range(len(G)):
                xi_n = G[g, 0]
                eta_n = G[g, 1]

                # Gauss coordinates in parameter space
                xi = xi_ve[j] + (xi_n + 1) * (xi_ve[j + 1] - xi_ve[j]) / 2
                eta = eta_ve[i] + (eta_n + 1) * (eta_ve[i + 1] - eta_ve[i]) / 2

                N_xi, dN_xi, N_eta, dN_eta = nurb_basis_IGA(xi, eta, nx, ny, degree)
                #N_xi, dN_xi = basis_function_point(xi, knots[0], degree)
                #N_eta, dN_eta = basis_function_point(eta, knots[1], degree)

                # Mapping of derivatives from parameter space to reference element
                dN_xi = dN_xi/2
                dN_eta = dN_eta/2

                # Collect the basis functions and derivatives which support the element
                N_xi = N_xi[j:j+degree+1]
                dN_xi = dN_xi[j:j + degree+1]
                N_eta = N_eta[i:i + degree+1]
                dN_eta = dN_eta[i:i + degree+1]

                J, dxy = jacobian_IGA(N_xi, dN_xi, N_eta, dN_eta, P, IEN_e, degree)

                k = k + (dxy.T @ dxy) * np.linalg.det(J) * W[g] * kappa #kappa_e

            K_glb[np.ix_(eDOF, eDOF)] = K_glb[np.ix_(eDOF, eDOF)] + k
            e +=1
    return K_glb



def nurb_basis_IGA(xi, eta, nx=None, ny=None, degree=None, U_knot=None, V_knot=None, knots=None):
    """
    DEPRECATED
    Function that forms basis functions and derivatives
    Args:
        xi: coordinate in 1st parametric direction
        eta: coordinate in 2nd parametric direction
        nx: number of elements in xi direction
        ny: number of elements in eta direction
        degree: polynomial order - same in both directions

    Returns:

    """
    if nx is not None:
        xi_vector = np.r_[np.zeros(degree),
                          np.arange(nx),
                          np.ones(degree+1)*nx]
    if ny is not None:
        eta_vector = np.r_[np.zeros(degree),
                           np.arange(ny),
                           np.ones(degree+1)*ny]

    #global knot_save
    if knots is not None:
        xi_vector = knots[0]#*nx
        eta_vector = knots[1]#*ny

    if U_knot is not None:
        xi_vector= U_knot
    if V_knot is not None:
        eta_vector=V_knot

    N_xi, dN_xi = basis_function_point(xi, xi_vector, degree)
    N_eta, dN_eta = basis_function_point(eta, eta_vector, degree)
    #n1 = len(xi_vector) - degree - 1
    #n2 = len(eta_vector) - degree - 1
    #N_xi = N_xi[:n1]
    #if xi==N_xi[-1]:
    #    N_xi[-1]=nx
    #N_eta = N_eta[:n2]
    #if eta == N_eta[-1]:
    #    N_eta[-1] = ny

    return N_xi, dN_xi, N_eta, dN_eta

def jacobian_IGA(N_xi, dN_xi, N_eta, dN_eta, P, IEN_e, degree):
    """
    Calculate the Jacobian matrix and x,y derivatives
    Args:
        N_xi:
        dN_xi:
        N_eta:
        dN_eta:
        P:
        IEN_e:
        degree:

    Returns:

    """
    J = np.zeros((2,2))
    dR = np.zeros((2, (degree+1)**2))

    k=0
    for j in range(degree+1):
        for i in range(degree+1):

            dR[0, k] = dN_xi[i] * N_eta[j]
            dR[1, k] = N_xi[i] * dN_eta[j]

            #Control points that support the element
            J[0, 0] = J[0, 0] + dR[0, k] * P[IEN_e[k], 0]
            J[0, 1] = J[0, 1] + dR[0, k] * P[IEN_e[k], 1]
            J[1, 0] = J[1, 0] + dR[1, k] * P[IEN_e[k], 0]
            J[1, 1] = J[1, 1] + dR[1, k] * P[IEN_e[k], 1]

            k+=1

    dxy = np.linalg.solve(J, dR)
    return J, dxy


def map_solution_elements(D, degree, P, nx, ny, n, m, ncp, IEN, weight, knots):
    """

    Args:
        D: global displacement vector
        degree: degree of the nurb object
        P: control points
        nx: number of elements in xi direction
        ny: number of elements in eta direction
        n: number of control points in xi direction
        m: number of control points in eta direction
        ncp: number of control points
        nel: number of elements
        IEN: element topology: numbering of control points
        weight: weight of the control points
        ax: matplotlib axes

    Returns:

    """
    # call parametric coordinates which equals nodes in FEA
    nodes = map_nodes(degree)
    t_temp = np.zeros(ncp)
    x_temp = np.zeros(ncp)
    y_temp = np.zeros(ncp)

    # Vectors defining the parameter space
    xi_ve = np.arange(0, nx + 1)
    eta_ve = np.arange(0, ny + 1)
    #xi_ve = knots[0][degree:-degree]
    #eta_ve = knots[1][degree:-degree]
    e = 0
    for i in range(ny):
        for j in range(nx):
            IEN_e = IEN[e]

            for g in range(len(nodes)):
                xi_n = nodes[g, 0]
                eta_n = nodes[g, 1]

                xi = xi_ve[j] + (xi_n + 1) * (xi_ve[j + 1] - xi_ve[j]) / 2
                eta = eta_ve[i] + (eta_n + 1) * (eta_ve[i + 1] - eta_ve[i]) / 2

                N_xi, _, N_eta, _ = nurb_basis_IGA(xi, eta, nx, ny, degree)

                #N_xi, _ = basis_function_point(xi, knots[0], degree)
                #N_eta, _ = basis_function_point(eta, knots[1], degree)

                N_xi_temp = N_xi[j:j + degree + 1]
                N_eta_temp = N_eta[i:i + degree + 1]
                if i == ny-1 and np.all(N_eta_temp == 0):
                    N_eta_temp[-1] = 1
                if j == nx-1 and np.all(N_xi_temp == 0):
                    N_xi_temp[-1] = 1
                Rb_temp = np.kron(N_eta_temp, N_xi_temp)
                t_temp[IEN_e[g]] = Rb_temp @ D[IEN_e]

                x_temp[IEN_e[g]] = Rb_temp @ P[IEN_e, 0]
                y_temp[IEN_e[g]] = Rb_temp @ P[IEN_e, 1]

            e += 1

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