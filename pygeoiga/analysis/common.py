import numpy as np
from scipy.sparse.linalg import cg
from scipy.stats import mode


def IEN_element_topology(nx, ny, degree):
    """
    Function that return the element topology of the control points:
    Numbering of control points counter clock-wise
    Args:
        nx: number of elements in xi direction
        ny: number of element in eta direction
        degree: polynomial order

    Returns:
        ET: Mesh counter clock-wise indicating the face of the mesh
    """
    # Utility functions
    def face(row, column, p):
        """ Create a single face """
        ls = [(nx + degree) * (column + i) + row + j
              for i in range(p + 1)
              for j in range(p + 1)]
        return ls
    IEN = [face(x, y, degree) for y in range(ny) for x in range(nx)]
    return np.asarray(IEN)


def transform_matrix(cpoints, weights):
    """
    Is the same as INN? NURBS coordinates array
    Convert the 3 control point matrixes in a single matrix with its control points
    Args:
        cpoints: control points
        weights: weights

    Returns:

    """
    n, m = cpoints.shape[0], cpoints.shape[1]

    P = np.asarray([(cpoints[x, y, :]) for x in range(n) for y in range(m)])
    W = np.asarray([(weights[x, y, :]) for x in range(n) for y in range(m)])

    return P, W


def transform_matrix_B(B):
    """
    Convert the 3 control point matrixes in a single matrix with its control points
    Args:
        cpoints: control points
        weights: weights

    Returns:

    """
    n, m = B.shape[0], B.shape[1]

    P = np.asarray([(B[x, y, 0], B[x, y, 1]) for y in range(m) for x in range(n)])
    W = np.asarray([(B[x, y, -1]) for y in range(m) for x in range(n)])

    return P, W


def gauss_points(degree): #Modify here to extend to 3d
    """
    Gauss quadrature for performing numerical integration.
    low-order quadrature rules are tabulated below (over interval [−1, 1])
    Args:
        degree: polynomial order

    Returns:
        G: gauss points
        W: weights
    """
    if degree == 1:
        Gtemp = [-1/np.sqrt(3), 1/np.sqrt(3)]
        wtemp = [1, 1]
    elif degree == 2:
        Gtemp = [-np.sqrt(0.6), 0, np.sqrt(0.6)]
        wtemp = [5/9, 8/9, 5/9]
    elif degree == 3:
        Gtemp = [-np.sqrt((3+2*np.sqrt(6/5))/7), -np.sqrt((3-2*np.sqrt(6/5))/7),
                 np.sqrt((3-2*np.sqrt(6/5))/7), np.sqrt((3+2*np.sqrt(6/5))/7)]
        wtemp = [(18-np.sqrt(30))/36, (18+np.sqrt(30))/36,
                 (18+np.sqrt(30))/36, (18-np.sqrt(30))/36]
    elif degree == 4:
        Gtemp = [-(1/3)*np.sqrt(5+2*np.sqrt(10/7)), -(1/3)*np.sqrt(5-2*np.sqrt(10/7)), 0,
                 (1/3)*np.sqrt(5-2*np.sqrt(10/7)), (1/3)*np.sqrt(5+2*np.sqrt(10/7))]
        wtemp = [(322-13*np.sqrt(70))/900, (322+13*np.sqrt(70))/900, 128/225,
                 (322+13*np.sqrt(70))/900, (322-13*np.sqrt(70))/900]
    else:
        raise NotImplementedError

    G = np.asarray([(x, y) for y in Gtemp for x in Gtemp])
    W = np.asarray([k * t for k in wtemp for t in wtemp])

    return G, W


def boundary_condition(T_t, T_b, T_l, T_r, D, m, n, ncp):
    """
    Construct the "Drichlet" boundary conditions constrained by the top and bottom temperatures
    and get the corresponding degrees of freedom from the mesh
    Args:
        T_t: Top temperature
        T_b: Bottom temperature/ can be a function aswell
        T_l: Left temperature
        T_r right temperature
        m: number of basis functions in xi direction
        n: number of basis functions in eta direction
        ncp: number of control points
    Returns:
        dictionary containing the indexes where the boundary conditions are applied, the top and bottom temperature
    """
    nodes = np.arange(0, ncp)
    acD = np.array([])
    if T_b is not None:
        b = nodes[:n].astype("int")  # T_b
        if callable(T_b):
            #T_b = lambda x, m: T_t + 10 * np.sin(np.pi * x / m)  # something like this
            D[b] = T_b((b), len(b))
        else:
            D[b] = T_b
        acD = np.hstack([acD, b])
    if T_t is not None:
        t = nodes[-n:].astype("int")  # T_t
        if callable(T_t):
            D[t] = T_t((t), len(t))
        else:
            D[t] = T_t
        acD = np.hstack([acD, t])
    if T_l is not None:
        l = nodes[::n].astype("int")  # T_l
        if callable(T_l):
            D[l] = T_l((l), len(l))
        else:
            D[l] = T_l
        acD = np.hstack([acD, l])
    if T_r is not None:
        r = nodes[n - 1::n].astype("int")  # T_r
        if callable(T_r):
            D[r] = T_r((r), len(r))
        else:
            D[r] = T_r
        acD = np.hstack([acD, r])
    acD = acD.astype('int')
    return {"prDOF": np.unique(acD),
            "gDOF": ncp}, D


def solve(bc, K, b, D):
    """
    This will solve the global system of matrixes with the current active DOF
    Args:
        gDOF: global degrees of freedom
        bc: boundary conditions
        K: global stiffness matrix
        b: global load vector
        D: global displacement vector empty
    Returns:
        D: global displacement vector
    """

    prDOF = bc.get("prDOF")
    gDOF = bc.get("gDOF")
    assert prDOF is not None, "patch degrees of freedom not found"
    assert gDOF is not None, "globa degrees of freedom not found"
    acDOF = np.setxor1d(np.arange(0, gDOF), prDOF).astype('int')

    Kfs = K[np.ix_(acDOF, prDOF)]
    bf = b[acDOF]
    bf_n = bf - Kfs@D[prDOF]

    D[acDOF], info = cg(A=K[np.ix_(acDOF, acDOF)], b=bf_n)

    b = K@D

    return D, b


def kappa_domain(kappa, IEN):
    """
    Assign a value of thermal conductivity to each node and calculate the element thermal conductivity
    Args:
        kappa: matrix
        IEN:
        gDOF:
        n:
        m:
        degree:

    Returns:

    """
    return np.asarray([mode(kappa.ravel()[IEN[i]])[0][0] for i in range(len(IEN))])


def map_nodes(degree):
    if degree == 1:
        nodes = np.array([[-1, - 1],[1, - 1],[-1, 1],[1, 1]])
    elif degree== 2:
        nodes = np.array([[-1, - 1],[0, - 1],[1, - 1],[-1,0],[0,0],
                          [1,0],[-1,1],[0,1],[1,1]])
    elif degree == 3:
        nodes = np.array([[-1, - 1],[-1 / 3, - 1],[1 / 3, - 1],[1, - 1],
                          [-1, - 1 / 3],[-1 / 3, - 1 / 3],[1 / 3, - 1 / 3],
                          [1, - 1 / 3],[-1,1 / 3],[-1 / 3,1 / 3],
                          [1 / 3,1 / 3],[1, 1 / 3],[-1, 1],
                          [-1 / 3, 1],[1 / 3, 1],[1, 1]])
    elif degree== 4:
        nodes = np.array([[-1, - 1],[-1 / 2, - 1],[0, - 1],[1 / 2, - 1],
                          [1, -1],[-1, - 1 / 2],[-1 / 2, - 1 / 2],
                          [0, - 1 / 2],[1 / 2, - 1 / 2],[1, - 1 / 2],
                          [-1, 0],[-1 / 2, 0],[0, 0],[1 / 2, 0],
                          [1, 0],[-1,1 / 2],[-1 / 2,1 / 2],[0,1 / 2],
                          [1 / 2, 1 / 2],[1,1 / 2],[-1,1],
                          [-1 / 2,1],[0,1],[1 / 2,1],[1,1]])
    else:
        print(degree)
        raise NotImplementedError
    return nodes

def position_borders_nrb(B, knots, resolution=50):
    """
    Obtain the NURBS curve defining the external border of a NURBS surface
    Args:
        B: Control point matrix
        knots: knot vector (U,V)
        resolution: Number of points per border
    Returns:
        Positions as a list going:
            4<----------3
            |           |
            |           |
            |           |
            1---------->2
    """
    from pygeoiga.engine.NURB_engine import surface_point

    degree1 = len(np.where(np.asarray(knots[0]) == 0.)[0]) - 1
    degree2 = len(np.where(np.asarray(knots[1]) == 0.)[0]) - 1
    cp = B[...,:-1]
    weight = B[..., -1, np.newaxis]

    points1 = np.linspace(knots[0][0], knots[0][-1], resolution)
    points2 = np.linspace(knots[1][0], knots[1][-1], resolution)

    positions_xi = []
    for u in [knots[0][0], knots[0][-1]]:
        positions_1 = []
        for v in points2:
            positions_1.append(surface_point(u,
                                             v,
                                             degree1,
                                             degree2,
                                             knots[0],
                                             knots[1],
                                             cp,
                                             weight))
        positions_xi.append(positions_1)
    positions_eta = []
    for u in [knots[1][0], knots[1][-1]]:
        positions_1 = []
        for v in points1:
            positions_1.append(surface_point(v,
                                             u,
                                             degree1,
                                             degree2,
                                             knots[0],
                                             knots[1],
                                             cp,
                                             weight))

        positions_eta.append(positions_1)
    pos = np.vstack((positions_xi[0], positions_eta[1], np.flipud(positions_xi[1]), np.flipud(positions_eta[0])))

    return pos

def point_solution(x, y, temperature, B, knots, tolerance = 1e-9, itera = 1000):
    """
    Get the value of the field (temperature) at the location (x, y) from the NURBS patch.
    If outside of pacth skip process
    Args:
        x: x coordinate
        y: y coordinate
        temperature: temperature field at nodal positions
        B: control points
        knots: knots
        tolerance: accurate to which decimal. Default 1e-9
        itera: to avoid infinite while loop. Maximum of iterations to be 10000
    Returns:
        temperature value at points (x, y)
    """
    pos = position_borders_nrb(B, knots, resolution=100)
    import matplotlib.path as mpltPath
    path = mpltPath.Path(pos)
    if not path.contains_point((x,y)) and not [x, y] in pos:
         #print("position (%s, %s) outside of range"%(x,y))
         return None
    knots1 = knots[0]
    knots2 = knots[1]
    degree1 = len(np.where(np.asarray(knots1) == 0.)[0]) - 1
    degree2 = len(np.where(np.asarray(knots2) == 0.)[0]) - 1
    cp = B[...,:-1]
    weight = B[...,-1]
    u_i = 0.5
    v_i = 0.5  # in the middle of the structure
    x_step = 0.1
    y_step = 0.1

    # Determine the value of u and v for that points
    from pygeoiga.engine.NURB_engine import surface_point
    accepted = False
    i = 0
    xpol = None
    xprev_pol = None
    x_accepted = False

    ypol = None
    yprev_pol = None
    y_accepted = False
    while not accepted:
        if i > itera:
            break
        position = surface_point(u_i, v_i, degree1, degree2, knots1, knots2, cp, weight)

        x_err = position[0] - x
        y_err = position[1] - y

        if not -tolerance < x_err < tolerance:
            if x_err < tolerance:
                u_i += x_step
                xpol = "right"
            elif x_err > tolerance:
                u_i -=x_step
                xpol = "left"

            if xprev_pol is None:
                xprev_pol = xpol

            if xpol != xprev_pol:
                x_step /= 2
                xprev_pol = xpol
        else:
            x_accepted = True

        if not -tolerance < y_err < tolerance:
            if y_err < tolerance:
                v_i += y_step
                ypol = "right"
            elif y_err > tolerance:
                v_i -= y_step
                ypol = "left"

            if yprev_pol is None:
                yprev_pol = ypol

            if ypol != yprev_pol:
                y_step /= 2
                yprev_pol = ypol
        else:
            y_accepted = True

        i +=1
        if y_accepted and x_accepted:
            accepted = True

    from pygeoiga.engine.NURB_engine import basis_function_point

    N_u, _ = basis_function_point(u_i, knots1, degree1)
    N_v, _ = basis_function_point(v_i, knots2, degree2)
    n1 = len(knots1) - degree1 - 1
    n2 = len(knots2) - degree2 - 1

    N_u = N_u[:n1]
    if u_i == knots1[-1]:
        N_u[-1] = 1
    N_v = N_v[:n2]
    if v_i == knots2[-1]:
        N_v[-1] = 1

    Rb_temp = np.kron(N_u, N_v)
    t = Rb_temp @ temperature

    #return u_i, v_i, position, i, x_step, y_step, t
    return t


