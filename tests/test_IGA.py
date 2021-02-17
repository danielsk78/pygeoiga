import matplotlib.pyplot as plt
from time import process_time
from pygeoiga.analysis.common import *
from pygeoiga.analysis.iga import *
from pygeoiga.nurb.cad import *
from pygeoiga.nurb.refinement import knot_insertion

def basic_geometry(XY = np.asarray([[0, -10], [100, -10], [0, 10], [100, 10]]),
                   degree = 2,
                   refine = 10,
                   k = {}):
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
    U, V, B = make_surface_square(XY, degree)
    degree_u = len(np.where(U == 0.)[0]) - 1
    n_xi = len(U) - (degree_u + 1) * 2

    degree_v = len(np.where(V == 0.)[0]) - 1
    n_eta = len(V) - (degree_v + 1) * 2

    ### Knot refinement
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

    return degree, n_xi, n_eta, nel, n, m, ncp, gDof, P, W, IEN, K, b, D

def test_nurb_basis_IGA():
    xi = 19.5
    nx = 22
    eta = 0.8873
    ny = 11
    degree = 2

    nxi, dxi, neta, deta = nurb_basis_IGA(xi, eta, nx, ny, degree)
    print(nxi, dxi, neta, deta)

def test_jacobian_IGA():
    nxi = np.asarray([0.125, 0.75, 0.125])
    dxi = np.asarray([-0.25, 0, 0.25])
    neta = np.asarray([0.0127, 0.5936, 0.3936])
    deta = np.asarray([-0.1127, -0.3309, 0.4436])

    degree, n_xi, n_eta, nel, n, m, ncp, gDof, P, W, IEN, K, b, D = basic_geometry()

    J, dxy = jacobian_IGA(nxi, dxi, neta, deta, P, IEN[19], degree)
    print(J, dxy)

def test_form_stiffnes_matrix_IGA():
    degree, n_xi, n_eta, nel, n, m, ncp, gDof, P, W, IEN, K, b, D = basic_geometry(refine = 9)
    kappa = np.ones((n, m))
    K = form_k_IGA(K, IEN, P, kappa, n_xi, n_eta, degree)
    print(K, K.shape)

def test_solution_IGA():
    degree, n_xi, n_eta, nel, n, m, ncp, gDof, P, W, IEN, K, b, D = basic_geometry()

    kappa = np.ones((n, m)) * 0.4
    K = form_k_IGA(K, IEN, P, kappa, n_xi, n_eta, degree)

    ############## Play with the boundary conditions
    T_t = 1
    T_l = 1
    T_r = 1
    T_b = lambda x, m: T_t + 10 * np.sin(np.pi * x / m)  # [°C]

    bc, D = boundary_condition(T_t, T_b, T_l, T_r, D, m, n, ncp)

    D, b = solve(bc, K, b, D)

    print(D)
    T = D.reshape((n, m))
    plt.imshow(T, origin="lower", alpha=0.8)
    plt.colorbar()
    cs = plt.contour(T, colors="black")
    plt.clabel(cs)
    plt.show()

def test_solve_complete_3_IGA():
    t1 = process_time()
    degree, n_xi, n_eta, nel, n, m, ncp, gDof, P, W, IEN, K, b, D = basic_geometry(refine = 50)
    kappa = np.empty((n,m))

    kappa[:int(n/3):,:] = 3.4 # Granite
    kappa[int(int(n/3)/3):-int(int(n/3)/3)-int(2*n/3),int(m/3):-int(m/3)] = 0.6
    kappa[int(n/3):-int(n/3),:] = 1.25 # shale
    kappa[-int(n / 3):, :] = 2.25  # shale
    kappa_e = kappa_domain(kappa, IEN)
    print(kappa_e)
    ####
    t2 = process_time()
    K = form_k_IGA(K, IEN, P, kappa, n_xi, n_eta, degree)

    plt.imshow(kappa, origin = "lower")
    cb = plt.colorbar()
    cb.set_label("Thermal conductivity [W/mK]")
    plt.show()

    ############## Play with the boundary conditions
    T_t = 10
    T_l = None
    T_r = None
    T_b = 40
    bc, D = boundary_condition(T_t, T_b, T_l, T_r, D, m, n, ncp)

    D, b = solve(bc, K, b, D)
    t3 = process_time()
    print(D)
    T = D.reshape((n, m))
    plt.imshow(T, origin="lower", alpha=0.8)
    cb = plt.colorbar()
    cb.set_label("Temperature [°C]")
    cs = plt.contour(T, colors="black")
    plt.clabel(cs)
    plt.show()

    t4 = process_time()

    print("Time run all script: ", t4-t1) #71.921875
    print("Time run IGA: ", t3-t2)#70.609375

def test_solution_IGA():
    knots, B = quarter_disk()
    degree, n_xi, n_eta, nel, n, m, ncp, gDof, P, W, IEN, K, b, D, knots, B = analyze_IGA(knots, B, refine=10)

    kappa = 1
    K = form_k_IGA(K, IEN, P, kappa, n_xi, n_eta, degree, knots=knots)
    ############## Play with the boundary conditions
    T_t = 0
    T_l = None
    T_r = None
    T_b = 10#lambda x, m: T_t + 10 * np.sin(np.pi * x / m)  # [°C]

    bc, D = boundary_condition(T_t, T_b, T_l, T_r, D, m, n, ncp)

    D, b = solve(bc, K, b, D)

    print(D)
    T = D.reshape((n, m))
    plt.imshow(T, origin="lower", alpha=0.8)
    plt.colorbar()
    cs = plt.contour(T, colors="black")
    plt.clabel(cs)
    plt.show()

    print(D)
    T = b.reshape((n, m))
    plt.imshow(T, origin="lower", alpha=0.8)
    plt.colorbar()
    cs = plt.contour(T, colors="black")
    plt.clabel(cs)
    plt.show()


def test_get_point_solution():
    knots, B = quarter_disk()
    degree, n_xi, n_eta, nel, n, m, ncp, gDof, P, W, IEN, K, b, D, knots, B = analyze_IGA(knots, B, refine=10)

    kappa = np.ones((n, m))
    K = form_k_IGA(K, IEN, P, kappa, n_xi, n_eta, degree, knots=knots)
    ############## Play with the boundary conditions
    T_t = 0
    T_l = None
    T_r = None
    T_b = 10  # lambda x, m: T_t + 10 * np.sin(np.pi * x / m)  # [°C]

    bc, D = boundary_condition(T_t, T_b, T_l, T_r, D, m, n, ncp)

    D, b = solve(bc, K, b, D)

    from pygeoiga.analysis.common import point_solution

    val = point_solution(6, 5, D, B, knots, tolerance=1e-11, itera =1000)
    print(val) # knots (0.5548848818987607, 0.7526260007172824),
    # approx value (5.999999999923407, 4.999999997422794),
    # itera = 84,
    # stepx and y 3.7252902984619143e-10, 3.7252902984619143e-10,
    # temperature = 4.435870165147437)