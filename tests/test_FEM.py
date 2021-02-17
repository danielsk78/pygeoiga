import matplotlib.pyplot as plt
from time import process_time
from pygeoiga.analysis.bezier_FE import *
from pygeoiga.analysis.bezier_extraction import *
from pygeoiga.analysis.common import *
from pygeoiga.nurb.cad import *
from pygeoiga.nurb.refinement import knot_insertion


def test_gauss():
    for degree in range(1, 5, 1):
        G, W = gauss_points(degree)
        print("Degree: ", degree,"\n G: ", G,"\n W: ", W)
        print(G[3,0])

def test_element_topology():
    nx = 10
    ny = 20
    degree = 1
    IEN = IEN_element_topology(nx, ny, degree)
    print(IEN)
    print(IEN[0])
    print(IEN[0][0])

def test_bernstein_basis():
    xi = 4
    degree = 3

    B_b, dB_b = bernstein_basis(xi, degree)
    ans = np.asarray([ -3.375,  16.875, -28.125,  15.625])
    ans_der = np.asarray([ -3.375,  14.625, -20.625,   9.375])
    assert np.allclose(ans, B_b)
    assert np.allclose(ans_der, dB_b)


def test_bivariate_basis():
    degree = 3
    xi = 4
    B_b_xi, dB_b_xi = bernstein_basis(xi, degree)
    eta = 6
    B_b_eta, dB_b_eta = bernstein_basis(eta, degree)
    B_b, dB_b = bernstein_basis_bivariate(B_b_xi, dB_b_xi, B_b_eta, dB_b_eta, degree)

    ans = np.asarray([   52.734375,  -263.671875,   439.453125,
                         -244.140625,  -221.484375,  1107.421875,
                         -1845.703125,  1025.390625,   310.078125,
                         -1550.390625, 2583.984375, -1435.546875,
                         -144.703125,   723.515625, -1205.859375,
                         669.921875])
    assert np.allclose(B_b, ans)

def test_transform_matrix():
    U, V, C, weight, B = make_surface_3d()
    P, W = transform_matrix(C, weight)

    print(P, W)

    XY = np.asarray([[0, -10], [100, -10], [0, 10], [100, 10]])
    U2, V2, B2 = make_surface_square(XY, 2)
    P2, W2 = transform_matrix_B(B2)

    print(P2, W2)

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
     #same number of control points as basis functions
    n_xi = B.shape[0] - degree_u
    n_eta = B.shape[1] - degree_v
    #n_xi = len(U) - degree_u - 3
    #n_eta = len(V) - degree_u - 3

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
    D = np.zeros(gDof)

    return degree, n_xi, n_eta, nel, n, m, ncp, gDof, C, P, W, IEN, K, b, D


def test_nurb_basis():
    degree, n_xi, n_eta, nel, n, m, ncp, gDof, C, P, W, IEN, K, b, D = basic_geometry()
    ##################
    #Now to form the stiffnes matrix first procedure
    gauss_G, gauss_W = gauss_points(degree)
    #element
    e = 0
    IEN_e = IEN[e]
    eDOF = np.append(IEN_e, IEN_e + ncp)
    xi = gauss_G[0, 0]
    eta = gauss_G[0, 1]

    Rb, dR = nurbs_basis(xi, eta, degree, e, IEN_e, W, C)


    print(Rb.shape, dR.shape)

    return dR, P, IEN_e

def test_jacobian():
    dR, P, IEN_e = test_nurb_basis()
    J, dxy = jacobian(dR, P, IEN_e)
    print("J", J, J.shape, "dxy", dxy, dxy.shape)
    ans = np.asarray([[ 2.28248578e-03, -1.15792577e-02, -7.53116206e-04,  1.00474119e-02,
                        -5.57001047e-04, -1.16318555e-04,  6.19127468e-04,  5.79792449e-05,
                        -1.31092322e-06],
                       [-3.62876369e+00,  2.64896817e+00,  1.97326515e-01, -2.13594279e-01,
                        8.85536810e-01,  5.79100398e-02,  1.56927596e-02,  3.48395280e-02,
                        2.08414468e-03]])
    assert np.allclose(ans, dxy)
    return J, dxy

def test_k_form():
    degree, n_xi, n_eta, nel, n, m, ncp, gDof, C, P, W, IEN, K, b, D = basic_geometry()
    ##################
    kappa = np.ones((n, m)) * 0.4 #[W/mK]
    K = form_k(K, IEN, P, kappa, gDof, nel, ncp, degree, W, C)

    print(np.linalg.det(K))
    plt.spy(K)
    plt.show()
    #assert np.linalg.det(K) != 0.


def test_BC():
    degree, n_xi, n_eta, nel, n, m, ncp, gDof, C, P, W, IEN, K, b, D = basic_geometry()
    T_t = 10 #[°C]
    T_b = 40 #[°C]
    T_l = None
    T_r = None
    bc, D = boundary_condition(T_t, T_b, T_l, T_r, D, m, n, ncp)
    print(bc.get('prDOF'))

def test_solve():
    degree, n_xi, n_eta, nel, n, m, ncp, gDof, C, P, W, IEN, K, b, D = basic_geometry(refine = 9)
    T_t = 0  # [°C]
    T_b = 1  # [°C]
    T_l = None
    T_r = None
    bc, D = boundary_condition(T_t, T_b, T_l, T_r, D, m, n, ncp)

    ##################
    kappa = np.ones((n, m))   # [W/mK]
    K = form_k(K, IEN, P, kappa, gDof, nel, ncp, degree, W, C)

    D, b = solve(bc, K, b, D)
    print(D)
    T = D.reshape((n,m))
    plt.imshow(T)#, origin="lower")
    plt.colorbar()
    plt.show()

def test_solve():
    degree, n_xi, n_eta, nel, n, m, ncp, gDof, C, P, W, IEN, K, b, D = basic_geometry(refine = 9)
    T_t = 0  # [°C]
    T_b = 1  # [°C]
    T_l = None
    T_r = None
    bc, D = boundary_condition(T_t, T_b, T_l, T_r, D, m, n, ncp)

    ##################
    kappa = np.ones((n, m))   # [W/mK]
    K = form_k(K, IEN, P, kappa, gDof, nel, ncp, degree, W, C)

    D, b = solve(bc, K, b, D)
    print(D)
    T = D.reshape((n,m))
    plt.imshow(T)#, origin="lower")
    plt.colorbar()
    plt.show()

def test_solve_fine_mesh():
    degree, n_xi, n_eta, nel, n, m, ncp, gDof, C, P, W, IEN, K, b, D = basic_geometry(refine=50)

    ##################
    kappa = np.ones((n, m)) * 0.4  # [W/mK]
    K = form_k(K, IEN, P, kappa, gDof, nel, ncp, degree, W, C)

    T_t = 0  # [°C]
    T_b = 1  # [°C]
    T_l = None
    T_r = None
    bc, D = boundary_condition(T_t, T_b, T_l, T_r, D, m, n, ncp)

    D, b = solve(bc, K, b, D)
    print(D)
    T = D.reshape((n,m))
    plt.imshow(T, origin="lower")
    plt.colorbar()
    plt.show()

def test_different_bc():
    degree, n_xi, n_eta, nel, n, m, ncp, gDof, C, P, W, IEN, K, b, D = basic_geometry(refine=20)

    ##################
    kappa = np.ones((n, m)) * 0.4 # [W/mK]
    K = form_k(K, IEN, P, kappa, gDof, nel, ncp, degree, W, C)
    D = np.zeros(gDof)

    ############## Play with the boundary conditions
    T_l = 0.5
    T_r = 0  # [°C]
    T_t = 0
    T_b = 1  # [°C]

    bc, D = boundary_condition(T_t, T_b, T_l, T_r, D, m, n, ncp)

    D, b = solve(bc, K, b, D)

    print(D)
    T = D.reshape((n,m))
    plt.imshow(T, origin="lower")
    plt.colorbar()
    plt.show()

def test_different_bc_2():

    degree, n_xi, n_eta, nel, n, m, ncp, gDof, C, P, W, IEN, K, b, D = basic_geometry(refine=30)

    ##################
    kappa = np.ones((n, m)) * 0.4  # [W/mK]
    K = form_k(K, IEN, P, kappa, gDof, nel, ncp, degree, W, C)

    ############## Play with the boundary conditions
    T_l = 0
    T_r = 0  # [°C]
    T_t = 100
    T_b = 0  # [°C]
    bc, D = boundary_condition(T_t, T_b, T_l, T_r, D, m, n, ncp)

    D, b = solve(bc, K, b, D)

    print(D)
    T = D.reshape((n, m))
    plt.imshow(T, origin="lower",alpha=0.5)
    plt.colorbar()
    cs = plt.contour(T, colors="black")
    plt.clabel(cs)
    plt.show()


def test_different_bc_3():

    degree, n_xi, n_eta, nel, n, m, ncp, gDof, C, P, W, IEN, K, b, D = basic_geometry(refine=50)

    ##################
    kappa = np.ones((n, m)) * 0.4
    K = form_k(K, IEN, P, kappa, gDof, nel, ncp, degree, W, C)

    ############## Play with the boundary conditions
    T_t = 1
    T_l = 1
    T_r = 5
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

def test_kappa_domain():
    degree, n_xi, n_eta, nel, n, m, ncp, gDof, C, P, W, IEN, K, b, D = basic_geometry(refine=9)
    kappa = np.empty((n,m))
    kappa[:int(n/2):,:] = 2.0 # Granite
    kappa[int(n/2):,:] = 0.25 # shale
    kappa_e = kappa_domain(kappa, IEN)
    print(kappa_e)
    plt.imshow(kappa)
    plt.show()

def test_solve_complete_1():
    degree, n_xi, n_eta, nel, n, m, ncp, gDof, C, P, W, IEN, K, b, D = basic_geometry(refine=20)
    kappa = np.empty((n,m))

    kappa[:int(n/3):,:] = 3.4 # Granite
    kappa[int(n/3):-int(n/3),:] = 1.25 # shale
    kappa[-int(n / 3):, :] = 2.25  # shale
    kappa_e = kappa_domain(kappa, IEN)
    print(kappa_e)
    plt.imshow(kappa)
    plt.colorbar()
    plt.show()
    K = form_k(K, IEN, P, kappa, gDof, nel, ncp, degree, W, C)

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

def test_solve_complete_2():
    degree, n_xi, n_eta, nel, n, m, ncp, gDof, C, P, W, IEN, K, b, D = basic_geometry(refine=20)
    kappa = np.empty((n,m))

    kappa[:int(n/3):,:] = 3.4 # Granite
    kappa[int(n/3):-int(n/3),:] = 1.25 # shale
    kappa[-int(n / 3):, :] = 2.25  # shale
    kappa_e = kappa_domain(kappa, IEN)
    print(kappa_e)
    plt.imshow(kappa)
    plt.colorbar()
    plt.show()
    K = form_k(K, IEN, P, kappa, gDof, nel, ncp, degree, W, C)

    ############## Play with the boundary conditions
    T_t = 10
    T_l = None
    T_r = None
    T_b = 40
    bc, D = boundary_condition(T_t, T_b, T_l, T_r, D, m, n, ncp)

    D, b = solve(bc, K, b, D)

    print(D)
    T = D.reshape((n, m))
    plt.imshow(T, origin="lower", alpha=0.8)
    plt.colorbar()
    cs = plt.contour(T, colors="black")
    plt.clabel(cs)
    plt.show()

def test_solve_complete_3():
    t1 = process_time()
    degree, n_xi, n_eta, nel, n, m, ncp, gDof, C, P, W, IEN, K, b, D = basic_geometry(refine=50)
    kappa = np.empty((n,m))

    kappa[:int(n/3):,:] = 3.4 # Granite
    kappa[int(int(n/3)/3):-int(int(n/3)/3)-int(2*n/3),int(m/3):-int(m/3)] = 0.6
    kappa[int(n/3):-int(n/3),:] = 1.25 # shale
    kappa[-int(n / 3):, :] = 2.25  # shale
    kappa_e = kappa_domain(kappa, IEN)
    print(kappa_e)
    plt.imshow(kappa, origin = "lower")
    cb = plt.colorbar()
    cb.set_label("Thermal conductivity [W/mK]")
    plt.show()
    t2 = process_time()
    K = form_k(K, IEN, P, kappa, gDof, nel, ncp, degree, W, C)

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

    print("Time run all script: ", t4 - t1) #18.78125
    print("Time run IGA: ", t3 - t2) #17.125

def test_solve_complete_2():
    knots, B = quarter_disk()
    degree, n_xi, n_eta, nel, n, m, ncp, gDof, C, P, W, IEN, K, b, D, B, knots = analyze_bezier(knots, B, refine=20)

    kappa = np.ones((n,m))
    K = form_k(K, IEN, P, kappa, gDof, nel, ncp, degree, W, C)

    ############## Play with the boundary conditions
    T_t = 10
    T_l = None
    T_r = None
    T_b = 40
    bc, D = boundary_condition(T_t, T_b, T_l, T_r, D, m, n, ncp)
    D, b = solve(bc, K, b, D)
    T = D.reshape((n, m))
    plt.imshow(T, origin="lower", alpha=0.8)
    cb = plt.colorbar()
    cb.set_label("Temperature [°C]")
    cs = plt.contour(T, colors="black")
    plt.clabel(cs)
    plt.show()



