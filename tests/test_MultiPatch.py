from pygeoiga.nurb.cad import make_3_layer_patches, make_fault_model, make_salt_dome, make_unconformity_model
from pygeoiga.nurb import NURB
from pygeoiga.analysis.common import IEN_element_topology, solve
import matplotlib.pyplot as plt
import numpy as np
from pygeoiga.analysis.MultiPatch import *
from pygeoiga.analysis.iga import *
from time import process_time

def test_visualize_patches():
    geometry = make_3_layer_patches()
    resolution = 20
    fig = plt.figure("first")
    ax = fig.add_subplot(111, projection='3d')

    for patch_id in geometry.keys():
        B = geometry[patch_id].get("B")
        knots = geometry[patch_id].get("knots")
        nurb = NURB(B, knots, resolution=resolution, engine="python")
        ax.plot(nurb.model[:, 0], nurb.model[:, 1], nurb.model[:, 2], linestyle='None', marker='.')#, color='blue')
        ax.plot_wireframe(nurb.cpoints[..., 0], nurb.cpoints[..., 1], nurb.cpoints[..., 2])#, color='red')
    fig.show()

def test_conectivity_arrays():
    geometry = make_3_layer_patches()
    geometry, _ = patch_topology(geometry)
    for patch_id in geometry.keys():
        print(patch_id, " have: ", geometry[patch_id]["patch_topology"])

def test_fill_info():
    geometry = make_3_layer_patches()
    geometry, _ = patch_topology(geometry)
    for patch_id in geometry.keys():
        print(patch_id, " have: ", geometry[patch_id]["patch_DOF"])
        print(patch_id, " have: ", geometry[patch_id]["n_element"])

def test_get_common_cp():
    geometry = make_3_layer_patches()
    geometry, _ = patch_topology(geometry)
    for patch_id in geometry.keys():
        print(patch_id, " have: ", geometry[patch_id]["cp_repeated"])

def test_glob_num():
    geometry = make_3_layer_patches()
    geometry, global_degrees_freedom = patch_topology(geometry)
    print("Geometry have a total of: ", global_degrees_freedom, " DOF")
    for patch_id in geometry.keys():
        print(patch_id, " have glob_num: ", geometry[patch_id]["glob_num"])


def test_stiffness_matrix():
    geometry = make_3_layer_patches()
    geometry, gDoF = patch_topology(geometry)

    K_glob = np.zeros((gDoF, gDoF))
    npatches = len(geometry)
    #form the stiffness matrix
    for patch_id in geometry.keys():
        B = geometry[patch_id].get("B")
        U, V = geometry[patch_id].get("knots")

        ### For the moment we will not make a refinement

        degree_u = len(np.where(U == 0.)[0]) - 1
        degree_v = len(np.where(V == 0.)[0]) - 1

        # n_xi and n_eta are the number of elements
        n_xi = B.shape[0] - degree_u
        n_eta = B.shape[1] - degree_v
        nel = n_xi * n_eta  # total number of elements
        n = n_xi + degree_u  # Number of basis functions/ control points in xi direction
        m = n_eta + degree_v  # Number of basis functions/ control points in eta direction
        ncp = n * m  # Total number of control points
        pDof = ncp  # patch degrees of freedom - Temperature

        K = np.zeros((pDof, pDof))  # stiffness matrix size
        degree = degree_u

        P = geometry[patch_id].get("list_cp")
        W = geometry[patch_id].get("list_weight")

        # Element topology
        IEN = IEN_element_topology(n_xi, n_eta, degree)

        b = np.zeros(pDof)  # load vector size
        D = np.zeros(pDof)
        kappa = np.ones((n, m)) * geometry[patch_id].get("kappa")
        K = form_k_IGA(K, IEN, P, kappa, n_xi, n_eta, degree)

        patch_glob_num = geometry[patch_id].get("glob_num")

        K_glob[np.ix_(patch_glob_num, patch_glob_num)] = K_glob[np.ix_(patch_glob_num, patch_glob_num)] + K #[np.ix_(patch_DOF, patch_DOF)]


    plt.spy(K_glob)
    plt.show()

def test_form_k():
    geometry = make_3_layer_patches()
    geometry, gDoF = patch_topology(geometry)
    K_glob = np.zeros((gDoF, gDoF))
    K_glob = form_k_IGA_mp(geometry, K_glob)
    plt.spy(K_glob)
    plt.show()

def test_boundary_condition():
    geometry = make_3_layer_patches()
    geometry, gDoF = patch_topology(geometry)

    D = np.zeros(gDoF)
    T_t = 10  # [°C]
    T_b = 40  # [°C]
    T_l = None
    T_r = None
    prDOF, D = boundary_condition_mp(geometry, D, T_t, T_b, T_l, T_r)
    print(prDOF, D)

def test_solve_mp():
    geometry = make_3_layer_patches()
    geometry, gDoF = patch_topology(geometry)
    K_glob = np.zeros((gDoF, gDoF))
    K_glob = form_k_IGA_mp(geometry, K_glob)

    D = np.zeros(gDoF)
    b = np.zeros(gDoF)

    T_t = 10  # [°C]
    T_b = 40  # [°C]
    T_l = None
    T_r = None
    bc, D = boundary_condition_mp(geometry, D, T_t, T_b, T_l, T_r)
    bc["gDOF"] = gDoF
    D, b = solve(bc, K_glob, b, D)
    plt.spy(K_glob)
    plt.show()
    print(D)

def test_solve_fault():
    geometry = make_fault_model()
    geometry, gDoF = patch_topology(geometry)
    K_glob = np.zeros((gDoF, gDoF))
    K_glob = form_k_IGA_mp(geometry, K_glob)

    D = np.zeros(gDoF)
    b = np.zeros(gDoF)

    T_t = 10  # [°C]
    T_b = 40  # [°C]
    T_l = None
    T_r = None
    bc, D = boundary_condition_mp(geometry, D, T_t, T_b, T_l, T_r)
    bc["gDOF"] = gDoF
    D, b = solve(bc, K_glob, b, D)

    plt.spy(K_glob)
    plt.show()
    print(D)

def test_solve_salt_dome():
    geometry = make_salt_dome(refine=True)
    geometry, gDoF = patch_topology(geometry)
    K_glob = np.zeros((gDoF, gDoF))
    K_glob = form_k_IGA_mp(geometry, K_glob)

    D = np.zeros(gDoF)
    b = np.zeros(gDoF)

    T_t = 10  # [°C]
    T_b = 40  # [°C]
    T_l = None
    T_r = None
    bc, D = boundary_condition_mp(geometry, D, T_t, T_b, T_l, T_r)
    bc["gDOF"] = gDoF
    D, b = solve(bc, K_glob, b, D)

    plt.spy(K_glob)
    plt.show()
    print(D)

def test_solve_unconformity():
    geometry = make_unconformity_model(refine=True)
    geometry, gDoF = patch_topology(geometry)
    K_glob = np.zeros((gDoF, gDoF))
    K_glob = form_k_IGA_mp(geometry, K_glob)

    D = np.zeros(gDoF)
    b = np.zeros(gDoF)

    T_t = 10  # [°C]
    T_b = 40  # [°C]
    T_l = None
    T_r = None
    bc, D = boundary_condition_mp(geometry, D, T_t, T_b, T_l, T_r)
    bc["gDOF"] = gDoF
    D, b = solve(bc, K_glob, b, D)

    plt.spy(K_glob)
    plt.show()
    print(D)


def test_get_point_solution_mp():
    geometry = make_3_layer_patches(refine=True)
    geometry, gDoF = patch_topology(geometry)
    K_glob = np.zeros((gDoF, gDoF))
    K_glob = form_k_IGA_mp(geometry, K_glob)

    D = np.zeros(gDoF)
    b = np.zeros(gDoF)

    T_t = 10  # [°C]
    T_b = 40  # [°C]
    T_l = None
    T_r = None
    bc, D = boundary_condition_mp(geometry, D, T_t, T_b, T_l, T_r)
    bc["gDOF"] = gDoF
    D, b = solve(bc, K_glob, b, D)

    geometry = map_MP_elements(geometry, D)

    from pygeoiga.analysis.MultiPatch import point_solution_mp

    t = point_solution_mp(0, 0, geometry, tolerance=1e-9, itera =1000)
    print(t) # knots (0.5548848818987607, 0.7526260007172824),
    # approx value (5.999999999923407, 4.999999997422794),
    # itera = 84,
    # stepx and y 3.7252902984619143e-10, 3.7252902984619143e-10,
    # temperature = 4.435870165147437)

