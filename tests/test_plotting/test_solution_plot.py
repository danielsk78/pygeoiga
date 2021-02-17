from pygeoiga.analysis.bezier_FE import map_bezier_elements
from pygeoiga.plot.solution_mpl import *
from pygeoiga.nurb.cad import  *
from pygeoiga.analysis.MultiPatch import patch_topology, form_k_IGA_mp, boundary_condition_mp, map_MP_elements
from pygeoiga.analysis.common import solve, boundary_condition
from pygeoiga.analysis.iga import form_k_IGA, map_solution_elements
from pygeoiga.analysis.bezier_FE import form_k


def test_plot_solution_bezier():
    knots, B = make_surface_biquadratic()
    from pygeoiga.nurb.nurb_creation import NURB

    from pygeoiga.analysis.bezier_FE import analyze_bezier
    degree, n_xi, n_eta, nel, n, m, ncp, gDof, C, P, W, IEN, K, b, D, B, knots = analyze_bezier(knots, B, refine=10)

    kappa = 5
    K = form_k(K, IEN, P, kappa, nel, degree, W, C)
    ############## Play with the boundary conditions
    T_t =None
    T_l = 5
    T_r = 10
    T_b =  None #lambda x, m: T_l + 10 * np.sin(np.pi * x / m)  # [°C]
    bc, D = boundary_condition(T_t, T_b, T_l, T_r, D, m, n, ncp)
    D, b = solve(bc, K, b, D)
    x, y, t = map_bezier_elements(D, degree, P, m,n, ncp, nel, IEN, W, C)

    from pygeoiga.plot.nrbplotting_mpl import p_knots, p_cpoints
    fig, ax = plt.subplots()
    ax = p_cpoints(B, ax=ax, dim=2, color="red", marker=".", markersize=5, point=True, line=False)
    ax = p_knots(knots, B, ax=ax, dim=2, color="black", linestyle="--", point=False, line=True)
    ax = p_temperature(x, y, t, levels=100, show=False, colorbar=True, ax=ax, point=True, fill=False)
    plt.show()

    fig, ax = plt.subplots()
    #fig, ax = create_figure()
    ax = p_cpoints(B, ax=ax, dim=2, color="red", marker=".", markersize=5, point=True, line=False)
    ax = p_knots(knots, B, ax=ax, dim=2, color="black", linestyle="--", point=False, line=True)
    ax = p_temperature(x, y, t, levels=100, show=False, colorbar=True, ax = ax, point = True, fill=True)
    plt.show()

def test_plot_solution_bezier_square():
    U, V, B = make_surface_square()
    knots= (U,V)
    from pygeoiga.analysis.bezier_FE import analyze_bezier
    degree, n_xi, n_eta, nel, n, m, ncp, gDof, C, P, W, IEN, K, b, D, B, knots = analyze_bezier(knots, B, refine=10)

    kappa = 5
    K = form_k(K, IEN, P, kappa, nel, degree, W, C)
    ############## Play with the boundary conditions
    T_t =10
    T_l = None
    T_r = None
    T_b =  40 #lambda x, m: T_l + 10 * np.sin(np.pi * x / m)  # [°C]
    bc, D = boundary_condition(T_t, T_b, T_l, T_r, D, m, n, ncp)
    D, b = solve(bc, K, b, D)
    x, y, t = map_bezier_elements(D, degree, P, n, m, ncp, nel, IEN, W, C)

    from pygeoiga.plot.nrbplotting_mpl import p_knots, p_cpoints
    fig, ax = plt.subplots()
    ax = p_cpoints(B, ax=ax, dim=2, color="red", marker=".", markersize=5, point=True, line=False)
    ax = p_knots(knots, B, ax=ax, dim=2, color="black", linestyle="--", point=False, line=True)
    plt.show()

    fig, ax = plt.subplots()
    #fig, ax = create_figure()
    ax = p_cpoints(B, ax=ax, dim=2, color="red", marker=".", markersize=5, point=True, line=False)
    ax = p_knots(knots, B, ax=ax, dim=2, color="black", linestyle="--", point=False, line=True)
    ax = p_temperature(x, y, t, levels=100, show=False, colorbar=True, ax = ax, point = True, fill=True)
    plt.show()

def test_plot_solution_IGA():
    knots, B = make_surface_biquadratic()
    from pygeoiga.analysis.iga import analyze_IGA
    degree, n_xi, n_eta, nel, n, m, ncp, gDof, P, W, IEN, K, b, D, knots, B = analyze_IGA(knots, B, refine=5, _refine=True)
    print(knots)
    kappa = 5
    K = form_k_IGA(K, IEN, P, kappa, n_xi, n_eta, degree, knots=knots)
    ############## Play with the boundary conditions
    T_t = 10
    T_l = None
    T_r = None
    T_b = 40  # lambda x, m: T_t + 10 * np.sin(np.pi * x / m)  # [°C]

    bc, D = boundary_condition(T_t, T_b, T_l, T_r, D, m, n, ncp)
    D, b = solve(bc, K, b, D)

    x, y, t = map_solution_elements(D, degree, P, n_xi, n_eta, n, m, ncp, IEN, W, knots)

    from pygeoiga.plot.nrbplotting_mpl import p_knots, p_cpoints
    fig, ax = plt.subplots()
    ax = p_cpoints(B, ax=ax, dim=2, color="red", marker=".", markersize=5, point=True, line=False)
    ax = p_knots(knots, B, ax=ax, dim=2, color="black", linestyle="--", point=False, line=True)
    ax = p_temperature(x, y, t, levels=10, show=False, colorbar=True, ax=ax, point=True, fill=False)

    plt.show()

    fig, ax = plt.subplots()
    # fig, ax = create_figure()
    ax = p_cpoints(B, ax=ax, dim=2, color="red", marker=".", markersize=5, point=True, line=False)
    ax = p_knots(knots, B, ax=ax, dim=2, color="black", linestyle="--", point=False, line=True)
    ax = p_temperature(x, y, t, levels=100, show=False, colorbar=True, ax=ax, point = True, fill=True)
    plt.show()

def test_plot_solution_IGA_square():
    U, V, B = make_surface_square()
    knots = (U,V)
    from pygeoiga.analysis.iga import analyze_IGA
    degree, n_xi, n_eta, nel, n, m, ncp, gDof, P, W, IEN, K, b, D, knots, B = analyze_IGA(knots, B, refine=10)

    kappa = 5
    K = form_k_IGA(K, IEN, P, kappa, n_xi, n_eta, degree, knots=knots)
    ############## Play with the boundary conditions
    T_t = 10
    T_b = 40  # lambda x, m: T_t + 10 * np.sin(np.pi * x / m)  # [°C]
    T_l = None
    T_r = None

    bc, D = boundary_condition(T_t, T_b, T_l, T_r, D, m, n, ncp)
    D, b = solve(bc, K, b, D)

    x, y, t = map_solution_elements(D, degree, P, n_xi, n_eta, n, m, ncp, IEN, W, knots)

    from pygeoiga.plot.nrbplotting_mpl import p_knots, p_cpoints
    fig, ax = plt.subplots()
    ax = p_cpoints(B, ax=ax, dim=2, color="red", marker=".", markersize=5, point=True, line=False)
    ax = p_knots(knots, B, ax=ax, dim=2, color="black", linestyle="--", point=False, line=True)
    ax = p_temperature(x, y, t, levels=100, show=False, colorbar=True, ax=ax, point=True, fill=False)

    plt.show()

    fig, ax = plt.subplots()
    # fig, ax = create_figure()
    ax = p_cpoints(B, ax=ax, dim=2, color="red", marker=".", markersize=5, point=True, line=False)
    ax = p_knots(knots, B, ax=ax, dim=2, color="black", linestyle="--", point=False, line=True)
    ax = p_temperature(x, y, t, levels=100, show=False, colorbar=True, ax=ax, point=True, fill=True)
    plt.show()

def test_plot_solution_3_layer_mp():
    geometry = make_3_layer_patches(refine=True, )
    geometry, gDoF = patch_topology(geometry)
    K_glob = np.zeros((gDoF, gDoF))
    K_glob = form_k_IGA_mp(geometry, K_glob)

    D = np.zeros(gDoF)
    b = np.zeros(gDoF)

    T_t = 10  # [°C]
    T_b = 25  # [°C]
    T_l = None#10
    T_r = None#40
    bc, D = boundary_condition_mp(geometry, D, T_t, T_b, T_l, T_r)
    bc["gDOF"] = gDoF
    D, b = solve(bc, K_glob, b, D)

    geometry = map_MP_elements(geometry, D)
    from pygeoiga.plot.nrbplotting_mpl import p_surface, p_knots, create_figure
    fig, ax = create_figure("2d")
    #ax.view_init(azim=270, elev=90)
    fig_sol, ax_sol = create_figure("2d")
    geometrypatch = make_3_layer_patches(refine=False)
    figpatch, axpatch = create_figure("2d")
    for patch_id in geometry.keys():
        ax = p_surface(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2,
                       color=geometry[patch_id].get("color"), alpha=0.5)
        #ax = p_cpoints(geometry[patch_id].get("B"), ax=ax, dim=2, color="black", marker=".", point=True, line=False)
        ax = p_knots(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2, point=False,
                         line=True)

        axpatch = p_surface(geometrypatch[patch_id].get("knots"), geometrypatch[patch_id].get("B"), ax=axpatch, dim=2,
                       color=geometrypatch[patch_id].get("color"), alpha=0.5)
        #axpatch = p_cpoints(geometry[patch_id].get("B"), ax=axpatch, dim=2, color="black", marker=".", point=True, line=False)
        axpatch = p_knots(geometrypatch[patch_id].get("knots"), geometrypatch[patch_id].get("B"), ax=axpatch, dim=2, point=False,
                     line=True)

        #ax_sol = p_cpoints(geometry[patch_id].get("B"), ax=ax_sol, dim=2, color=geometry[patch_id].get("color"), marker=".", point=True, line=False)
        ax_sol = p_surface(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax_sol, dim=2,
                       color="k", fill=False)
        x = geometry[patch_id].get("x_sol")
        y = geometry[patch_id].get("y_sol")
        t = geometry[patch_id].get("t_sol")
        ax_sol = p_temperature(x,y,t, vmin = np.min(D), vmax = np.max(D), levels=50, show=False, colorbar=True, ax=ax_sol,
                               point = False, fill=True)#, color = "k")

    fig.show()
    fig_sol.show()
    figpatch.show()

    fig_all, ax_all = create_figure("2d")
    p_temperature_mp(geometry=geometry, vmin=np.min(D), vmax=np.max(D), levels=200, show=False, colorbar=True,
                     ax=ax_all, point=False, fill=True, contour=False)
    plt.show()

    #for patch_id in geometry.keys():

     #   fig_patch, ax_patch = plt.subplots()
     #   ax_patch = p_cpoints(geometry[patch_id].get("B"), ax=ax_patch, dim=2, color="black", marker=".", point=True,
     #                        line=False)
     #   ax_patch = p_knots(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax_patch, dim=2, point=False,
     #                      line=True)
     #   x = geometry[patch_id].get("x_sol")
     #   y = geometry[patch_id].get("y_sol")
     #   t = geometry[patch_id].get("t_sol")
     #   ax_patch = p_temperature(x, y, t, levels=50, show=False, colorbar=True, point=True, ax=ax_patch,
     #                            fill=True, color="k", vmin=np.min(t), vmax=np.max(t))
     #   plt.show()


def test_plot_solution_mp_3d():
    import matplotlib
    matplotlib.use('Qt5Agg')
    geometry = make_3_layer_patches()
    geometry, gDoF = patch_topology(geometry)
    K_glob = np.zeros((gDoF, gDoF))
    K_glob = form_k_IGA_mp(geometry, K_glob)

    D = np.zeros(gDoF)
    b = np.zeros(gDoF)

    T_t = 10  # [°C]
    T_b = lambda x, m: T_t + 10 * np.sin(np.pi * x / m)  # [°C]
    T_l = None
    T_r = None
    bc, D = boundary_condition_mp(geometry, D, T_t, T_b, T_l, T_r)
    bc["gDOF"] = gDoF
    D, b = solve(bc, K_glob, b, D)

    geometry = map_MP_elements(geometry, D)
    from pygeoiga.plot.nrbplotting_mpl import p_knots, create_figure
    fig, ax = create_figure()
    for patch_id in geometry.keys():
        ax = p_knots(geometry[patch_id].get("knots"), geometry[patch_id].get("B"),
                         color = geometry[patch_id].get("color"), ax=ax, dim=2, point=False, line=True)
        x = geometry[patch_id].get("x_sol")
        y = geometry[patch_id].get("y_sol")
        t = geometry[patch_id].get("t_sol")
        ax = p_temperature(x,y,t, vmin = np.min(D), vmax = np.max(D), levels=50, show=False, colorbar=True, ax=ax,
                               point = True, fill=True, color = "k")

    plt.show()


def test_plot_L_MP():
    geometry = make_L_shape(refine=True, knot_ins=(np.arange(0.1,0.9,0.1), np.arange(0.1,0.9,0.1)))
    geometry, gDoF = patch_topology(geometry)
    K_glob = np.zeros((gDoF, gDoF))
    K_glob = form_k_IGA_mp(geometry, K_glob)

    D = np.zeros(gDoF)
    b = np.zeros(gDoF)

    T_t = 10  # [°C]
    T_b = 40  # [°C]
    T_l = 5
    T_r = 10
    bc, D = boundary_condition_mp(geometry, D, T_t, T_b, T_l, T_r)
    bc["gDOF"] = gDoF
    D, b = solve(bc, K_glob, b, D)

    geometry = map_MP_elements(geometry, D)
    from pygeoiga.plot.nrbplotting_mpl import p_surface, p_cpoints, p_knots, create_figure
    fig, ax = create_figure("2d")
    fig_sol, ax_sol = create_figure("2d")  #
    fig_point, ax_point = create_figure("2d")
    for patch_id in geometry.keys():
        ax = p_surface(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2,
                       color=geometry[patch_id].get("color"), alpha=0.5)
        ax = p_cpoints(geometry[patch_id].get("B"), ax=ax, dim=2, color="black", marker=".", point=True, line=False)
        ax = p_knots(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2, point=False,
                     line=True)

        # ax_sol = p_cpoints(geometry[patch_id].get("B"), ax=ax_sol, dim=2, color=geometry[patch_id].get("color"), marker=".", point=True, line=False)
        # ax_sol = p_knots(geometry[patch_id].get("knots"), geometry[patch_id].get("B"),
        #                 color = geometry[patch_id].get("color"), ax=ax_sol, dim=2, point=False, line=True)
        x = geometry[patch_id].get("x_sol")
        y = geometry[patch_id].get("y_sol")
        t = geometry[patch_id].get("t_sol")
        ax_point = p_temperature(x, y, t, vmin=np.min(D), vmax=np.max(D), levels=100, show=False, colorbar=True,
                               ax=ax_point, point=True, fill=False)
        ax_sol = p_temperature(x, y, t, vmin=np.min(D), vmax=np.max(D), levels=100, show=False, colorbar=True,
                               ax=ax_sol, point=False, fill=True)
        #ax_sol = p_temperature(x, y, t, vmin=np.min(D), vmax=np.max(D), levels=5, show=False, colorbar=False,
        #                       ax=ax_sol, point=False, fill=False, contour=True)

    fig.show()
    fig_sol.show()
    fig_point.show()

    fig_all, ax_all = create_figure("2d")
    p_temperature_mp(geometry=geometry, vmin=np.min(D), vmax=np.max(D), levels=50, show=False, colorbar=True, ax=ax_all,
                     point=False, fill=True, contour=False)
    fig_all.show()

def test_plot_solution_mp_fault():
    geometry = make_fault_model(refine=True)
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
    from pygeoiga.plot.nrbplotting_mpl import p_surface, p_cpoints, p_knots, create_figure
    fig, ax = create_figure("2d")
    fig_sol, ax_sol = create_figure("2d")#
    fig_point, ax_point = create_figure("2d")  #
    for patch_id in geometry.keys():
        ax = p_surface(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2,
                       color=geometry[patch_id].get("color"), alpha=0.5)
        ax = p_cpoints(geometry[patch_id].get("B"), ax=ax, dim=2, color="black", marker=".", point=True, line=False)
        ax = p_knots(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2, point=False,
                         line=True)

        #ax_sol = p_cpoints(geometry[patch_id].get("B"), ax=ax_sol, dim=2, color=geometry[patch_id].get("color"), marker=".", point=True, line=False)
        #ax_sol = p_knots(geometry[patch_id].get("knots"), geometry[patch_id].get("B"),
        #                 color = geometry[patch_id].get("color"), ax=ax_sol, dim=2, point=False, line=True)
        x = geometry[patch_id].get("x_sol")
        y = geometry[patch_id].get("y_sol")
        t = geometry[patch_id].get("t_sol")
        ax_sol = p_temperature(x,y,t, vmin = np.min(D), vmax = np.max(D), levels=50, show=False, colorbar=True, ax=ax_sol,
                               point = False, fill=True, contour=False)
        ax_point = p_temperature(x, y, t, vmin=np.min(D), vmax=np.max(D), levels=100, show=False, colorbar=True, ax=ax_point,
                      point=True, fill=False)
        ax_sol = p_surface(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax_sol, dim=2,
                       color="k", alpha=1, fill=False, border=True)

    fig.show()
    fig_sol.show()
    fig_point.show()

    fig_all, ax_all = create_figure("2d")
    p_temperature_mp(geometry=geometry, vmin=np.min(D), vmax=np.max(D), levels=200, show=False, colorbar=True, ax=ax_all,
                     point=False, fill=True, contour=False)
    fig_all.show()

def test_plot_solution_mp_dome():
    from pygeoiga.plot.nrbplotting_mpl import p_surface, p_cpoints, p_knots, create_figure
    geometry_init = make_salt_dome(refine=False)
    fig, ax = create_figure("2d")
    fig2, ax2 = create_figure("2d")
    for patch_id in geometry_init.keys():
        ax = p_surface(geometry_init[patch_id].get("knots"), geometry_init[patch_id].get("B"), ax=ax, dim=2,
                       color=geometry_init[patch_id].get("color"), alpha=0.5)
        ax = p_cpoints(geometry_init[patch_id].get("B"), ax=ax, dim=2, color="black", marker=".", point=True, line=False)
        ax = p_knots(geometry_init[patch_id].get("knots"), geometry_init[patch_id].get("B"), ax=ax, dim=2, point=False,
                         line=True)
        ax2 = p_surface(geometry_init[patch_id].get("knots"), geometry_init[patch_id].get("B"), ax=ax2, dim=2,
                       color=geometry_init[patch_id].get("color"), alpha=0.5)
    fig.show()
    fig2.show()
    geometry = make_salt_dome(refine=True, knot_ins=[np.arange(0.3,1,0.3), np.arange(0.3,1,0.3)])#refine=np.arange(0.05,1,0.05))
    geometry, gDoF = patch_topology(geometry)
    K_glob = np.zeros((gDoF, gDoF))
    K_glob = form_k_IGA_mp(geometry, K_glob)

    D = np.zeros(gDoF)
    b = np.zeros(gDoF)

    T_t = 10  # [°C]
    T_b = 90  # [°C]
    T_l = None
    T_r = None
    bc, D = boundary_condition_mp(geometry, D, T_t, T_b, T_l, T_r)
    bc["gDOF"] = gDoF
    D, b = solve(bc, K_glob, b, D)

    geometry = map_MP_elements(geometry, D)

    fig, ax = create_figure("2d")
    fig_sol, ax_sol = create_figure("2d")#
    fig_point, ax_point = create_figure("2d")  #
    for patch_id in geometry.keys():
        ax = p_surface(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2,
                       color=geometry[patch_id].get("color"), alpha=0.5)
        ax = p_cpoints(geometry[patch_id].get("B"), ax=ax, dim=2, color="black", marker=".", point=True, line=False)
        ax = p_knots(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2, point=False,
                         line=True)

        #ax_sol = p_cpoints(geometry[patch_id].get("B"), ax=ax_sol, dim=2, color=geometry[patch_id].get("color"), marker=".", point=True, line=False)
        #ax_sol = p_knots(geometry[patch_id].get("knots"), geometry[patch_id].get("B"),
        #                 color = geometry[patch_id].get("color"), ax=ax_sol, dim=2, point=False, line=True)
        x = geometry[patch_id].get("x_sol")
        y = geometry[patch_id].get("y_sol")
        t = geometry[patch_id].get("t_sol")
        ax_sol = p_temperature(x,y,t, vmin = np.min(D), vmax = np.max(D), levels=50, show=False, colorbar=True, ax=ax_sol,
                               point = False, fill=True, contour=False)
        ax_point = p_temperature(x, y, t, vmin=np.min(D), vmax=np.max(D), levels=100, show=False, colorbar=True, ax=ax_point,
                      point=True, fill=False)
        #ax_sol = p_surface(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax_sol, dim=2,
        #               color="k", alpha=1, fill=False, border=True)
    fig.show()
    fig_sol.show()
    fig_point.show()

    fig_all, ax_all = create_figure("2d")
    p_temperature_mp(geometry=geometry, vmin=np.min(D), vmax=np.max(D), levels=200, show=False, colorbar=True, ax=ax_all,
                     point=False, fill=True, contour=False)
    fig_all.show()
    print(gDoF)

def test_plot_solution_unconformity():
    from pygeoiga.plot.nrbplotting_mpl import p_surface, p_cpoints, p_knots, create_figure
    geometry_init = make_unconformity_model(refine=False)
    fig, ax = create_figure("2d")
    for patch_id in geometry_init.keys():
        ax = p_surface(geometry_init[patch_id].get("knots"), geometry_init[patch_id].get("B"), ax=ax, dim=2,
                       color=geometry_init[patch_id].get("color"), alpha=0.5)
        ax = p_cpoints(geometry_init[patch_id].get("B"), ax=ax, dim=2, color="black", marker=".", point=True, line=False)
        ax = p_knots(geometry_init[patch_id].get("knots"), geometry_init[patch_id].get("B"), ax=ax, dim=2, point=False,
                         line=True)
    fig.show()
    geometry = make_unconformity_model(refine=True)#refine=np.arange(0.05,1,0.05))
    geometry, gDoF = patch_topology(geometry)
    K_glob = np.zeros((gDoF, gDoF))
    K_glob = form_k_IGA_mp(geometry, K_glob)

    D = np.zeros(gDoF)
    b = np.zeros(gDoF)

    T_t = 10  # [°C]
    T_b = 60  # [°C]
    T_l = None
    T_r = None
    bc, D = boundary_condition_mp(geometry, D, T_t, T_b, T_l, T_r)
    bc["gDOF"] = gDoF
    D, b = solve(bc, K_glob, b, D)

    geometry = map_MP_elements(geometry, D)

    fig, ax = create_figure("2d")
    fig_sol, ax_sol = create_figure("2d")#
    fig_point, ax_point = create_figure("2d")  #
    for patch_id in geometry.keys():
        ax = p_surface(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2,
                       color=geometry[patch_id].get("color"), alpha=0.5)
        ax = p_cpoints(geometry[patch_id].get("B"), ax=ax, dim=2, color="black", marker=".", point=True, line=False)
        ax = p_knots(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2, point=False,
                         line=True)

        #ax_sol = p_cpoints(geometry[patch_id].get("B"), ax=ax_sol, dim=2, color=geometry[patch_id].get("color"), marker=".", point=True, line=False)
        #ax_sol = p_knots(geometry[patch_id].get("knots"), geometry[patch_id].get("B"),
        #                 color = geometry[patch_id].get("color"), ax=ax_sol, dim=2, point=False, line=True)
        x = geometry[patch_id].get("x_sol")
        y = geometry[patch_id].get("y_sol")
        t = geometry[patch_id].get("t_sol")
        #if patch_id=="E3":
        #    t=t.T
        ax_sol = p_temperature(x,y,t, vmin = np.min(D), vmax = np.max(D), levels=50, show=False, colorbar=True, ax=ax_sol,
                               point = False, fill=True, contour=False)
        ax_point = p_temperature(x, y, t, vmin=np.min(D), vmax=np.max(D), levels=100, show=False, colorbar=True, ax=ax_point,
                      point=True, fill=False)
        ax_sol = p_surface(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax_sol, dim=2,
                       color="k", alpha=1, fill=False, border=True)
    fig.show()
    fig_sol.show()
    fig_point.show()

    fig_all, ax_all = create_figure("2d")
    p_temperature_mp(geometry=geometry, vmin=np.min(D), vmax=np.max(D), levels=50, show=False, colorbar=True, ax=ax_all,
                     point=False, fill=True, contour=False)
    fig_all.show()

def test_plot_field_iga_3layer():
    geometry = make_3_layer_patches(refine=True, )
    geometry, gDoF = patch_topology(geometry)
    K_glob = np.zeros((gDoF, gDoF))
    K_glob = form_k_IGA_mp(geometry, K_glob)

    D = np.zeros(gDoF)
    b = np.zeros(gDoF)

    T_t = 10  # [°C]
    T_b = 25  # [°C]
    T_l = None  # 10
    T_r = None  # 40
    bc, D = boundary_condition_mp(geometry, D, T_t, T_b, T_l, T_r)
    bc["gDOF"] = gDoF
    D, b = solve(bc, K_glob, b, D)

    geometry = map_MP_elements(geometry, D)

    from pygeoiga.analysis.MultiPatch import point_solution_mp
    x = np.arange(0, 500, 25)
    y = np.arange(0, 500, 25)
    x = x[1:-1]
    y = y[1:-1]
    xx, yy = np.meshgrid(x,y)
    tt = np.empty(xx.shape)
    for i in range(len(x)):
        for j in range(len(y)):
            tt[i,j] = point_solution_mp(xx[i,j], yy[i,j], geometry)
    #tt = point_solution_mp(xx,yy,geometry)
    plt.imshow(tt, cmap="viridis", origin="lower")
    plt.colorbar()
    plt.show()

def test_plot_field_iga_fault():
    geometry = make_fault_model(refine=True, )
    geometry, gDoF = patch_topology(geometry)
    K_glob = np.zeros((gDoF, gDoF))
    K_glob = form_k_IGA_mp(geometry, K_glob)

    D = np.zeros(gDoF)
    b = np.zeros(gDoF)

    T_t = 10  # [°C]
    T_b = 40  # [°C]
    T_l = None  # 10
    T_r = None  # 40
    bc, D = boundary_condition_mp(geometry, D, T_t, T_b, T_l, T_r)
    bc["gDOF"] = gDoF
    D, b = solve(bc, K_glob, b, D)

    geometry = map_MP_elements(geometry, D)

    from pygeoiga.analysis.MultiPatch import point_solution_mp
    x = np.arange(0, 1000, 50)
    y = np.arange(0, 1000, 50)
    x = x[1:-1]
    y = y[1:-1]
    xx, yy = np.meshgrid(x,y)
    tt = np.empty(xx.shape)
    for i in range(len(x)):
        for j in range(len(y)):
            tt[i,j] = point_solution_mp(xx[i,j], yy[i,j], geometry)
    #tt = point_solution_mp(xx,yy,geometry)
    plt.imshow(tt, cmap="viridis", origin="lower")
    plt.colorbar()
    plt.show()

def test_plot_field_iga_salt():
    geometry = make_salt_dome(refine=True, )
    geometry, gDoF = patch_topology(geometry)
    K_glob = np.zeros((gDoF, gDoF))
    K_glob = form_k_IGA_mp(geometry, K_glob)

    D = np.zeros(gDoF)
    b = np.zeros(gDoF)

    T_t = 10  # [°C]
    T_b = 90  # [°C]
    T_l = None  # 10
    T_r = None  # 40
    bc, D = boundary_condition_mp(geometry, D, T_t, T_b, T_l, T_r)
    bc["gDOF"] = gDoF
    D, b = solve(bc, K_glob, b, D)

    geometry = map_MP_elements(geometry, D)

    from pygeoiga.analysis.MultiPatch import point_solution_mp
    x = np.arange(0, 6000, 200)
    y = np.arange(0, 3000, 200)
    x = x[1:-1]
    y = y[1:-1]
    xx, yy = np.meshgrid(x,y)
    tt = np.empty(xx.shape)
    for i in range(len(x)):
        for j in range(len(y)):
            tt[i,j] = point_solution_mp(xx[i,j], yy[i,j], geometry)
    #tt = point_solution_mp(xx,yy,geometry)
    plt.imshow(tt, cmap="viridis", origin="lower")
    plt.colorbar()
    plt.show()