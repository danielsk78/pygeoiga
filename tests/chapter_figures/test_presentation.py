import pygeoiga as gn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
fig_folder=gn.myPath+'/../../../../manuscript/Thesis/figures/04_NURBS'
dpi=200

def make_plot(control, knot, basis_fun, resol, positions):
    from  pygeoiga.plot.nrbplotting_mpl import p_knots
    dpi=200
    figsize=(5,6)
    fig, (cp, kn, bas) = plt.subplots(3, 1, dpi=dpi, figsize=figsize)
    bas.plot(resol, basis_fun)
    bas.spines["top"].set_visible(False)
    bas.spines["right"].set_visible(False)
    bas.set_xticks([])
    bas.set_xlim(0,1)
    bas.set_ylim(0,1)
    # bas.set_xticklabels([])
    bas.set_yticks([])
    bas.set_aspect(0.2)

    cp.plot(positions[..., 0], positions[..., 1], 'b')
    cp.plot(control[..., 0], control[..., 1], color='red', marker='s',
            linestyle='--')
    #ax.set_ylim(top=1)
    cp.axis('off')
    #cp.set_aspect(1.2)
    cp.spines["top"].set_visible(False)
    cp.spines["right"].set_visible(False)
    #for i in range(len(control)):
    #    cp.annotate("$P_{%i}$" % i, xy=(control[i, 0], control[i, 1]), xytext=(5, 0),
    #                textcoords="offset points", fontsize=15)

    kn.plot(positions[..., 0], positions[..., 1], 'b')
    kn = p_knots(knot, control[:, :3], ax=kn, point=True, line=False, dim=2, color="red")

    kn.axis('off')
    #kn.set_aspect(1.2)
    kn.spines["top"].set_visible(False)
    kn.spines["right"].set_visible(False)
    cp.set_xlim(-0.2, 5.2)
    cp.set_ylim(-0.2, 2.2)
    kn.set_xlim(-0.2, 5.2)
    kn.set_ylim(-0.2, 2.2)
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    fig.show()
    return fig

def test_knot_refinement():
    points_ori = np.array([[0, 0, 0 ],  # , 0],
                               [2, 0.2, 0],  # 0],
                               [0, 1, 0],  # , 0],
                               [2, 1, 0],
                               [4, 2, 0],
                               [5, 1, 0],
                               [3, 0, 0]])  # , 0]])
    knot = np.array([0, 0, 0, 0.4, 0.6, 0.8, 0.8, 1, 1, 1])
    resolution = 500
    reso = np.linspace(0, 1, resolution)

    nr = gn.NURB(points_ori, [knot])
    points = nr.cpoints
    B=nr.B
    weight=nr.weight

    x, y, z = gn.engine.NURB_construction([knot],points, resolution, weight)
    basis_fun, derivatives = gn.NURB_engine.basis_function_array_nurbs(knot, 2, resolution, weight)
    fig = make_plot(B, [knot], basis_fun, reso, np.asarray([x, y]).T)

    knot_ins1 = np.asarray([0.5])
    points1, knot1 = gn.nurb.knot_insertion(B, [2], [knot], knot_ins1)

    basis_fun1, derivatives1 = gn.NURB_engine.basis_function_array_nurbs(knot1[0], 2, resolution, weight)
    x1, y1, z = gn.engine.NURB_construction(knot1, points1, resolution, weight)
    fig1 = make_plot(points1, knot1, basis_fun1, reso, np.asarray([x1, y1]).T)

    nr2 = gn.NURB(points1, knot1)
    points2 = nr2.cpoints
    B2 = nr2.B
    weight = nr2.weight

    knot_ins2 = np.asarray([0.3])
    points2, knot2 = gn.nurb.knot_insertion(B2, [2], knot1, knot_ins2)

    basis_fun2, derivatives2 = gn.NURB_engine.basis_function_array_nurbs(knot2[0], 2, resolution, weight)
    x2, y2, z = gn.engine.NURB_construction(knot2, points2, resolution, weight)
    fig2 = make_plot(points2,knot2, basis_fun2, reso, np.asarray([x2, y2]).T)

    nr3 = gn.NURB(points2, knot2)
    points3 = nr3.cpoints
    B3 = nr3.B
    weight = nr3.weight

    knot_ins3 = np.asarray([0.9])
    points3, knot3 = gn.nurb.knot_insertion(B3, [2], knot2, knot_ins3)

    basis_fun3, derivatives3 = gn.NURB_engine.basis_function_array_nurbs(knot3[0], 2, resolution, weight)
    x3, y3, z = gn.engine.NURB_construction(knot3, points3, resolution, weight)
    fig3 = make_plot(points3, knot3, basis_fun3, reso, np.asarray([x3, y3]).T)

    knot_ins4 = np.arange(0.05,1,0.1)
    points4, knot4 = gn.nurb.knot_insertion(B, [2], [knot], knot_ins4)

    basis_fun4, derivatives4 = gn.NURB_engine.basis_function_array_nurbs(knot4[0], 2, resolution, weight=None)
    x4, y4, z = gn.engine.NURB_construction(knot4, points4, resolution, weight)
    fig4 = make_plot(points4, knot4, basis_fun4, reso, np.asarray([x4, y4]).T)

def test_degree_refinement():
    points_ori = np.array([[0, 0, 0],  # , 0],
                           [2, 0.2, 0],  # 0],
                           [0, 1, 0],  # , 0],
                           [2, 1, 0],
                           [4, 2, 0],
                           [5, 1, 0],
                           [3, 0, 0]])  # , 0]])
    knot = np.array([0, 0, 0, 0.4, 0.6, 0.8, 0.8, 1, 1, 1])
    resolution=500
    reso = np.linspace(0, 1, resolution)
    from pygeoiga.nurb.refinement import degree_elevation
    nr = gn.NURB(points_ori, [knot])
    B, knots = degree_elevation(nr.B, nr.knots, times=1)
    basis_fun, derivatives = gn.NURB_engine.basis_function_array_nurbs(knots[0], 3, resolution, weight=None)
    x, y, z = gn.engine.NURB_construction(knots, B, resolution, weight=None)
    fig3 = make_plot(B[...,:-1], knots, basis_fun, reso, np.asarray([x, y]).T)

    B, knots = degree_elevation(B, knots, times=1)
    basis_fun, derivatives = gn.NURB_engine.basis_function_array_nurbs(knots[0], 4, resolution, weight=None)
    x, y, z = gn.engine.NURB_construction(knots, B, resolution, weight=None)
    fig4 = make_plot(B[..., :-1], knots, basis_fun, reso, np.asarray([x, y]).T)

    B, knots = degree_elevation(B, knots, times=1)
    basis_fun, derivatives = gn.NURB_engine.basis_function_array_nurbs(knots[0], 5, resolution, weight=None)
    x, y, z = gn.engine.NURB_construction(knots, B, resolution, weight=None)
    fig5 = make_plot(B[..., :-1], knots, basis_fun, reso, np.asarray([x, y]).T)
    print(knots)
    #array([0. , 0. , 0. , 0. , 0. , 0. , 0.4, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6,
    # 0.6, 0.8, 0.8, 0.8, 0.8, 0.8, 1. , 1. , 1. , 1. , 1. , 1. ]),
    save = True
    if save:
        fig3.savefig("3_degree.png", transparent=True)
        fig4.savefig("4_degree.png", transparent=True)
        fig5.savefig( "5_degree.png", transparent=True)

def test_plot_biquadratic():
    from pygeoiga.nurb.cad import make_surface_biquadratic
    from pygeoiga.plot.nrbplotting_mpl import p_knots, create_figure, p_cpoints
    from pygeoiga.nurb.refinement import knot_insertion
    knots, cp = make_surface_biquadratic()
    shape = np.asarray(cp.shape)
    shape[-1] = cp.shape[-1] + 1
    B = np.ones((shape))
    B[..., :cp.shape[-1]] = cp

    fig, ax = create_figure("2d")
    ax = p_knots(knots, B, ax=ax, dim=2, point=False, line=True, color="b")
    ax.set_axis_off()
    plt.show()

    direction = 0
    knot_ins = np.asarray([0.2, 0.8])
    B_new, knots_new = knot_insertion(B.copy(), (2, 2), knots.copy(), knot_ins, direction=direction)
    knot_ins = np.asarray([0.3, 0.7])
    direction = 1
    B_new2, knots_new2 = knot_insertion(B_new, (2, 2), knots_new, knot_ins, direction=direction)

    fig, ax = create_figure("2d")
    ax = p_knots(knots_new2, B_new2, ax=ax, dim=2, point=False, line=True, color="b")
    ax = p_cpoints(B_new2, ax=ax, dim=2, point=True, line=False, color="red")
    ax.set_axis_off()
    plt.show()

    direction = 0
    knot_ins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    B_new, knots_new = knot_insertion(B.copy(), (2, 2), knots.copy(), knot_ins, direction=direction)
    direction = 1
    knot_ins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    B_new2, knots_new2 = knot_insertion(B_new, (2, 2), knots_new, knot_ins, direction=direction)

    fig, ax = create_figure("2d")
    ax = p_knots(knots_new2, B_new2, ax=ax, dim=2, point=False, line=True, color="b")
    #ax = p_cpoints(geometry[patch_id].get("B"), ax=ax, dim=2, color="black", marker=".", point=True, line=False)

    ax.set_axis_off()
    plt.show()

def test_plot_solution_3_layer_mp():
    from pygeoiga.nurb.cad import make_3_layer_patches
    from pygeoiga.analysis.MultiPatch import patch_topology, form_k_IGA_mp, boundary_condition_mp
    from pygeoiga.analysis.common import solve
    from pygeoiga.plot.solution_mpl import p_temperature, p_temperature_mp
    from pygeoiga.analysis.MultiPatch import map_MP_elements
    geometry = make_3_layer_patches(refine=True)
    geometry, gDoF = patch_topology(geometry)
    K_glob = np.zeros((gDoF, gDoF))
    K_glob = form_k_IGA_mp(geometry, K_glob)

    D = np.zeros(gDoF)
    b = np.zeros(gDoF)

    T_t = 10  # [°C]
    T_b = 40  # [°C]
    T_l = None#10
    T_r = None#40
    bc, D = boundary_condition_mp(geometry, D, T_t, T_b, T_l, T_r)
    bc["gDOF"] = gDoF
    D, b = solve(bc, K_glob, b, D)

    geometry = map_MP_elements(geometry, D)
    from pygeoiga.plot.nrbplotting_mpl import p_surface, p_cpoints, p_knots, create_figure
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
    p_temperature_mp(geometry=geometry, vmin=np.min(D), vmax=np.max(D), levels=50, show=False, colorbar=True,
                     ax=ax_all, point=False, fill=True, contour=False)
    plt.show()

def test_plot_solution_inverse_fault():
    from pygeoiga.nurb.cad import make_fault_model
    from pygeoiga.analysis.MultiPatch import patch_topology, form_k_IGA_mp, boundary_condition_mp
    from pygeoiga.analysis.common import solve
    from pygeoiga.plot.solution_mpl import p_temperature, p_temperature_mp
    from pygeoiga.analysis.MultiPatch import map_MP_elements
    geometry = make_fault_model(refine=True)
    geometry, gDoF = patch_topology(geometry)
    K_glob = np.zeros((gDoF, gDoF))
    K_glob = form_k_IGA_mp(geometry, K_glob)

    D = np.zeros(gDoF)
    b = np.zeros(gDoF)

    T_t = 10  # [°C]
    T_b = 40  # [°C]
    T_l = None#10
    T_r = None#40
    bc, D = boundary_condition_mp(geometry, D, T_t, T_b, T_l, T_r)
    bc["gDOF"] = gDoF
    D, b = solve(bc, K_glob, b, D)

    geometry = map_MP_elements(geometry, D)
    from pygeoiga.plot.nrbplotting_mpl import p_surface, p_cpoints, p_knots, create_figure
    fig, ax = create_figure("2d")
    #ax.view_init(azim=270, elev=90)
    fig_sol, ax_sol = create_figure("2d")
    geometrypatch = make_fault_model(refine=False)
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
    p_temperature_mp(geometry=geometry, vmin=np.min(D), vmax=np.max(D), levels=50, show=False, colorbar=True,
                     ax=ax_all, point=False, fill=True, contour=False)
    plt.show()

def test_plot_solution_salt_dome():
    from pygeoiga.nurb.cad import make_salt_dome
    from pygeoiga.analysis.MultiPatch import patch_topology, form_k_IGA_mp, boundary_condition_mp
    from pygeoiga.analysis.common import solve
    from pygeoiga.plot.solution_mpl import p_temperature, p_temperature_mp
    from pygeoiga.analysis.MultiPatch import map_MP_elements
    geometry = make_salt_dome(refine=True, knot_ins= [np.arange(0.2,1,0.2), np.arange(0.2,1,0.2)])
    geometry, gDoF = patch_topology(geometry)
    K_glob = np.zeros((gDoF, gDoF))
    K_glob = form_k_IGA_mp(geometry, K_glob)

    D = np.zeros(gDoF)
    b = np.zeros(gDoF)

    T_t = 10  # [°C]
    T_b = 90  # [°C]
    T_l = None#10
    T_r = None#40
    bc, D = boundary_condition_mp(geometry, D, T_t, T_b, T_l, T_r)
    bc["gDOF"] = gDoF
    D, b = solve(bc, K_glob, b, D)

    geometry = map_MP_elements(geometry, D)
    from pygeoiga.plot.nrbplotting_mpl import p_surface, p_cpoints, p_knots, create_figure
    fig, ax = create_figure("2d")
    #ax.view_init(azim=270, elev=90)
    fig_sol, ax_sol = create_figure("2d")
    geometrypatch = make_salt_dome(refine=False)
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
    p_temperature_mp(geometry=geometry, vmin=np.min(D), vmax=np.max(D), levels=50, show=False, colorbar=True,
                     ax=ax_all, point=False, fill=True, contour=False)
    plt.show()
