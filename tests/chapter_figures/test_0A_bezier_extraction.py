import pygeoiga as gn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os
datapath = os.path.abspath(gn.myPath+"/../tests/chapter_figures/data/") + os.sep
fig_folder=gn.myPath+'/../../manuscript_IGA_MasterThesis/Thesis/figures/A_bezier_extraction/'
kwargs_savefig=dict(transparent=True, box_inches='tight', pad_inches=0)
save_all=False

def test_plot_bernstein_polynomials_():
    resolution = 1000
    points = np.linspace(0,1, resolution)
    p_0 = [0]
    p_1 = [1]
    knot_vector = lambda p: (p_0*(p+1)) + (p_1*(p+1))

    fig, ax = plt.subplots(2, 2, sharey=True, sharex=True)

    p = 1
    N, _ = gn.NURB_engine.basis_function_array_spline(knot_vector(p),
                                                      p,
                                                      resolution)
    ax[0,0].plot(points, N)

    p = 2
    N, _ = gn.NURB_engine.basis_function_array_spline(knot_vector(p),
                                                      p,
                                                      resolution)
    ax[0, 1].plot(points, N)
    p = 3
    N, _ = gn.NURB_engine.basis_function_array_spline(knot_vector(p),
                                                      p,
                                                      resolution)
    ax[1, 0].plot(points, N)
    p = 4
    N, _ = gn.NURB_engine.basis_function_array_spline(knot_vector(p),
                                                      p,
                                                      resolution)
    ax[1, 1].plot(points, N)

    count=1
    for axess in ax:
        for axe in axess:
            axe.spines["top"].set_visible(False)
            axe.spines["right"].set_visible(False)
            axe.set_ylim((0, 1))
            axe.set_xticks([0, 0.5, 1])
            axe.set_xticklabels([-1, 0, 1])
            axe.set_title("$p$ = %s" % count)
            count+=1

    fig.show()

    save = False
    if save or save_all:
        fig.savefig(fig_folder + "bernstein_polynomials.pdf", **kwargs_savefig)


def make_plot(control, basis_fun, resolution, positions, figsize=(5,5)):
    resol = np.linspace(0,1,resolution)
    fig, (ax, bas) = plt.subplots(2, 1, figsize=figsize)
    bas.plot(resol, basis_fun)
    bas.spines["top"].set_visible(False)
    bas.spines["right"].set_visible(False)
    bas.set_xticks([0, 0.25, 0.5, 0.75, 1])
    bas.set_xlim(0,1)
    bas.set_ylim(0,1)
    # bas.set_xticklabels([])
    bas.set_yticks([1])
    bas.set_aspect(0.2)

    ax.plot(positions[..., 0], positions[..., 1], 'b')
    ax.plot(control[..., 0], control[..., 1], color='red', marker='s',
            linestyle='--')
    ax.set_xlim(-0.2, positions[..., 0].max()+0.2)
    ax.set_ylim(-0.2, 2.5)
    #ax.set_ylim(top=1)
    ax.axis('off')
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i in range(len(control)):
        ax.annotate("$P_{%i}$" % i, xy=(control[i, 0], control[i, 1]), xytext=(5, 0),
                    textcoords="offset points", fontsize=12)
    #fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    return fig

def test_knot_insertion():
    cp = np.array([[0, 0],
                   [1.2, 0.6],
                   [0.2, 1.5],
                   [3, 2.2],
                   [1.8, 1],
                   [3, 0.5],
                   [4, 1.5]
                   ])
    knot = np.array([0, 0, 0, 0, 0.25, 0.5, 0.75,  1, 1, 1, 1])  # D1
    from pygeoiga.nurb.nurb_creation import NURB

    nr = NURB(cp, [knot], resolution=500)
    N, _ = nr.get_basis_function()
    nr.create_model()
    fig = make_plot(nr.B, N, resolution=nr.resolution, positions = np.asarray([nr.model[0], nr.model[1]]).T)
    fig.show()
    nr.knot_insert([0.25,0.25])
    N, _ = nr.get_basis_function()
    nr.create_model()
    fig1 = make_plot(nr.B, N, resolution=nr.resolution, positions=np.asarray([nr.model[0], nr.model[1]]).T)
    fig1.show()

    nr.knot_insert([0.5, 0.5])
    N, _ = nr.get_basis_function()
    nr.create_model()
    fig2 = make_plot(nr.B, N, resolution=nr.resolution, positions=np.asarray([nr.model[0], nr.model[1]]).T)
    fig2.show()

    nr.knot_insert([0.75, 0.75])
    N, _ = nr.get_basis_function()
    nr.create_model()
    fig3 = make_plot(nr.B, N, resolution=nr.resolution, positions=np.asarray([nr.model[0], nr.model[1]]).T)
    fig3.show()

    save = False
    if save or save_all:
        fig.savefig(fig_folder + "knot_original.pdf", **kwargs_savefig)
        fig1.savefig(fig_folder + "knot_insert_25.pdf", **kwargs_savefig)
        fig2.savefig(fig_folder + "knot_insert_5.pdf", **kwargs_savefig)
        fig3.savefig(fig_folder + "knot_insert_75.pdf", **kwargs_savefig)


def test_image_example():
    def plot(B, knots, file_name):
        from pygeoiga.plot.nrbplotting_mpl import create_figure, p_cpoints, p_knots, p_curve, p_surface
        fig, ax = plt.subplots(constrained_layout=True)

        ax = p_knots(knots,B,  ax=ax, dim=2, point=False, line=True, color ="k")
        ax = p_cpoints(B,  ax=ax, dim=2, point=True, line=True, color ="red")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_aspect(0.8)
        fig.tight_layout(pad=0, h_pad=0, w_pad=0)
        fig.show()

        from pygeoiga.engine.NURB_engine import basis_function_array_nurbs
        fig2 = plt.figure(constrained_layout=True)
        gs = fig2.add_gridspec(2, 2, hspace=0, wspace=0,
                               width_ratios=[0.2, 1],
                               height_ratios=[1, 0.2])
        (ax_v, ax2), (no, ax_u) = gs.subplots(sharex=True, sharey=True)
        no.remove()
        N_spline_u, _ = basis_function_array_nurbs(knot_vector=knots[0], degree=2, resolution=100)
        N_spline_v, _ = basis_function_array_nurbs(knot_vector=knots[1], degree=2, resolution=100)
        resol = np.linspace(0, 1, 100)

        ax_u.plot(resol, N_spline_u)
        ax_u.spines["top"].set_visible(False)
        ax_u.spines["right"].set_visible(False)

        ax_u.set_xlim(0, 1)
        ax_u.set_ylim(0, 1)

        ax_v.plot(N_spline_v, resol)
        ax_v.spines["top"].set_visible(False)
        ax_v.spines["right"].set_visible(False)
        ax_v.set_yticks(knots[2:-2])
        ax_v.set_xlim(1, 0)
        ax_v.set_ylim(0, 1)

        for i in knots[0]:
            ax2.vlines(i, 0, 1, 'k')
        for j in knots[1]:
            ax2.hlines(j, 0, 1, 'k')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_axis_off()

        ax_u.set_xlabel("$u$")
        ax_v.set_ylabel("$v$")
        ax_v.set_yticks(knots[1][2:-2])
        ax_u.set_xticks(knots[0][2:-2])
        for ax in ax_u, ax_v, ax2, no:
            ax.label_outer()
        fig2.show()

        save = False
        if save or save_all:
            fig2.savefig(fig_folder + file_name, **kwargs_savefig)

    from pygeoiga.nurb.cad import make_surface_biquadratic
    knots, B = make_surface_biquadratic()

    from pygeoiga.nurb.refinement import knot_insertion
    knot_ins_1 = [0.3, 0.6]
    knot_ins_0 = [0.25, 0.75]
    B, knots = knot_insertion(B, (2, 2), knots, knot_ins_0, direction=0)
    B, knots = knot_insertion(B, (2, 2), knots, knot_ins_1, direction=1)
    plot(B, knots, "mapping.pdf")

    knots, B = make_surface_biquadratic()

    from pygeoiga.nurb.refinement import knot_insertion
    knot_ins_1 = [0.3,0.3, 0.6,0.6]
    knot_ins_0 = [0.25,0.25, 0.5, 0.75,0.75]
    B, knots = knot_insertion(B, (2, 2), knots, knot_ins_0, direction=0)
    B, knots = knot_insertion(B, (2, 2), knots, knot_ins_1, direction=1)
    plot(B, knots, "mapping.pdf")


def test_image_example2():

    from pygeoiga.nurb.cad import make_surface_biquadratic
    knots, B = make_surface_biquadratic()

    from pygeoiga.nurb.refinement import knot_insertion
    knot_ins_1 = [0.3, 0.6]
    knot_ins_0 = [0.25, 0.75]
    B, knots = knot_insertion(B, (2, 2), knots, knot_ins_0, direction=0)
    B, knots = knot_insertion(B, (2, 2), knots, knot_ins_1, direction=1)
    before = B
    from pygeoiga.plot.nrbplotting_mpl import create_figure, p_cpoints, p_knots, p_curve, p_surface
    from pygeoiga.engine.NURB_engine import basis_function_array_nurbs
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 3, hspace=0, wspace=0,
                           width_ratios=[0.2, 0.2, 1],
                           height_ratios=[1, 0.2, 0.2])
    (ax_bv, ax_v, ax2), (no1, no2, ax_u), (no3, no4, ax_bu) = gs.subplots(sharex=True, sharey=True)
    for no in no1, no2, no3, no4:
        no.remove()

    N_spline_u, _ = basis_function_array_nurbs(knot_vector=knots[0], degree=2, resolution=100)
    N_spline_v, _ = basis_function_array_nurbs(knot_vector=knots[1], degree=2, resolution=100)
    resol = np.linspace(0, 1, 100)

    ax_u.plot(resol, N_spline_u)
    ax_u.spines["top"].set_visible(False)
    ax_u.spines["right"].set_visible(False)

    ax_u.set_xlim(0, 1)
    ax_u.set_ylim(0, 1)

    ax_v.plot(N_spline_v, resol)
    ax_v.spines["top"].set_visible(False)
    ax_v.spines["right"].set_visible(False)
    ax_v.set_yticks(knots[2:-2])
    ax_v.set_xlim(0, 1)
    ax_v.set_ylim(0, 1)

    ax_u.set_xlabel("$u$")
    ax_v.set_ylabel("$v$")

    ax_v.set_xlabel("$N(v)$")
    ax_u.set_ylabel("$N(u)$")

    ax_v.set_yticks(knots[1][2:-2])
    ax_u.set_xticks(knots[0][2:-2])

    for i in knots[0]:
        ax2.vlines(i, 0, 1, 'k')
    for j in knots[1]:
        ax2.hlines(j, 0, 1, 'k')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_axis_off()





    fig2, ax = plt.subplots(constrained_layout=True)
    ax = p_cpoints(B, ax=ax, dim=2, point=False, line=True, color="black", linestyle="-")
    ax = p_cpoints(B, ax=ax, dim=2, point=True, line=False, color="red")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect(0.8)
    fig2.tight_layout(pad=0, h_pad=0, w_pad=0)
    fig2.show()

    from pygeoiga.nurb.refinement import knot_insertion
    knot_ins_1 = [0.3, 0.6]
    knot_ins_0 = [0.25, 0.5, 0.75]
    B, knots = knot_insertion(B, (2, 2), knots, knot_ins_0, direction=0)
    B, knots = knot_insertion(B, (2, 2), knots, knot_ins_1, direction=1)
    new = B
    N_spline_u, _ = basis_function_array_nurbs(knot_vector=knots[0], degree=2, resolution=100)
    N_spline_v, _ = basis_function_array_nurbs(knot_vector=knots[1], degree=2, resolution=100)
    resol = np.linspace(0, 1, 100)

    ax_bu.plot(resol, N_spline_u)
    ax_bu.spines["top"].set_visible(False)
    ax_bu.spines["right"].set_visible(False)

    ax_bu.set_xlim(0, 1)
    ax_bu.set_ylim(0, 1)

    ax_bv.plot(N_spline_v, resol)
    ax_bv.spines["top"].set_visible(False)
    ax_bv.spines["right"].set_visible(False)
    ax_bv.set_yticks(knots[2:-2])
    ax_bv.set_xlim(0, 1)
    ax_bv.set_ylim(0, 1)

    ax_bu.set_xlabel("$u$")
    ax_bv.set_ylabel("$v$")
    ax_bv.set_xlabel("$B(v)$")
    ax_bu.set_ylabel("$B(u)$")
    ax_bv.set_yticks(knots[1][2:-2])
    ax_bu.set_xticks(knots[0][2:-2])

    #for ax in ax_u, ax_v, ax2, no, ax_bu, ax_bv:
    #    ax.label_outer()
    fig2.show()

    degree_u = len(np.where(knots[0] == 0.)[0]) - 1
    degree_v = len(np.where(knots[1] == 0.)[0]) - 1

    n_xi = len(knots[0]) - degree_u - 3
    n_eta = len(knots[1]) - degree_v - 3

    from pygeoiga.analysis.common import IEN_element_topology, transform_matrix_B
    IEN = IEN_element_topology(n_xi, n_eta, degree_u)
    P, W = transform_matrix_B(B)

    fig3, ax = plt.subplots(constrained_layout=True)
    for i in range(0, n_xi, 2):
        pos = IEN[i::n_xi]
        for e in pos[::2]:
            # cont = np.hstack([IEN[0][:degree_u+1], IEN[0][degree_u+3], np.flip(IEN[0][-degree_u-1:]), IEN[0][degree_u+1], IEN[0][0]])
            cont = np.hstack([e[:degree_u + 1], e[degree_u + 3], np.flip(e[-degree_u - 1:]), e[degree_u + 1], e[0]])
            ax.plot(P[cont][:, 0], P[cont][:, 1], linestyle="-", color="black", marker=None)
    ax = p_cpoints(B, ax=ax, dim=2, point=True, line=False, color="red")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect(0.8)
    fig3.tight_layout(pad=0, h_pad=0, w_pad=0)
    fig.show()

    fig2.show()
    fig3.show()

    fig_0, ax = plt.subplots(constrained_layout=True)
    ax = p_knots(knots, B, ax=ax, dim=2, point=False, line=True, color="k")

    ax = p_surface(knots, B, ax=ax, dim=2, fill=True, border=False, color="gray", alpha=0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect(0.8)
    fig_0.tight_layout(pad=0, h_pad=0, w_pad=0)
    fig_0.show()

    save = False
    if save or save_all:
        fig.savefig(fig_folder + "bezier_extraction.pdf", **kwargs_savefig)
        fig_0.savefig(fig_folder + "curve_original.pdf", **kwargs_savefig)
        fig2.savefig(fig_folder + "control_original.pdf", **kwargs_savefig)
        fig3.savefig(fig_folder + "control_bezier.pdf", **kwargs_savefig)

def test_listings():
    from pygeoiga.nurb.cad import make_3_layer_patches

    geometry = make_3_layer_patches(refine=True)

    from pygeoiga.analysis.MultiPatch import patch_topology

    geometry, gDoF = patch_topology(geometry)

    def bezier_extraction(geometry):
        from pygeoiga.analysis.bezier_extraction import bezier_extraction_operator_bivariate, bezier_extraction_operator

        for patch_id in geometry.keys():
            U, V = geometry[patch_id].get("knots")
            degree = geometry[patch_id].get("degree")
            n_u, n_v = geometry[patch_id].get("n_element")
            assert degree[0] == degree[1], "Degree of the geometry is not the same"
            degree = degree[0]

            C_u = bezier_extraction_operator(U, degree)
            C_v = bezier_extraction_operator(V, degree)
            C = bezier_extraction_operator_bivariate(C_u, C_v, n_u, n_v, degree)

            geometry[patch_id]["bezier_extraction"] = C

        return geometry

    geometry = bezier_extraction(geometry)

    from pygeoiga.analysis.bezier_FE import form_k

    def assemble_stiffness_matrix_bezier(geometry: dict, gDoF: int):
        # Set empty the stiffness matrix according to the global degrees of freedom
        K_glob = np.zeros((gDoF, gDoF))
        for patch_id in geometry.keys():
            pDof = geometry[patch_id].get("patch_DOF")  # Degrees of freedom per patch
            nx, ny = geometry[patch_id].get("n_element")  # number of elements in parametric space
            nel = nx * ny  # total number of elements
            K = np.zeros((pDof, pDof))  # Initialize empty patch stiffness matrix

            P = geometry[patch_id].get("list_cp")  # Get list with location of control points
            W = geometry[patch_id].get("list_weight")  # Get list of weights
            # Support only the same degree for both parametric directions
            degree = geometry[patch_id].get("degree")[0]

            # Bezier exatractor operator of the patch
            C = geometry[patch_id].get("bezier_extraction")

            kappa = geometry[patch_id].get("kappa")  # Get thermal conductivity of patch

            IEN = geometry[patch_id].get("IEN")  # Global connectivity array (Element topology)

            K = form_k(K, IEN, P, kappa, nel, degree, W, C)
            geometry[patch_id]["K"] = K
            patch_glob_num = geometry[patch_id].get("glob_num")
            K_glob[np.ix_(patch_glob_num,
                          patch_glob_num)] = K_glob[np.ix_(patch_glob_num,
                                                           patch_glob_num)] + K

        return K_glob

    K_glob = assemble_stiffness_matrix_bezier(geometry, gDoF)
    plt.spy(K_glob)
    plt.show()




def do_IGA(function, T_t, T_b, knot_ins):
    geometry =  function(refine=True, knot_ins=knot_ins)
    from pygeoiga.analysis.MultiPatch import patch_topology, form_k_IGA_mp
    geometry, gDoF = patch_topology(geometry)
    K_glob_IGA = np.zeros((gDoF, gDoF))
    F_IGA = np.zeros(gDoF)
    a_IGA = np.zeros(gDoF)
    K_glob_IGA = form_k_IGA_mp(geometry, K_glob_IGA)
    from pygeoiga.analysis.MultiPatch import boundary_condition_mp
    bc_IGA, a_IGA = boundary_condition_mp(geometry, a_IGA, T_t, T_b, None, None)
    bc_IGA["gDOF"] = gDoF
    from pygeoiga.analysis.common import solve
    a_IGA, F_IGA = solve(bc_IGA, K_glob_IGA, F_IGA, a_IGA)
    from pygeoiga.analysis.MultiPatch import map_MP_elements
    geometry = map_MP_elements(geometry, a_IGA)
    return geometry, gDoF

def do_Bezier(function, T_t, T_b, knot_ins):
    bezier_geometry = function(refine=True, knot_ins=knot_ins)
    from pygeoiga.analysis.MultiPatch import patch_topology, bezier_extraction_mp, form_k_IGA_mp, form_k_bezier_mp
    bezier_geometry, gDoF = patch_topology(bezier_geometry)
    bezier_geometry = bezier_extraction_mp(bezier_geometry)
    K_glob_be = np.zeros((gDoF, gDoF))
    F_be = np.zeros(gDoF)
    a_be = np.zeros(gDoF)
    K_glob_be = form_k_bezier_mp(bezier_geometry, K_glob_be)
    from pygeoiga.analysis.MultiPatch import boundary_condition_mp
    bc_be, a_be = boundary_condition_mp(bezier_geometry, a_be, T_t, T_b, None, None)
    bc_be["gDOF"] = gDoF
    from pygeoiga.analysis.common import solve
    a_be, F_be = solve(bc_be, K_glob_be, F_be, a_be)
    from pygeoiga.analysis.MultiPatch import map_MP_elements
    bezier_geometry = map_MP_elements(bezier_geometry, a_be)
    return bezier_geometry, gDoF

def do_FEM(function, T_t, T_b, knot_ins , size):
    from pygeoiga.nurb.nrb_to_gmsh import convert_geometry_mp_to_gmsh
    geometry = function(refine=True, knot_ins=knot_ins)
    name = "temporal"
    mesh, script, physical_tag_id = convert_geometry_mp_to_gmsh(geometry,
                                                                size=size,
                                                                save_geo=datapath + name + ".geo",
                                                                save_msh=datapath + name + ".msh")
    from pygeoiga.FE_solvers.run_fenics import convert_msh_to_xdmf, run_simulation
    input = datapath + name + ".msh"
    convert_msh_to_xdmf(input)
    nodal_coordinates, temperature_nodes, mesh, u = run_simulation(input,
                                                                   topology_info=physical_tag_id,
                                                                   top_bc=T_t,
                                                                   bot_bc=T_b,
                                                                   geometry=geometry,
                                                                   show=False)
    return nodal_coordinates, temperature_nodes, u


def comparison_all_meshes(function_callable, T_t, T_b, filepath, size=100,
                          knot_ins=[np.arange(0.1,1,0.1), np.arange(0.1,1,0.1)],
                          bezier=True,
                          save=False,
                          name="temp.none",
                          label="Difference (Solution - "):
    import matplotlib
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    geometry_IGA, dof_IGA = do_IGA(function_callable, T_t, T_b, knot_ins)

    geometry_BE, dof_BE = do_Bezier(function_callable, T_t, T_b, knot_ins)
    coor, temp, u_FEM = do_FEM(function_callable, T_t, T_b, knot_ins, size)
    u_FEM.set_allow_extrapolation(True)
    dof_FEM = temp.shape[0]

    from pygeoiga.FE_solvers.run_fenics import read_fenics_solution
    u, mesh, dofs = read_fenics_solution(filepath)
    u.set_allow_extrapolation(True)  # TODO: Not understand when this is needed

    if bezier:
        fig, [ax_IGA, ax_BE, ax_FEM] = plt.subplots(1,3, sharey=True, figsize=(17,5))
    else:
        fig, [ax_IGA, ax_FEM] = plt.subplots(1, 2, sharey=True, figsize=(11, 5))
    #cma = plt.cm.seismic
    #cma = plt.cm.Reds
    cma = plt.cm.RdBu
    def max_min(IGA, BE, FEM):
        diff_all = np.array([])
        for geometry in [IGA, BE]:
            for patch_id in geometry.keys():
                x = geometry[patch_id].get("x_sol")
                y = geometry[patch_id].get("y_sol")
                m, n = x.shape
                correct = np.zeros((m, n))
                for x_i in range(n):
                    for y_i in range(m):
                        correct[y_i, x_i] = u(x[y_i, x_i], y[y_i, x_i], 0)

                err = correct - geometry[patch_id].get("t_sol")
                #err = (correct - geometry[patch_id].get("t_sol"))**2
                diff_all = np.r_[diff_all, err.ravel()]

        t_fun = np.vectorize(u)
        val = t_fun(coor[:, 0], coor[:, 1], np.zeros(dof_FEM))
        erro = val - temp
        diff_all = np.r_[diff_all, erro.ravel()]

        vmin = np.min(diff_all)
        vmax = np.max(diff_all)
        #if vmin == 0:
        #    vmin = -1e-3
        #if vmax==0:
        #    vmax = 1e-3
        return vmin, vmax

    def geometry_difference(geometry, u, ax, vmin, vmax):
        x_all = np.array([])
        y_all = np.array([])
        diff_all = np.array([])
        count = 0
        for patch_id in geometry.keys():
            x = geometry[patch_id].get("x_sol")
            y = geometry[patch_id].get("y_sol")
            m, n = x.shape
            correct = np.zeros((m, n))
            for x_i in range(n):
                for y_i in range(m):
                    correct[y_i, x_i] = u(x[y_i, x_i], y[y_i, x_i], 0)

            err = correct - geometry[patch_id].get("t_sol")
            #err = (correct - geometry[patch_id].get("t_sol")) ** 2

            x_all = np.r_[x_all, x.ravel()]
            y_all = np.r_[y_all, y.ravel()]
            diff_all = np.r_[diff_all, err.ravel()]
            count +=1

        vmin = np.min(diff_all)
        vmax = np.max(diff_all)

        #if np.abs(vmin)<vmax:
        #    vmin=-vmax
        #else:
        #    vmax = np.abs(vmin)

        if vmin >= 0:
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            val = cma(np.linspace(0.5, 1, 256))
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list('seismic_2', val)
        elif vmax <= 0:
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            val = cma(np.linspace(0, 0.5, 256))
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list('seismic_2', val)
        else:
            norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            cmap = cma
        #cmap = cma
        mappeable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

        if count == 1:
            ax.contourf(x, y, err, vmin=vmin, vmax=vmax, cmap=cmap, levels=100)
        else:
            ax.tricontourf(x_all, y_all, diff_all, vmin=vmin, vmax=vmax, cmap=cmap, levels=100)

        divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad="2%")
        cax = divider.append_axes("bottom", size="5%", pad="10%")
        cbar = ax.figure.colorbar(mappeable, cax=cax, ax=ax, label=label+"IGA)", orientation="horizontal", format='%.0e')
        return ax
    vmin, vmax = max_min(geometry_IGA, geometry_BE, u)
    ax_IGA = geometry_difference(geometry_IGA, u, ax_IGA, vmin, vmax)
    ax_IGA.set_title("%s DoF IGA" %dof_IGA)
    if bezier:
        ax_BE = geometry_difference(geometry_BE, u, ax_BE,  vmin, vmax)
        ax_BE.set_title("%s DoF Bezier" %dof_BE)

    #FEM
    count=0
    x_all = np.array([])
    y_all = np.array([])
    diff_all = np.array([])
    for patch_id in geometry_IGA.keys():
        x = geometry_IGA[patch_id].get("x_sol")
        y = geometry_IGA[patch_id].get("y_sol")
        m, n = x.shape
        correct = np.zeros((m, n))
        correct_FEM = np.zeros((m, n))
        for x_i in range(n):
            for y_i in range(m):
                correct_FEM[y_i, x_i] = u_FEM(x[y_i, x_i], y[y_i, x_i], 0)
                correct[y_i, x_i] = u(x[y_i, x_i], y[y_i, x_i], 0)
        erro = correct - correct_FEM
        #erro = (correct-correct_FEM)**2
        diff_all = np.r_[diff_all, erro.ravel()]
        x_all = np.r_[x_all, x.ravel()]
        y_all = np.r_[y_all, y.ravel()]
        count +=1
    vmin=np.min(diff_all)
    vmax=np.max(diff_all)
    #if np.abs(vmin) < vmax:
    #    vmin = -vmax
    #else:
    #    vmax = np.abs(vmin)
    #t_fun = np.vectorize(u)
    #val = t_fun(coor[:,0], coor[:,1], np.zeros(dof_FEM))
    #erro = (val - temp)**2
    #vmin = np.min(erro)
    #vmax = np.max(erro)
    #count=2
    if vmin == 0:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        val = cma(np.linspace(0.5, 1, 256))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('seismic_2', val)

    elif vmax == 0:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        val = cma(np.linspace(0, 0.5, 256))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('seismic_2', val)
    else:
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        cmap = cma
    #cmap = cma
    # norm = matplotlib.colors.Normalize(vmin=min_p, vmax=max_p, v)
    if count == 1:
        ax_FEM.contourf(x, y, erro, vmin=vmin, vmax=vmax, cmap=cmap, levels=100)
    else:
        ax_FEM.tricontourf(x_all, y_all, diff_all, vmin=vmin, vmax=vmax, cmap=cmap, levels=100)

    #ax_FEM.tricontourf(coor[:, 0], coor[:, 1], erro, vmin=vmin, vmax=vmax, cmap=cmap, levels=100)
    ax_FEM.set_title("%s DoF FEM" % dof_FEM)

    divider = make_axes_locatable(ax_FEM)
    cax = divider.append_axes("bottom", size="5%", pad="10%")

    mappeable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = ax_FEM.figure.colorbar(mappeable, cax=cax, ax=ax_FEM, label=label+"FEM)", orientation="horizontal", format='%.0e')

    for c in ax_IGA.collections:
        c.set_edgecolor("face")

    for c in ax_FEM.collections:
        c.set_edgecolor("face")
    fig.show()

    if save:
        fig.savefig(fig_folder+name+".pdf", **kwargs_savefig)


def plot_IGA(geometry, a, gDoF, figsize=(5,5), file_name="temp.pdf", save=False, levels=None, name="_"):
    from pygeoiga.plot.nrbplotting_mpl import p_surface, p_cpoints, p_knots
    from pygeoiga.plot.solution_mpl import p_temperature, p_triangular_mesh, p_temperature_mp

    figsize=(figsize[0]*2, figsize[1])
    fig_sol, [ax2, ax3] = plt.subplots(1, 2, figsize=figsize, sharey=True)

    xmin=0
    xmax=0
    ymin=0
    ymax=0
    cbar=True
    for patch_id in geometry.keys():
        x = geometry[patch_id].get("x_sol")
        y = geometry[patch_id].get("y_sol")
        t = geometry[patch_id].get("t_sol")

        xmin = x.min() if x.min() < xmin else xmin
        xmax = x.max() if x.max() > xmax else xmax
        ymin = y.min() if y.min() < ymin else ymin
        ymax = y.max() if y.max() > ymax else ymax

        ax2 = p_temperature(x, y, t, vmin=np.min(a), vmax=np.max(a), show=False, colorbar=False,
                                 ax=ax2, point=True, fill=False,
                            markersize=25)

        ax2 = p_knots(geometry[patch_id].get("knots"),
                     geometry[patch_id].get("B"),
                     ax=ax2,
                     color='k',
                     dim=2,
                     point=False,
                     line=True,
                      linestyle="--",
                      linewidth=0.2)

        ax3 = p_temperature(x, y, t, vmin=np.min(a), vmax=np.max(a), levels=200, show=False, colorbar=cbar, ax=ax3,
                               point=False, fill=True, contour=False)
        cbar = False

    ax2.set_title("%s DoF_%s"%(gDoF,name))

    for ax in ax2, ax3:
        ax.set_aspect("equal")
        ax.set_ylabel(r"$y$")
        ax.set_xlabel(r"$x$")
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)

    for c in ax3.collections:
        c.set_edgecolor("face")

    for patch_id in geometry.keys():
        x = geometry[patch_id].get("x_sol")
        y = geometry[patch_id].get("y_sol")
        t = geometry[patch_id].get("t_sol")

        ax3 = p_temperature(x, y, t, vmin=np.min(a), vmax=np.max(a), levels=levels,
                            show=False, colorbar=False, colors=["black"], ax=ax3,
                            point=False, fill=False, contour=True, cmap=None,
                            linewidths=0.5)
    #ax3 = p_temperature_mp(geometry, vmin=np.min(a), vmax=np.max(a), levels=levels,
    #                                         show=False, colorbar=False, colors=["black"], ax=ax3,
    #                                            point=False, fill=False, contour=True, cmap=None,
    #                                            linewidths=0.5)

    plt.tight_layout()
    fig_sol.show()

    if save or save_all:
        fig_sol.savefig(fig_folder + file_name, **kwargs_savefig)

def plot_field(geometry, a, gDoF, figsize=(5, 5), file_name="temp.pdf", save=False, levels=None, name="_"):
    from pygeoiga.plot.nrbplotting_mpl import p_surface, p_cpoints, p_knots
    from pygeoiga.plot.solution_mpl import p_temperature, p_triangular_mesh, p_temperature_mp

    fig_sol, ax = plt.subplots(figsize=figsize)
    xmin = 0
    xmax = 0
    ymin = 0
    ymax = 0
    cbar = True
    for patch_id in geometry.keys():
        x = geometry[patch_id].get("x_sol")
        y = geometry[patch_id].get("y_sol")
        t = geometry[patch_id].get("t_sol")

        xmin = x.min() if x.min() < xmin else xmin
        xmax = x.max() if x.max() > xmax else xmax
        ymin = y.min() if y.min() < ymin else ymin
        ymax = y.max() if y.max() > ymax else ymax

        ax = p_temperature(x, y, t, vmin=np.min(a), vmax=np.max(a), levels=200, show=False, colorbar=cbar, ax=ax,
                            point=False, fill=True, contour=False)
        cbar = False

    ax.set_title("%s DoF %s" % (gDoF, name))

    ax.set_aspect("equal")
    ax.set_ylabel(r"$y$")
    ax.set_xlabel(r"$x$")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    for c in ax.collections:
        c.set_edgecolor("face")

    for patch_id in geometry.keys():
        x = geometry[patch_id].get("x_sol")
        y = geometry[patch_id].get("y_sol")
        t = geometry[patch_id].get("t_sol")

        ax = p_temperature(x, y, t, vmin=np.min(a), vmax=np.max(a), levels=levels,
                            show=False, colorbar=False, colors=["black"], ax=ax,
                            point=False, fill=False, contour=True, cmap=None,
                            linewidths=0.5)
    plt.tight_layout()
    fig_sol.show()

    if save or save_all:
        fig_sol.savefig(fig_folder + file_name, **kwargs_savefig)


def same_IGA_BEZIER(geometry, T_t, T_b, save=False, filename="temp", levels=None):
    from pygeoiga.analysis.MultiPatch import patch_topology, bezier_extraction_mp, form_k_IGA_mp, form_k_bezier_mp
    geometry, gDoF = patch_topology(geometry)

    import copy
    bezier_geometry = copy.deepcopy(geometry)
    bezier_geometry = bezier_extraction_mp(bezier_geometry)

    K_glob_IGA = np.zeros((gDoF, gDoF))
    F_IGA = np.zeros(gDoF)
    a_IGA = np.zeros(gDoF)
    K_glob_be = np.zeros((gDoF, gDoF))
    F_be = np.zeros(gDoF)
    a_be = np.zeros(gDoF)

    K_glob_IGA = form_k_IGA_mp(geometry, K_glob_IGA)
    K_glob_be = form_k_bezier_mp(bezier_geometry, K_glob_be)

    from pygeoiga.analysis.MultiPatch import boundary_condition_mp
    bc_IGA, a_IGA = boundary_condition_mp(geometry, a_IGA, T_t, T_b, None, None)
    bc_IGA["gDOF"] = gDoF
    bc_be, a_be = boundary_condition_mp(bezier_geometry, a_be, T_t, T_b, None, None)
    bc_be["gDOF"] = gDoF
    from pygeoiga.analysis.common import solve

    a_IGA, F_IGA = solve(bc_IGA, K_glob_IGA, F_IGA, a_IGA)
    a_be, F_be = solve(bc_be, K_glob_be, F_be, a_be)

    from pygeoiga.analysis.MultiPatch import map_MP_elements

    geometry = map_MP_elements(geometry, a_IGA)
    bezier_geometry = map_MP_elements(bezier_geometry, a_be)
    figsize = (6, 5)
    #plot_IGA(geometry, a_IGA, gDoF, name="IGA")
    plot_field(geometry, a_IGA, gDoF, file_name=filename+"_IGA.pdf", name="IGA", figsize=figsize, save=save, levels=levels)
    #plot_IGA(bezier_geometry, a_be, gDoF, name ="Bezier")
    plot_field(bezier_geometry, a_be, gDoF, file_name=filename+"_bezier.pdf", name="Bezier", figsize=figsize, save=save, levels=levels)
    fig, ax = plt.subplots(figsize=figsize)
    min_p = None
    max_p = None
    cmap = plt.get_cmap("RdBu")
    xmin = 0
    xmax = 0
    ymin = 0
    ymax = 0
    cbar = True

    for patch_id in geometry.keys():
        err = geometry[patch_id].get("t_sol") - bezier_geometry[patch_id].get("t_sol")
        x = geometry[patch_id].get("x_sol")
        y = geometry[patch_id].get("y_sol")
        xmin = x.min() if x.min() < xmin else xmin
        xmax = x.max() if x.max() > xmax else xmax
        ymin = y.min() if y.min() < ymin else ymin
        ymax = y.max() if y.max() > ymax else ymax
        if min_p is None or min_p > err.min():
            min_p = err.min()
        if max_p is None or max_p < err.max():
            max_p = err.max()

    ax.set_aspect("equal")
    ax.set_ylabel(r"$y$")
    ax.set_xlabel(r"$x$")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    for patch_id in geometry.keys():
        err = geometry[patch_id].get("t_sol") - bezier_geometry[patch_id].get("t_sol")
        x = geometry[patch_id].get("x_sol")
        y = geometry[patch_id].get("y_sol")
        ax.contourf(x, y, err, vmin=min_p, vmax=max_p, cmap=cmap)

    import matplotlib
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad="2%")
    norm = matplotlib.colors.TwoSlopeNorm(vmin=min_p, vcenter=0, vmax=max_p)
    #norm = matplotlib.colors.Normalize(vmin=min_p, vmax=max_p, v)
    mappeable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = ax.figure.colorbar(mappeable, cax=cax, ax=ax, label="Difference (IGA-Bezier)")

    fig.show()
    if save or save_all:
        fig.savefig(fig_folder + filename+"_difference.pdf", **kwargs_savefig)


def test_plot_solution_biquadratic():
    def create_geom(**kwargs):
        from pygeoiga.nurb.cad import make_surface_biquadratic
        knots, B = make_surface_biquadratic()

        from pygeoiga.nurb.refinement import knot_insertion
        to_ins = kwargs.get("knot_ins", np.arange(0.1, 1, 0.1))

        knots_ins_0 = [x for x in to_ins if x not in knots[0]]
        B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_0, direction=0)
        knots_ins_1 = [x for x in to_ins if x not in knots[1]]
        B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_1, direction=1)

        geometry = dict()
        geometry["quadrat"] = {"B": B,
                               "knots": knots,
                               "kappa": 4,
                               'color': "gray",
                               "position": (1, 1),
                               "BC": {0: "bot_bc", 2: "top_bc"}}
        return geometry

    save = True
    T_t = 10
    T_b = 25
    T_l = None
    T_r = None
    name = "biquadratic"
    lith = ["4 [W/mK]"]
    levels = [12, 14, 17, 20, 22, 24]


    comparison_all_meshes(create_geom,
                          T_t=T_t,
                          T_b=T_b,
                          filepath=datapath + "solution_biquadratic",
                          size=0.4,
                          knot_ins=np.arange(0.1, 1, 0.1),
                          bezier=True,
                          save=save,
                          name=name+"_error")

def test_compare_biquadratic():
    from pygeoiga.nurb.cad import make_surface_biquadratic
    knots, B = make_surface_biquadratic()
    from pygeoiga.nurb.refinement import knot_insertion
    knot_ins = [np.arange(0.05,1,0.05), np.arange(0.05,1,0.05)]

    knots_ins_0 = [x for x in knot_ins[0] if x not in knots[0]]
    B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_0, direction=0)
    knots_ins_1 = [x for x in knot_ins[1] if x not in knots[1]]
    B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_1, direction=1)

    geometry=dict()
    geometry["quadrat"] = {"B": B,
                            "knots": knots,
                            "kappa": 4,
                            'color': "red",
                            "position": (1,1),
                            "BC": {0: "bot_bc", 2: "top_bc"}
                           }
    T_t = 10
    T_b = 25
    same_IGA_BEZIER(geometry, T_t, T_b, save = True, filename="biquadratic", levels=[12, 14, 17, 20, 22, 24])


def test_same_anticline():
    from pygeoiga.nurb.cad import make_3_layer_patches
    geometry = make_3_layer_patches(refine=True)
    same_IGA_BEZIER(geometry, 10, 25, save = True, filename="anticline", levels=[11, 12, 14, 17, 20, 22,23, 24])

def test_same_fault():
    from pygeoiga.nurb.cad import make_fault_model
    geometry = make_fault_model(refine=True)
    same_IGA_BEZIER(geometry, 10, 40, save = True, filename="fault", levels = [12, 16, 20, 24, 28, 32, 36])

def test_same_dome():
    from pygeoiga.nurb.cad import make_salt_dome
    #geometry = make_salt_dome(refine=False)
    #knot_ins = [np.arange(0.1, 1, 0.1), np.arange(0.1, 1, 0.1)]
    #for patch_id in geometry.keys():
    #    knots = geometry[patch_id].get("knots")
    #    knot_ins[0] = [x for x in knot_ins[0] if x not in knots[0]]
    #    knot_ins[1] = [x for x in knot_ins[1] if x not in knots[1]]
    knot_ins = [[0.1,0.2,0.3,0.6,0.7,],[0.1,0.2,0.3,0.6,0.7]]
    geometry = make_salt_dome(refine=True, knot_ins=knot_ins)
    same_IGA_BEZIER(geometry, 10, 90,  save=True, filename="fault",levels = [15, 20, 30, 40, 50, 60, 70, 80, 85])