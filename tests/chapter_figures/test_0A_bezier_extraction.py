import pygeoiga as gn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

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


