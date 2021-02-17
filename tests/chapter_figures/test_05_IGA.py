import pygeoiga as gn
import numpy as np
import matplotlib.pyplot as plt
fig_folder=gn.myPath+'/../../manuscript/Thesis/figures/05_IGA/'
kwargs_savefig=dict(transparent=True, box_inches='tight', pad_inches=0)
save_all=False


def test_show_simple_mesh():
    from pygeoiga.plot.nrbplotting_mpl import create_figure, p_cpoints, p_knots, p_curve, p_surface

    cpoints = np.array([[[1, 1], [3,0]],
                        [[1, 2], [2.5, 2.5]]]
                  )
    knots = [[0, 0, 1, 1], [0, 0, 1, 1]]


    shape = np.asarray(cpoints.shape)
    shape[-1] = 3  # To include the weights in the last term
    B = np.ones(shape)
    B[...,:2] = cpoints

    from pygeoiga.nurb.refinement import knot_insertion
    knots_ins_0 = [1/3, 2/3]
    B, knots = knot_insertion(B, degree=(1, 1), knots=knots, knots_ins=knots_ins_0, direction=0)
    knots_ins_1 = [0.5]
    B, knots = knot_insertion(B, degree=(1, 1), knots=knots, knots_ins=knots_ins_1, direction=1)

    fig, ax = create_figure("2d")
    #fig, [ax,ax2] = plt.subplots(1,2)
    #ax.set_axis_off()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(0.5)
    ax = p_knots(knots, B, ax=ax, dim=2, point=False, line=True, color="black")
    ax = p_cpoints(B, ax=ax, dim=2, color="red", marker="o", point=True, line=False)
    n, m = B.shape[0], B.shape[1]
    P = np.asarray([(B[x, y, 0], B[x, y, 1]) for x in range(n) for y in range(m)])

    for count, point in enumerate(P):
        ax.annotate(str(count), point, xytext=(5, 5), textcoords="offset points")

    ax.annotate("1", (1.5, 0.9), fontsize=20)
    ax.annotate("2", (2.5, 0.6), fontsize=20)
    ax.annotate("3", (1.5, 1.4), fontsize=20)
    ax.annotate("4", (2.3, 1.3), fontsize=20)
    ax.annotate("5", (1.5, 1.8), fontsize=20)
    ax.annotate("6", (2.3, 1.8), fontsize=20)

    fig.show()
    from pygeoiga.nurb.refinement import degree_elevation
    B, knots = degree_elevation(B, knots, direction=0)
    B, knots = degree_elevation(B, knots, direction=1)

    fig2, ax2 = create_figure("2d")
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$y$")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_aspect(0.5)
    ax2 = p_knots(knots, B, ax=ax2, dim=2, point=False, line=True, color="black")
    ax2 = p_cpoints(B, ax=ax2, dim=2, color="red", marker="o", point=True, line=False)
    n, m = B.shape[0], B.shape[1]
    P = np.asarray([(B[x, y, 0], B[x, y, 1]) for x in range(n) for y in range(m)])

    for count, point in enumerate(P):
        ax2.annotate(str(count), point, xytext=(5, 5), textcoords="offset points")

    ax2.annotate("1", (1.5, 0.9), fontsize=20, xytext=(20,0), textcoords="offset points")
    ax2.annotate("2", (2.5, 0.6), fontsize=20, xytext=(10,-10), textcoords="offset points")
    ax2.annotate("3", (1.5, 1.4), fontsize=20, xytext=(17,6), textcoords="offset points")
    ax2.annotate("4", (2.3, 1.3), fontsize=20, xytext=(20,10), textcoords="offset points")
    ax2.annotate("5", (1.5, 1.8), fontsize=20, xytext=(13,15), textcoords="offset points")
    ax2.annotate("6", (2.3, 1.8), fontsize=20, xytext=(13,15), textcoords="offset points")

    fig2.show()

    save = False
    if save or save_all:
        fig.savefig(fig_folder + "NURB_mesh_1_degree.pdf", **kwargs_savefig)
        fig2.savefig(fig_folder + "NURB_mesh_2_degree.pdf", **kwargs_savefig)


def test_make_NURB_biquadratic():

    def plot(B, knots, file_name):
        from pygeoiga.plot.nrbplotting_mpl import create_figure, p_cpoints, p_knots, p_curve, p_surface
        fig, [ax2, ax] = plt.subplots(1,2, figsize=(10,5),constrained_layout=True)
        ax = p_knots(knots,B,  ax=ax, dim=2, point=False, line=True, color ="k")
        ax = p_surface(knots,B, ax=ax, dim=2, color="blue", border = False, alpha=0.5)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_aspect(0.8)

        ax2 = p_cpoints(B, ax=ax2, dim=2, color="blue", linestyle="-", point=False, line=True)
        ax2 = p_cpoints(B, ax=ax2, dim=2, color="red", marker="o", point=True, line=False)
        n, m = B.shape[0], B.shape[1]
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.set_xlabel("$x$")
        ax2.set_ylabel("$y$", rotation=90)
        ax2.set_aspect(0.8)

        ax.set_title("Physical space ($x,y$)", fontsize=20)
        ax2.set_title("Control net", fontsize=20)
        fig.tight_layout(pad=0, h_pad=0, w_pad=0)
        fig.show()

        from pygeoiga.engine.NURB_engine import basis_function_array_nurbs
        fig3 = plt.figure(constrained_layout=True)
        gs = fig3.add_gridspec(2, 2, hspace=0, wspace=0,
                               width_ratios=[0.2, 1],
                               height_ratios=[1, 0.2])
        (ax_v, ax3), (no, ax_u) = gs.subplots(sharex=True, sharey=True)
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
            ax3.vlines(i, 0, 1, 'k')
        for j in knots[1]:
            ax3.hlines(j, 0, 1, 'k')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.set_axis_off()
        ax3.set_title("Parametric space ($u,v$)", fontsize=20)

        ax_u.set_xlabel("$u$")
        ax_v.set_ylabel("$v$")
        ax_v.set_yticks(knots[1][2:-2])
        ax_u.set_xticks(knots[0][2:-2])
        for ax in ax_u, ax_v, ax3, no:
            ax.label_outer()
        fig3.show()

        save = False
        if save or save_all:
            fig.savefig(fig_folder+file_name, **kwargs_savefig)
            fig3.savefig(fig_folder + file_name.split(".")[0]+"_parameter.pdf", **kwargs_savefig)

    from pygeoiga.nurb.cad import make_surface_biquadratic
    knots, B = make_surface_biquadratic()
    plot(B, knots, "B-spline_biquadratic.pdf")
    from pygeoiga.nurb.refinement import knot_insertion
    knot_ins_1 = [0.3, 0.6]
    knot_ins_0 = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    B, knots = knot_insertion(B, (2, 2), knots, knot_ins_0, direction=0)
    B, knots = knot_insertion(B, (2,2), knots, knot_ins_1, direction=1)
    plot(B, knots, "B-spline_biquadratic_refined.pdf")

def test_between_mappings():

    def plot(B, knots, file_name):
        from pygeoiga.plot.nrbplotting_mpl import create_figure, p_cpoints, p_knots, p_curve, p_surface
        fig, ax = plt.subplots(constrained_layout=True)

        ax = p_knots(knots,B,  ax=ax, dim=2, point=False, line=True, color ="k")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_aspect(0.8)
        ax.plot(0.7,3, "r*", markersize=10)
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
        ax2.plot(0.65, 0.45, "r*", markersize=10)

        ax_u.set_xlabel("$u$")
        ax_v.set_ylabel("$v$")
        ax_v.set_yticks(knots[1][2:-2])
        ax_u.set_xticks(knots[0][2:-2])
        for ax in ax_u, ax_v, ax2, no:
            ax.label_outer()
        fig2.show()

        fig3, ax3 = plt.subplots()
        ax3.vlines(-1, -1, 1, 'k')
        ax3.vlines(1, -1, 1, 'k')
        ax3.hlines(-1, -1, 1, 'k')
        ax3.hlines(1, -1, 1, 'k')
        ax3.spines['left'].set_position('center')
        ax3.spines['bottom'].set_position('center')

        # Eliminate upper and right axes
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)

        # Show ticks in the left and lower axes only
        ax3.xaxis.set_ticks_position('bottom')
        ax3.yaxis.set_ticks_position('left')
        ax3.set_xlabel(r"$\xi $", fontsize=15)
        ax3.set_ylabel(r"$\eta$", rotation=0, fontsize=15)
        ax3.xaxis.set_label_coords(1.05, 0.475)
        ax3.yaxis.set_label_coords(0.475, 1.05)
        ax3.set_yticks([-1,-0.5, 0.5, 1])
        ax3.set_xticks([-1,-0.5, 0.5, 1])
        ax3.set_aspect("equal")
        fig3.show()
        save = True
        if save or save_all:
            fig.savefig(fig_folder+file_name, **kwargs_savefig)
            fig2.savefig(fig_folder + file_name.split(".")[0]+"_parameter.pdf", **kwargs_savefig)
            fig3.savefig(fig_folder + file_name.split(".")[0] + "_element.pdf", **kwargs_savefig)

    from pygeoiga.nurb.cad import make_surface_biquadratic
    knots, B = make_surface_biquadratic()

    from pygeoiga.nurb.refinement import knot_insertion
    knot_ins_1 = [0.3, 0.6]
    knot_ins_0 = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    B, knots = knot_insertion(B, (2, 2), knots, knot_ins_0, direction=0)
    B, knots = knot_insertion(B, (2,2), knots, knot_ins_1, direction=1)
    plot(B, knots, "mapping.pdf")


def test_make_mp():

    def plot(geometry, file_name, c, l):
        from pygeoiga.plot.nrbplotting_mpl import p_surface, p_cpoints, p_knots, create_figure
        #fig, ax = create_figure("2d")
        fig, [ax, ax2] = plt.subplots(1,2, figsize=(10,3))
        for patch_id in geometry.keys():
            ax2 = p_surface(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax2, dim=2,
                           color=geometry[patch_id].get("color"), alpha=0.5)
            ax = p_cpoints(geometry[patch_id].get("B"), ax=ax, dim=2, color=geometry[patch_id].get("color"), marker="o",
                           linestyle="-",
                           point=False, line=True)
            ax2 = p_knots(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax2, dim=2, point=False,
                         line=True, color="k")
        ax.set_title("Control net")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_aspect("equal")

        ax2.set_title("Physical space ($x,y$)")
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.set_xlabel("$x$")
        ax2.set_ylabel("$y$")
        ax2.set_aspect("equal")
        if c:
            ax2.annotate("$\Omega_1$", (70, 80), fontsize=20)
            ax2.annotate("$\Omega_2$", (300, 80), fontsize=20)
            ax2.annotate("$\Omega_3$", (100, 240), fontsize=20)
            c=False

        fig.show()

        save = False
        if save or save_all:
            fig.savefig(fig_folder + file_name, **kwargs_savefig)

    from pygeoiga.nurb.cad import make_L_shape
    geometry = make_L_shape(refine=False)
    from pygeoiga.analysis.MultiPatch import patch_topology
    geometry, gDoF = patch_topology(geometry)
    plot(geometry, "multi_patch.pdf", True, True)

    geometry2 = make_L_shape(refine = True, knot_ins=([0.3, 0.5, 0.7], [0.4, 0.6]))
    geometry2, gDoF = patch_topology(geometry2)
    plot(geometry2, "multi_patch_refined.pdf", False, True)

def test_numbering_glob():
    def plot(geometry, file_name):
        from pygeoiga.plot.nrbplotting_mpl import p_surface, p_cpoints, p_knots, create_figure
        #fig, ax = create_figure("2d")
        fig, ax = plt.subplots(constrained_layout=True)
        for patch_id in geometry.keys():
            ax = p_cpoints(geometry[patch_id].get("B"), ax=ax, dim=2, color=geometry[patch_id].get("color"), marker="o",
                           linestyle="-",
                           point=True, line=True)

            P = geometry[patch_id].get("list_cp")
            glob_num = geometry[patch_id].get("glob_num")
            for count, point in enumerate(P):
                ax.annotate(str(glob_num[count]), point, xytext=(8, 5), textcoords="offset points")

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_aspect("equal")

        fig.show()

        save = False
        if save or save_all:
            fig.savefig(fig_folder + file_name, **kwargs_savefig)

    from pygeoiga.nurb.cad import make_L_shape
    geometry = make_L_shape(refine = False, knot_ins=([0.3, 0.5, 0.7], [0.4, 0.6]))
    from pygeoiga.analysis.MultiPatch import patch_topology
    geometry, gDoF = patch_topology(geometry)
    plot(geometry, "multi_patch_global_numbering.pdf")

    geometry2 = make_L_shape(refine = True, knot_ins=([0.3, 0.5, 0.7], [0.4, 0.6]))
    from pygeoiga.analysis.MultiPatch import patch_topology
    geometry2, gDoF2 = patch_topology(geometry2)
    plot(geometry2, "multi_patch_global_numbering_refined.pdf")

def test_global_stifffness_matrix():
    from pygeoiga.nurb.cad import make_L_shape
    from pygeoiga.analysis.MultiPatch import patch_topology

    geometry = make_L_shape(refine=True, knot_ins=([0.3, 0.5, 0.7], [0.4,0.6]))
    geometry, gDoF = patch_topology(geometry)
    print(gDoF)
    from pygeoiga.analysis.MultiPatch import form_k_IGA_mp
    K_glob = np.zeros((gDoF, gDoF))
    K_glob = form_k_IGA_mp(geometry, K_glob)

    fig, ax = plt.subplots(1,3, sharey=True, figsize=(10,3))
    for count, patch_id in enumerate(geometry):
        K = geometry[patch_id].get("K")
        print(K.shape)
        ax[count].spy(K)
    ax[0].set_xlabel("$K$ for $\Omega_1$", fontsize=15)
    ax[1].set_xlabel("$K$ for $\Omega_2$", fontsize=15)
    ax[2].set_xlabel("$K$ for $\Omega_3$", fontsize=15)
    fig.show()

    fig2, ax2 = plt.subplots()
    ax2.spy(K_glob)
    ax2.set_xlabel("Global $K$ for $\Omega$", fontsize=15)
    fig2.show()

    save = False
    if save or save_all:
        fig.savefig(fig_folder + "K_single_patch.pdf", **kwargs_savefig)
        fig2.savefig(fig_folder + "K_multi_patch.pdf", **kwargs_savefig)
