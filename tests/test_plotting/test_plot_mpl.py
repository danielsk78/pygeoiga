from pygeoiga.plot.nrbplotting_mpl import *
from pygeoiga.nurb.cad import *
import matplotlib.pyplot as plt

#matplotlib.use("Qt5Agg")

def test_plot_cpoints():
    knots, curve_1d = curve_in_3D()
    p_cpoints(curve_1d, dim=1, show=True)

    p_cpoints(curve_1d[:, :2], dim=1, show=True)

    knots, cp = surface_in_3D()
    p_cpoints(cp, dim=2, show=True, color ="blue")
    fig, ax = create_figure()
    ax = p_cpoints(cp, ax = ax, dim=3, show=False, color="black", marker=">", point=True, line=False)
    ax = p_cpoints(cp, ax=ax, dim=3, show=True, color="blue", linestyle="-", point=False, line=True)


def test_plot_knots():
    knots, curve_1d = curve_in_3D()
    p_knots([knots], curve_1d, dim = 3, show=True)
    p_knots([knots], curve_1d, dim=2, show=True, color ="blue")

    knots, cp = surface_in_3D()
    p_knots(knots, cp, dim=3, show=True, linestyle="--")
    p_knots(knots, cp[..., :2], dim=2, show=True, marker="s")

def test_plot_curve():
    knots, curve_1d = curve_in_3D()
    p_curve(knots, curve_1d, dim = 2, show = True, color = "red")
    p_curve(knots, curve_1d, dim = 3, show = True)

def test_plot_surface():
    knots, cp = surface_in_3D()
    p_surface(knots, cp, dim =3, show=True, color ="red")
    #p_surface(knots, cp, dim=2, show=True, color="red")

def test_plot_all():
    knots, cp = surface_in_3D()
    fig, ax = create_figure()
    ax = p_cpoints(cp, ax=ax, color="black", marker="s", point=True, line=False)
    ax = p_cpoints(cp, ax=ax, color="red", linestyle="--", point=False, line=True)
    ax = p_knots(knots, cp, ax=ax, point=True, line=True)
    ax = p_surface(knots, cp, ax=ax, color="blue", alpha=0.5)
    plt.show()

def test_plot_biquadratic():
    knots, cp = make_surface_biquadratic()
    fig, ax = create_figure("2d")
    ax = p_cpoints(cp, ax=ax, dim =2, color="black", marker="s", point=True, line=False)
    ax = p_cpoints(cp, ax=ax, dim = 2, color="red", linestyle="--", point=False, line=True)
    ax = p_knots(knots, cp, ax=ax, dim=2,point=True, line=True)
    ax = p_surface(knots, cp, ax=ax, dim=2, color="blue", alpha=0.5)
    plt.show()

def test_plot_quarter_disk():
    knots, cp = quarter_disk()
    fig, ax = create_figure("2d")
    ax = p_cpoints(cp, ax=ax, dim=2, color="black", marker="s", point=True, line=False)
    ax = p_cpoints(cp, ax=ax, dim=2, color="red", linestyle="--", point=False, line=True)
    ax = p_knots(knots, cp, ax=ax, dim=2, point=True, line=True)
    ax = p_surface(knots, cp, ax=ax, dim=2, color="blue", alpha=0.5)
    plt.show()
    
def test_plot_refined_quarter_disk():
    knots, cp = quarter_disk()
    shape = np.asarray(cp.shape)
    shape[-1] = cp.shape[-1] + 1
    B = np.ones((shape))
    B[..., :cp.shape[-1]] = cp

    fig, ax = create_figure()
    ax = p_cpoints(B, ax=ax, dim=2, color="black", marker="s", point=True, line=False)
    ax = p_cpoints(B, ax=ax, dim=2, color="red", linestyle="--", point=False, line=True)
    ax = p_knots(knots, B, ax=ax, dim=2, point=True, line=True)
    plt.show()
        
    from pygeoiga.nurb.refinement import knot_insertion
    direction = 0
    knot_ins = np.asarray([0.2, 0.5, 0.7])
    B_new, knots = knot_insertion(B, (2,2), knots, knot_ins, direction=direction)
    direction = 1
    B_new, knots = knot_insertion(B_new, (2, 2), knots, knot_ins, direction=direction)

    fig, ax = create_figure()
    ax = p_cpoints(B_new, ax=ax, dim=2, color="black", marker="s", point=True, line=False)
    ax = p_cpoints(B_new, ax=ax, dim=2, color="red", linestyle="--", point=False, line=True)
    ax = p_knots(knots, B_new, ax=ax, dim=2, point=True, line=True)
    plt.show()

def test_plot_refined_quarter_disk_multiplicity():
    knots, B = quarter_disk()
    #shape = np.asarray(cp.shape)
    #shape[-1] = cp.shape[-1] + 1
    #B = np.ones((shape))
    #B[..., :cp.shape[-1]] = cp
    fig, ax = create_figure("2d")
    ax = p_cpoints(B, ax=ax, dim=2, color="black", marker="s", point=True, line=False)
    ax = p_cpoints(B, ax=ax, dim=2, color="red", linestyle="--", point=False, line=True)
    ax = p_knots(knots, B, ax=ax, dim=2, point=True, line=True)
    plt.show()

    from pygeoiga.nurb.refinement import knot_insertion
    direction = 0
    knot_ins = np.asarray([0.2, 0.2, 0.5,0.5, 0.7, 0.7])
    B_new, knots = knot_insertion(B, (2, 2), knots, knot_ins, direction=direction)
    direction = 1
    B_new, knots = knot_insertion(B_new, (2, 2), knots, knot_ins, direction=direction)

    fig, ax = create_figure("2d")
    ax = p_cpoints(B_new, ax=ax, dim=2, color="black", marker="s", point=True, line=False)
    ax = p_cpoints(B_new, ax=ax, dim=2, color="red", linestyle="--", point=False, line=True)
    ax = p_knots(knots, B_new, ax=ax, dim=2, point=True, line=True)
    plt.show()

def test_plot_qurter_disk_surface():
    knots, cp = quarter_disk()
    fig, ax = plt.subplots()
    ax = p_cpoints(cp, ax=ax, dim=2, color="black", marker="s", point=True, line=False)
    #ax = p_cpoints(cp, ax=ax, dim=2, color="red", linestyle="--", point=False, line=True)
    ax = p_knots(knots, cp, ax=ax, dim=2, point=True, line=True)
    ax = p_surface(knots, cp, ax=ax, dim=2, color="blue", alpha=0.5)
    plt.show()

def test_plot_3_layer_anticline_mp():
    geometry=make_3_layer_patches(refine=False)
    fig, ax = create_figure("2d")
    for patch_id in geometry.keys():
        ax = p_surface(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2, color=geometry[patch_id].get("color"), alpha=0.5)
        ax = p_cpoints(geometry[patch_id].get("B"), ax=ax, dim=2, color="black", marker="s", point=True, line=False)
        #ax = p_cpoints(geometry[patch_id].get("B"), ax=ax, dim=2, color="red", linestyle="--", point=False, line=True)
        ax = p_knots(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2, point=False, line=True)
    plt.show()
    geometry=make_3_layer_patches(refine=True)
    fig, ax = create_figure("2d")
    for patch_id in geometry.keys():
        ax = p_surface(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2, color=geometry[patch_id].get("color"), alpha=0.5)
        #ax = p_cpoints(geometry[patch_id].get("B"), ax=ax, dim=2, color="black", marker="s", point=True, line=False)
        #ax = p_cpoints(geometry[patch_id].get("B"), ax=ax, dim=2, color="red", linestyle="--", point=False, line=True)
        ax = p_knots(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2, point=False, line=True)
    plt.show()


def test_plot_fault_MP():
    geometry = make_fault_model(refine=False)
    fig, ax = create_figure("2d")
    for patch_id in geometry.keys():
        ax = p_surface(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2,
                       color=geometry[patch_id].get("color"), alpha=0.5)
        ax = p_cpoints(geometry[patch_id].get("B"), ax=ax, dim=2, color="black", marker="s", markersize=2, point=True,
                       line=False)
        # ax = p_cpoints(geometry[patch_id].get("B"), ax=ax, dim=2, color="red", linestyle="--", point=False, line=True)
        ax = p_knots(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2, point=False, line=True)
    plt.show()
    geometry = make_fault_model(refine=True)
    fig, ax = create_figure("2d")
    for patch_id in geometry.keys():
        ax = p_surface(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2,
                       color=geometry[patch_id].get("color"), alpha=0.5)
        ax = p_cpoints(geometry[patch_id].get("B"), ax=ax, dim=2, color="black", marker="s", markersize=2,point=True, line=False)
        # ax = p_cpoints(geometry[patch_id].get("B"), ax=ax, dim=2, color="red", linestyle="--", point=False, line=True)
        ax = p_knots(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2, point=False, line=True)
    plt.show()

def test_plot_L_MP():
    geometry = make_L_shape(refine=True)
    fig, ax = create_figure("2d")
    for patch_id in geometry.keys():
        ax = p_surface(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2,
                       color=geometry[patch_id].get("color"), alpha=0.5)
        ax = p_cpoints(geometry[patch_id].get("B"), ax=ax, dim=2, color="black", marker="s", markersize=2,point=True, line=False)
        # ax = p_cpoints(geometry[patch_id].get("B"), ax=ax, dim=2, color="red", linestyle="--", point=False, line=True)
        ax = p_knots(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2, point=False, line=True)
    plt.show()

def test_plot_salt_dome():
    geometry = make_salt_dome()
    fig, ax = create_figure("2d")
    for patch_id in geometry.keys():
        ax = p_surface(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2,
                       color=geometry[patch_id].get("color"), alpha=0.5)
        ax = p_cpoints(geometry[patch_id].get("B"), ax=ax, dim=2, color="black", marker="s", markersize=2, point=True,
                       line=False)
        #ax = p_cpoints(geometry[patch_id].get("B"), ax=ax, dim=2, color="red", linestyle="--", point=False, line=True)
        ax = p_knots(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2, point=False, line=True)
    plt.show()

    geometry = make_salt_dome(refine=True)
    fig, ax = create_figure("2d")
    for patch_id in geometry.keys():
        ax = p_surface(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2,
                       color=geometry[patch_id].get("color"), alpha=0.5)
        ax = p_cpoints(geometry[patch_id].get("B"), ax=ax, dim=2, color="black", marker="s", markersize=2, point=True,
                       line=False)
        # ax = p_cpoints(geometry[patch_id].get("B"), ax=ax, dim=2, color="red", linestyle="--", point=False, line=True)
        ax = p_knots(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2, point=False, line=True)
    plt.show()

def test_plot_unconformity():
    from pygeoiga.nurb.cad import _make_unconformity_model
    geometry = make_unconformity_model()
    fig, ax = create_figure("2d")
    for patch_id in geometry.keys():
        ax = p_surface(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2,
                       color=geometry[patch_id].get("color"), alpha=0.5)
        ax = p_cpoints(geometry[patch_id].get("B"), ax=ax, dim=2, color="black", marker="s", markersize=2, point=True,
                       line=False)
        #ax = p_cpoints(geometry[patch_id].get("B"), ax=ax, dim=2, color="red", linestyle="--", point=False, line=True)
        ax = p_knots(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2, point=False, line=True)
    plt.show()

    geometry = make_unconformity_model(refine=True)
    fig, ax = create_figure("2d")
    for patch_id in geometry.keys():
        ax = p_surface(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2,
                       color=geometry[patch_id].get("color"), alpha=0.5)
        ax = p_cpoints(geometry[patch_id].get("B"), ax=ax, dim=2, color="black", marker="s", markersize=2, point=True,
                       line=False)
        # ax = p_cpoints(geometry[patch_id].get("B"), ax=ax, dim=2, color="red", linestyle="--", point=False, line=True)
        ax = p_knots(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), ax=ax, dim=2, point=False, line=True)
    plt.show()