from pygeoiga.plot.nrbplotting_vtk import *
from pygeoiga.nurb.cad import *

def test_create_fig():
    p = create_figure()
    p_show(p)

def test_create_surf():
    knots, cp = surface_in_3D()
    p_surface(knots,cp, show = True, points=False)

def test_plot_cpoints():
    knots, cp = surface_in_3D()
    p_cpoints(cp, show=True)

def test_create_surf_animated():
    knots, cp = surface_in_3D()
    p_surface(knots, cp, show=True, points=True)

def test_combine():
    knots, cp = surface_in_3D()
    p = create_figure()
    p = p_cpoints(cp, p=p)
    p = p_surface(knots, cp, p=p, interactive = False)
    p_show(p)

def test_knot():
    knots, curve_1d = curve_in_3D()
    p_knots(knots, curve_1d, show=True)

    knots, cp = surface_in_3D()
    p_knots(knots, cp, show=True)

def test_all():
    knots, cp = surface_in_3D()
    p = create_figure()
    p_knots(knots, cp, p=p)
    p_cpoints(cp, p=p)
    p_surface(knots, cp, p=p, interactive=True, opacity=0.5)
    p_show(p)

def test_plot_biquadratic():
    knots, cp = make_surface_biquadratic()
    p = create_figure()
    p_knots(knots, cp, p=p)
    p_cpoints(cp, p=p)
    p_surface(knots, cp, p=p, interactive=False, opacity=0.5)
    p_show(p)

def test_plot_quarter_disk():
    knots, cp = quarter_disk()
    p = create_figure()
    p_knots(knots, cp, p=p)
    p_cpoints(cp, p=p)
    p_surface(knots, cp, p=p, interactive=False, opacity=0.5)
    p_show(p)


def test_plot_refined_quarter_disk():
    knots, cp = quarter_disk()
    shape = np.asarray(cp.shape)
    shape[-1] = cp.shape[-1] + 1
    B = np.ones((shape))
    B[..., :cp.shape[-1]] = cp

    p = create_figure()
    p_knots(knots, B[...,:-1], p=p)
    p_cpoints(B[...,:-1], p=p)
    p_show(p)

    from pygeoiga.nurb.refinement import knot_insertion
    direction = 0
    knot_ins = np.asarray([0.2, 0.5, 0.7])
    B_new, knots = knot_insertion(B, (2, 2), knots, knot_ins, direction=direction)
    direction = 1
    B_new, knots = knot_insertion(B_new, (2, 2), knots, knot_ins, direction=direction)

    p = create_figure()
    p_knots(knots, B_new[...,:-1], p=p)
    p_cpoints(B_new[...,:-1], p=p)
    p_show(p)


def test_plot_refined_quarter_disk_multiplicity():
    knots, cp = quarter_disk()
    shape = np.asarray(cp.shape)
    shape[-1] = cp.shape[-1] + 1
    B = np.ones((shape))
    B[..., :cp.shape[-1]] = cp

    p = create_figure()
    p_knots(knots, B[..., :-1], p=p)
    p_cpoints(B[..., :-1], p=p)
    p_show(p)

    from pygeoiga.nurb.refinement import knot_insertion
    direction = 0
    knot_ins = np.asarray([0.2, 0.2, 0.5, 0.5, 0.7, 0.7])
    B_new, knots = knot_insertion(B, (2, 2), knots, knot_ins, direction=direction)
    direction = 1
    B_new, knots = knot_insertion(B_new, (2, 2), knots, knot_ins, direction=direction)

    p = create_figure()
    p_knots(knots, B_new[...,:-1], p=p)
    p_cpoints(B_new[...,:-1], p=p)
    p_show(p)
