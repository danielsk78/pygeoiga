import numpy as np
from pygeoiga.analysis.MultiPatch import patch_topology, form_k_IGA_mp, boundary_condition_mp, map_MP_elements
from pygeoiga.analysis.common import solve, boundary_condition
from pygeoiga.analysis.iga import form_k_IGA
from pygeoiga.analysis.bezier_FE import form_k
import os
import pygeoiga as gn
datapath = os.path.abspath(gn.myPath+"/../tests/chapter_figures/data/") + os.sep


def plot_pvista(geometry):
    from pygeoiga.plot.nrbplotting_vtk import create_figure, p_show, p_cpoints, p_surface, p_knots
    from pygeoiga.plot.solution_vtk import p_temperature

    p = create_figure()
    for patch_id in geometry.keys():
        #p = p_cpoints(geometry[patch_id].get("B"), p=p )
        #p=p_knots(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), p=p)
        p=p_surface(geometry[patch_id].get("knots"), geometry[patch_id].get("B"), p=p,
                    color=geometry[patch_id].get("color"), interactive=False)
        x = geometry[patch_id].get("x_sol")
        y = geometry[patch_id].get("y_sol")
        t = geometry[patch_id].get("t_sol")
        p = p_temperature(x,y,t, p=p, show = False, label=patch_id)

    p_show(p)

def test_plot_solution_3_layer_mp():
    from pygeoiga.nurb.cad import make_3_layer_patches
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
    plot_pvista(geometry)


def test_plot_solution_mp_fault():
    from pygeoiga.nurb.cad import make_fault_model
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
    from pygeoiga.analysis.MultiPatch import map_MP_elements

    geometry = map_MP_elements(geometry, D)
    plot_pvista(geometry)

def test_plot_solution_salt_dome():
    from pygeoiga.nurb.cad import make_salt_dome
    geometry = make_salt_dome(refine=True, knot_ins=[np.arange(0.2,1,0.2), np.arange(0.2,1,0.2)])#refine=np.arange(0.05,1,0.05))
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
    from pygeoiga.analysis.MultiPatch import map_MP_elements

    geometry = map_MP_elements(geometry, D)
    plot_pvista(geometry)

def test_plot_solution_salt_dome_bezier():
    from pygeoiga.nurb.cad import make_salt_dome
    from pygeoiga.analysis.MultiPatch import bezier_extraction_mp
    geometry = make_salt_dome(refine=True, knot_ins=[np.arange(0.2,1,0.2), np.arange(0.2,1,0.2)])#refine=np.arange(0.05,1,0.05))
    geometry, gDoF = patch_topology(geometry)
    geometry = bezier_extraction_mp(geometry)
    K_glob = np.zeros((gDoF, gDoF))

    from pygeoiga.analysis.MultiPatch import form_k_bezier_mp
    K_glob = form_k_bezier_mp(geometry, K_glob)

    D = np.zeros(gDoF)
    b = np.zeros(gDoF)

    T_t = 10  # [°C]
    T_b = 90  # [°C]
    T_l = None
    T_r = None
    bc, D = boundary_condition_mp(geometry, D, T_t, T_b, T_l, T_r)
    bc["gDOF"] = gDoF
    D, b = solve(bc, K_glob, b, D)
    from pygeoiga.analysis.MultiPatch import map_MP_elements

    geometry = map_MP_elements(geometry, D)
    plot_pvista(geometry)
