import gempy as gp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('precision', 2)

def create_gempy_model(resolution=[20, 20, 20], type=2):
    if type==1:
        data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
        path_to_data = data_path + "/data/input_data/jan_models/"
        geo_data = gp.create_data('fold', extent=[0, 1000, 0, 1000, 0, 1000], resolution=resolution,
                                  path_o=path_to_data + "model2_orientations.csv",
                                  path_i=path_to_data + "model2_surface_points.csv")
        gp.map_stack_to_surfaces(geo_data, {"Strat_Series": ('rock2', 'rock1'), "Basement_Series": ('basement')})


        interp_data = gp.set_interpolator(geo_data, theano_optimizer='fast_compile')

        sol = gp.compute_model(geo_data)
    elif type==2:
        data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
        path_to_data = data_path + "/data/input_data/jan_models/"

        geo_data = gp.create_data('unconformity', extent=[0, 1000, 0, 1000, 0, 1000], resolution=[50, 50, 50],
                                  path_o=path_to_data + "model6_orientations.csv",
                                  path_i=path_to_data + "model6_surface_points.csv")
        gp.map_stack_to_surfaces(geo_data, {"Strat_Series1": ('rock3'),
                                            "Strat_Series2": ('rock2', 'rock1'),
                                            "Basement_Series": ('basement')})
        interp_data = gp.set_interpolator(geo_data, theano_optimizer='fast_compile')
        sol = gp.compute_model(geo_data)

    return geo_data

def plot_gempy(geo_model: gp.Project):
    vertices = geo_model.surfaces.df['vertices'][:-1].copy()
    fig = plt.figure("Curve")
    ax = fig.add_subplot(111, projection='3d')
    for surf in vertices:
            ax.plot_trisurf(surf[:, 0], surf[:, 1],surf[:, 2])
    plt.show()

def plot_control_points_multiple(control_points: list):
    if not isinstance(control_points, list):
        control_points=[control_points]

    fig = plt.figure("Control points")
    ax = fig.add_subplot(111, projection='3d')
    for surf in control_points:
        ax.plot_wireframe(surf[..., 0], surf[..., 1], surf[..., 2])
    plt.show()

def plot_surface_NURBS(xyz:list):
    if not isinstance(xyz, list):
        xyz=[xyz]
    fig = plt.figure("Surfaces")
    ax = fig.add_subplot(111, projection='3d')
    for points in xyz:
        ax.plot_trisurf(points[..., 0], points[..., 1], points[..., 2])
    plt.show()

def test_extract_control_points():
    from pygeoiga.nurb.gempy_to_NURBS import extract_control_points_from_gempy
    geo_model = create_gempy_model()
    plot_gempy(geo_model=geo_model)
    surface_list = extract_control_points_from_gempy(geo_model=geo_model)
    plot_control_points_multiple(surface_list)

def test_make_knot():
    from pygeoiga.nurb.gempy_to_NURBS import make_knot_vector
    degree_U = 2
    len_U = 30
    degree_V = 3
    len_V = 51
    degree_W = 5
    len_W = 103

    U = make_knot_vector(degree_U, len_U)
    V = make_knot_vector(degree_V, len_V)
    W = make_knot_vector(degree_W, len_W)

    assert len(U)== degree_U + len_U + 1
    assert len(V)== degree_V + len_V + 1
    assert len(W) == degree_W + len_W + 1

def test_convert_model():
    from pygeoiga.nurb.gempy_to_NURBS import construct_NURBS_from_gempy
    geo_model = create_gempy_model(resolution=[20, 20, 20])
    plot_gempy(geo_model=geo_model)

    NURBS_surfaces = construct_NURBS_from_gempy(geo_model=geo_model,
                                                degree=2,
                                                engine="python",
                                                resolution=300)

    nurb_cp = [surf_NURB.cpoints for surf_NURB in NURBS_surfaces]
    nurb_point= [surf_NURB.model for surf_NURB in NURBS_surfaces]
    plot_control_points_multiple(nurb_cp)
    plot_surface_NURBS(nurb_point)

def test_decrease_knots():
    deviation=1000000 # -> from here it starts to under-refine the curve but it changes the curve

    geo_model = create_gempy_model(resolution=[20, 20, 20])
    plot_gempy(geo_model=geo_model)
    from pygeoiga.nurb.gempy_to_NURBS import construct_NURBS_from_gempy
    NURBS_surfaces = construct_NURBS_from_gempy(geo_model=geo_model,
                                                degree=3,
                                                engine="igakit",
                                                resolution=20)

    from pygeoiga.nurb.gempy_to_NURBS import decrease_knots
    new_surf = [decrease_knots(NURBS=surf, deviation=deviation) for surf in NURBS_surfaces]
    nurb_cp = [surf.control for surf in new_surf]
    plot_control_points_multiple(nurb_cp)

def plot_pathdict(geo_model, section_name):
    from gempy.core.grid_modules import section_utils
    pathdict, cdict, extent = section_utils.get_polygon_dictionary(geo_model, section_name)
    import matplotlib.path
    import matplotlib.patches as patches
    surfaces=list(geo_model.surfaces.df['surface'])[:-1][::-1]
    fig, ax = plt.subplots()
    for formation in surfaces:
        for path in pathdict.get(formation):
            if path !=[]:
                if type(path) == matplotlib.path.Path:
                    patch = patches.PathPatch(path, fill=False, lw=1, edgecolor=cdict.get(formation, 'k'))
                    ax.add_patch(patch)
                elif type(path) == list:
                    for subpath in path:
                        assert type(subpath == matplotlib.path.Path)
                        patch = patches.PathPatch(subpath, fill=False, lw=1, edgecolor=cdict.get(formation, 'k'))
                        ax.add_patch(patch)
    ax.set_ylim(extent[2:4])
    ax.set_xlim(extent[:2])
    plt.show()

def plot_2d_control_points(control_points):
    if not isinstance(control_points, list):
        control_points=[control_points]

    fig, ax = plt.subplots()
    for curve in control_points:
        ax.plot(curve[:, 0],curve[:, 1], marker='s')
    plt.show()

def test_2d_from_gempy_to_NURBS_NOT_WORKING():
    geo_model = create_gempy_model(resolution=[5,5,5])
    extent = geo_model.grid.regular_grid.extent
    section_dict = {'section1': ([extent[0], extent[2] / 2], [extent[1], extent[3] / 2], [50, 50])}
    geo_model.set_section_grid(section_dict)
    gp.compute_model(geo_model)
    plot_pathdict(geo_model, 'section1')
    from pygeoiga.nurb.gempy_to_NURBS import extract_control_points_from_cross_section

    control_points = extract_control_points_from_cross_section(geo_model, 'section1')
    plot_2d_control_points(control_points)

def test_2d_from_gempy_to_NURBS():
    geo_model = create_gempy_model(resolution=[20, 20, 20])
    from pygeoiga.nurb.gempy_to_NURBS import extract_2d_control_points_from_3d
    control_points = extract_2d_control_points_from_3d(geo_model=geo_model, y=500)
    plot_2d_control_points(control_points)

def test_create_2d_from_gempy():
    geo_model = create_gempy_model(resolution=[50, 50, 50])
    from pygeoiga.nurb.gempy_to_NURBS import extract_2d_control_points_from_3d
    control_points = extract_2d_control_points_from_3d(geo_model=geo_model, y=500)
    plot_2d_control_points(control_points)

def plot_curve_nurbs(nrbs):
    if not isinstance(nrbs, list):
        nrbs = [nrbs]
    fig, ax = plt.subplots()
    for points in nrbs:
        ax.plot(points[0], points[1])
    plt.show()


def test_generate_2d_NURB_from_gempy():
    geo_model = create_gempy_model(resolution=[50, 50, 50])
    from pygeoiga.nurb.gempy_to_NURBS import extract_2d_control_points_from_3d
    control_points = extract_2d_control_points_from_3d(geo_model=geo_model, y=500)
    plot_2d_control_points(control_points)
    from pygeoiga.nurb.gempy_to_NURBS import construct_NURBS_from_gempy
    nrbs = construct_NURBS_from_gempy(geo_model=geo_model, degree=2, y=500, resolution=500)
    nurb_point = [curve_NURB.model for curve_NURB in nrbs]
    plot_curve_nurbs(nurb_point)

def test_decrease_knot_2d():
    geo_model = create_gempy_model(resolution=[50, 50, 50])
    from pygeoiga.nurb.gempy_to_NURBS import construct_NURBS_from_gempy
    nrbs = construct_NURBS_from_gempy(geo_model=geo_model, degree=2, y=500, resolution=300)
    nurb_point = [curve_NURB.model for curve_NURB in nrbs]
    control_points = [curve.cpoints for curve in nrbs]
    plot_2d_control_points(control_points)
    plot_curve_nurbs(nurb_point)

    deviation = 10000  # -> from here it starts to under-refine the curve but it changes the curve
    from pygeoiga.nurb.gempy_to_NURBS import decrease_knots
    new_curve = [decrease_knots(NURBS=curve, deviation=deviation) for curve in nrbs]
    nurb_cp = [curve.control for curve in new_curve]
    nurb_point = [curve(np.linspace(0,1,300)).T for curve in new_curve]
    plot_2d_control_points(nurb_cp)
    plot_curve_nurbs(nurb_point)
