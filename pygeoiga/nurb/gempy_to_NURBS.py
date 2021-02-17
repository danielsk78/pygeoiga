import numpy as np
import pandas as pd

def extract_control_points_from_gempy(geo_model):
    """
    Takes the vertices of the geo_model and translate them into the matrix
    required for NURBS control points. Ej. (1400,3) -> (50, 94, 3)

    Args:
        geo_model:

    Returns:
        list of control points ready as input for NURBS

    """
    # edges = geo_model.surfaces.df['edges'][:-1].copy()
    vertices = geo_model.surfaces.df['vertices'][:-1].copy() #To exclude the basement
    surfaces = []
    for surf in vertices:
        df = pd.DataFrame({'col1': surf[:, 0],
                           'col2': surf[:, 1],
                           'col3': surf[:, 2]
                           })
        df = df.sort_values(['col1', 'col2'])
        l_y = df['col2'].values.tolist()
        val_y = np.sort(list(set([x for x in l_y if l_y.count(x) > 1])))
        new_shape = []
        new_ = []
        for i in val_y:
            temp = df.loc[df['col2'] == i].to_numpy()
            shape_temp = temp.shape

            if new_ == []:
                new_.append(temp)
                new_shape.append(shape_temp[0])
            elif new_[-1].shape[0] - 5 < shape_temp[0] < new_[-1].shape[0] + 5:
                new_.append(temp)
                new_shape.append(shape_temp[0])

        mode_shape = np.argmax(np.bincount(new_shape))
        for i in range(len(new_)):
            if new_[i].shape[0] != mode_shape:
                if new_[i].shape[0] < mode_shape:
                    to_add = mode_shape - new_[i].shape[0]
                    to_stack = np.asarray([new_[i][-1] for j in range(to_add)])
                    new_[i] = np.vstack((new_[i], to_stack))
                else:
                    to_rest = new_[i].shape[0] - mode_shape
                    for k in range(to_rest):
                        new_[i] = np.delete(new_[i], int(new_[i].shape[0] / 2), 0)

        surfaces.append(np.asarray(new_))

    return surfaces

def extract_2d_control_points_from_3d(geo_model, y):
    """
    Calculates the 3d surface/control points and then slice along the coordinate y to give a curve
    Args:
        geo_model: gempy model
        y: part cut the model giving the values for x and z
    Returns:
        control points of curves

    """
    surfaces = extract_control_points_from_gempy(geo_model)
    control_points = []
    for surf in surfaces:
        spacing = surf[...,1][1,1]-surf[...,1][0,0]
        slice_num = [i for i in range(surf.shape[0]-1) if y-spacing<surf[...,1][i,0]<y+spacing ]
        df = pd.DataFrame({'col1': surf[slice_num[0], :, :][:,0], 'col2': surf[slice_num[0], :, :][:,-1]})
        df = df.sort_values(['col1'], ignore_index=True)
        control_points.append(df.to_numpy())
    return control_points

def make_knot_vector(degree:int, len_cp:int):
    """
    From the degree of the curve and the length of the knot vector generate a non-uniform knot vector
    Follows the rule of m=n+p+1
    Args:
        degree: degree of curve
        len_cp: length of knot vector
    Returns:

    """
    U = np.ones(len_cp+degree+1)
    to_fit = len(U[degree + 1:-degree - 1])
    num_add = to_fit/degree
    vec_add=np.asarray([])
    if num_add.is_integer():
        for i in range(degree):
            vec_add = np.concatenate((vec_add, np.linspace(0.01, 0.09, int(num_add))))
        U[degree + 1:-degree - 1] = vec_add
    else:
        for i in range(degree):
            vec_add = np.concatenate((vec_add, np.linspace(0.01, 0.09, int(np.ceil(num_add)))))#, decimals=0)))))
        U[degree + 1:-degree - 1] = vec_add[:to_fit]
    U = np.sort(U)
    U[-degree-1:]=1
    U[:degree+1]=0
    return U

def construct_NURBS_from_gempy(geo_model, degree:int, engine="python", resolution=30, y=None):
    """
    Create a nurbs surface from a gempy model for every surface in the model. When several surfaces it return
    a list of NURBS surfaces
    Args:
        geo_model: gempy geo_model
        degree: degree for the NURBS surface
        engine: for he NURBS construction ("python","gempyExplicit","igakit")
        resolution: number of points in the surface (point cloud)

    Returns:
        list of NURBS surfaces

    """
    from pygeoiga.nurb.nurb_creation import NURB
    control_points = extract_control_points_from_gempy(geo_model)
    if y is not None:
        control_points = extract_2d_control_points_from_3d(geo_model, y)
    NURBS_object = []
    for cp in control_points:
        len_U= cp.shape[0]
        U = make_knot_vector(degree, len_U)
        if y is None:
            len_V = cp.shape[1]
            V = make_knot_vector(degree, len_V)
            knot_vector = [U, V]
        else:
            knot_vector = [U]

        NURBS_object.append(NURB(cp, knot_vector, resolution=resolution, engine=engine))

    return NURBS_object

def decrease_knots(NURBS, deviation):
    """
    Takes a NURBS object that contains all the information and then decrease the knots to reduce the ammount of control points
    Args:
        NURBS: NURBS object.
        deviation: to approximate the curve
    Returns:
        a copy of the NURBS igakit object with decreased control points
    """
    from igakit.nurbs import NURBS as iga_NURBS
    if not isinstance(NURBS, iga_NURBS):
        if NURBS._NURB is None:
            NURBS.create_model(engine="igakit")
            assert NURBS._NURB is not None
            nrb = NURBS._NURB
        else:
            nrb = NURBS._NURB

    print("Shape before refining:", nrb.control.shape)
    new_surf = nrb.clone()
    for i in new_surf.knots[0]:
        new_surf = new_surf.remove(0, i, deviation=deviation)
    if NURBS.dim > 1:
        for j in new_surf.knots[1]:
            new_surf = new_surf.remove(1, j, deviation=deviation)
    print("Shape after refining:", new_surf.control.shape)

    return new_surf

def extract_control_points_from_cross_section(geo_model, x, y):
    vertices = geo_model.surfaces.df['vertices'][:-1].copy()  # To exclude the basement
    surfaces = []
    for surf in vertices:
        df = pd.DataFrame({'col1': surf[:, 0],
                           'col2': surf[:, 1],
                           'col3': surf[:, 2]
                           })

        l_y = df['col2'].values.tolist()
        val_y = np.sort(list(set([x for x in l_y if l_y.count(x) > 1])))
        new_shape = []
        new_ = []
        for i in val_y:
            temp = df.loc[df['col2'] == i].to_numpy()
            shape_temp = temp.shape

            if new_ == []:
                new_.append(temp)
                new_shape.append(shape_temp[0])
            elif new_[-1].shape[0] - 5 < shape_temp[0] < new_[-1].shape[0] + 5:
                new_.append(temp)
                new_shape.append(shape_temp[0])

        mode_shape = np.argmax(np.bincount(new_shape))
        for i in range(len(new_)):
            if new_[i].shape[0] != mode_shape:
                if new_[i].shape[0] < mode_shape:
                    to_add = mode_shape - new_[i].shape[0]
                    to_stack = np.asarray([new_[i][-1] for j in range(to_add)])
                    new_[i] = np.vstack((new_[i], to_stack))
                else:
                    to_rest = new_[i].shape[0] - mode_shape
                    for k in range(to_rest):
                        new_[i] = np.delete(new_[i], int(new_[i].shape[0] / 2), 0)

        surfaces.append(np.asarray(new_))

    return surfaces

def _extract_control_points_from_cross_section(geo_model, section_name:str):
    #TODO: ask miguel for an easier way of doing this
    import scipy
    grids = geo_model.get_active_grids()
    extent = geo_model.grid.regular_grid.extent
    if 'sections' not in grids:
        section_dict = {'section1': ([extent[0], extent[3]/2], [extent[1], extent[3]/2], [50, 50])}
        geo_model.set_section_grid(section_dict)
        import gempy as gp
        sol = gp.compute_model(geo_model)
        section_name = 'section1'

    from gempy.core.grid_modules import section_utils
    polygondict, cdict, extent = section_utils.get_polygon_dictionary(geo_model, section_name)

    surfaces = list(geo_model.surfaces.df['surface'])[:-1][::-1]
    control_points=[]
    for formation in surfaces:
        for polygon in polygondict.get(formation):
            if polygon != []:
                vertices = polygondict[formation][0].vertices
                df = pd.DataFrame({'col1': vertices[:, 0],
                                   'col2': vertices[:, 1],
                                  #'codes': polygondict[formation][0].codes
                                    })

                #df = df.sort_values(['col2'], ignore_index=True, ascending=False )
                #df = df.sort_values(['col1'], ignore_index=True)#, ascending=False)
                df = df.sort_values(['col1','col2'], ignore_index=True, ascending=True)

                df = df.drop_duplicates(subset=['col1'], keep='first', ignore_index=True )
                mat = scipy.spatial.distance.cdist(df[['col1', 'col2']],
                                                   df[['col1', 'col2']], metric='euclidean')

                new_df = pd.DataFrame(mat)
                closest = np.where(new_df.eq(new_df[new_df != 0].min(), 0), new_df.columns, False)
                df['close'] = [i[i.astype(bool)].tolist() for i in closest]
                cp=[]
                #max_count = df.shape[0]
                for count, position in enumerate(df.to_numpy()):
                    #if count + 1 ==
                    #print(count)
                    if position[-1]!=[]:
                        if position[-1][0] >= count+1:
                            cp.append(position[:2])
                control_points.append(np.asarray(cp))

    return control_points



