import numpy as np
from collections import OrderedDict

def make_surface_square(XY= np.asarray([[0, -10], [100, -10], [0, 10], [100, 10]]), degree = 2):
    """
    Generate square nurb with corners [X1Y1, X2Y1, X1Y2, Y2Y2]
    Args:
        XY:
        degree:

    Returns:

    """
    B = np.zeros((degree+1,degree+1, 3))
    B[:,:,2] = 1
    p1 = XY[0, :]
    p2 = XY[1, :]
    p3 = XY[2, :]
    p4 = XY[3, :]

    if degree == 1:
        U =  np.asarray([0, 0, 1, 1])
        V = np.asarray([0, 0, 1, 1])

        B[0, 0,:2] = p1
        B[1, 0, :2] = p2
        B[0, 1, :2] = p3
        B[1, 1, :2] = p4

    elif degree == 2:
        U = np.asarray([0, 0, 0, 1, 1, 1])
        V = np.asarray([0, 0, 0, 1, 1, 1])

        B[0, 0, :2] = p1
        B[1, 0, :2] = (p1 + p3) / 2
        B[2, 0, :2] = p3
        B[0, 1, :2] = (p1 + p2) / 2
        B[1, 1, :2] = (p1 + p2 + p3 + p4) / 4
        B[2, 1, :2] = (p3 + p4) / 2
        B[0, 2, :2] = p2
        B[1, 2, :2] = (p2 + p4) / 2
        B[2, 2, :2] = p4

    elif degree == 3:
        U = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
        V = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])

        B[0, 0, :2] = p1
        B[1, 0, :2] = (2*p1 + p3) / 3
        B[2, 0, :2] = (p1+2*p3) / 3
        B[3, 0, :2] = p3

        B[0, 1, :2] = (2 * p1 + p2) / 3
        B[1, 1, :2] = (4 * p1 + 2 * p2 + 2 * p3 + p4) / 9
        B[2, 1, :2] = (2 * p1 + p2 + 4 * p3 + 2 * p4) / 9
        B[3, 1, :2] = (2 * p3 + p4) / 3

        B[0, 2, :2] = (p1 + 2 * p2) / 3
        B[1, 2, :2] = (2 * p1 + 4 * p2 + p3 + 2 * p4) / 9
        B[2, 2, :2] = (p1 + 2 * p2 + 2 * p3 + 4 * p4) / 9
        B[3, 2, :2] = (p3 + 2 * p4) / 3

        B[0, 3, :2] = p2
        B[1, 3, :2] = (2 * p2 + p4) / 3
        B[2, 3, :2] = (p2 + 2 * p4) / 3
        B[3, 3, :2] = p4

    else: raise NotImplementedError

    return U, V, B

def make_surface_3d():
    """ Construct a surface in 3d space"""
    C = np.zeros((3, 5, 5))
    C[:, :, 0] = [[0.0, 3.0, 5.0, 8.0, 10.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0],
                  [2.0, 2.0, 7.0, 7.0, 8.0], ]
    C[:, :, 1] = [[0.0, 3.0, 5.0, 8.0, 10.0],
                  [3.0, 3.0, 3.0, 3.0, 3.0],
                  [0.0, 0.0, 5.0, 5.0, 7.0], ]
    C[:, :, 2] = [[0.0, 3.0, 5.0, 8.0, 10.0],
                  [5.0, 5.0, 5.0, 5.0, 5.0],
                  [0.0, 0.0, 5.0, 5.0, 7.0], ]
    C[:, :, 3] = [[0.0, 3.0, 5.0, 8.0, 10.0],
                  [8.0, 8.0, 8.0, 8.0, 8.0],
                  [5.0, 5.0, 8.0, 8.0, 10.0], ]
    C[:, :, 4] = [[0.0, 3.0, 5.0, 8.0, 10.0],
                  [10.0, 10.0, 10.0, 10.0, 10.0],
                  [5.0, 5.0, 8.0, 8.0, 10.0], ]
    C = C.transpose()
    U = np.asarray([0, 0, 0, 1 / 3., 2 / 3., 1, 1, 1])
    V = np.asarray([0, 0, 0, 1 / 3., 2 / 3., 1, 1, 1])
    W = np.ones((C.shape[0], C.shape[1], 1))
    shape = np.asarray(C.shape)
    shape[-1] = 4  # To include the weights in the last term

    B = np.ones((shape))
    if W is None:
        weight = B[..., 3, np.newaxis]
    else:
        weight = np.asarray(W)
        B[..., 3, np.newaxis] = weight
    B[..., :3] = C

    return U, V, C, W, B

def surface_in_3D():
    """
    used mainly in the surface creation for the NURBS
    Returns:

    """
    C = np.zeros((3, 5, 5))
    C[:, :, 0] = [[0.0, 3.0, 5.0, 8.0, 10.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0],
                  [2.0, 2.0, 7.0, 7.0, 8.0], ]
    C[:, :, 1] = [[0.0, 3.0, 5.0, 8.0, 10.0],
                  [3.0, 3.0, 3.0, 3.0, 3.0],
                  [0.0, 0.0, 5.0, 5.0, 7.0], ]
    C[:, :, 2] = [[0.0, 3.0, 5.0, 8.0, 10.0],
                  [5.0, 5.0, 5.0, 5.0, 5.0],
                  [0.0, 0.0, 5.0, 5.0, 7.0], ]
    C[:, :, 3] = [[0.0, 3.0, 5.0, 8.0, 10.0],
                  [8.0, 8.0, 8.0, 8.0, 8.0],
                  [5.0, 5.0, 8.0, 8.0, 10.0], ]
    C[:, :, 4] = [[0.0, 3.0, 5.0, 8.0, 10.0],
                  [10.0, 10.0, 10.0, 10.0, 10.0],
                  [5.0, 5.0, 8.0, 8.0, 10.0], ]
    C = C.transpose()
    U = [0, 0, 0, 1 / 3., 2 / 3., 1, 1, 1]
    V = [0, 0, 0, 1 / 3., 2 / 3., 1, 1, 1]

    return [U, V], C

def curve_in_3D():
    """
    Used mainly in the test_plot for plotting spline curve
    Returns:

    """
    curve1_cpoints = np.array([[0, 0, 0],
                               [0.5, 0.2, 2],
                               [1, 1, 3],
                               [0.2, 1.5, 1],
                               [1, 2.2, 2],
                               [2, 2.5, -1],
                               [2.5, 2.2, -2],
                               [3, 1.5, 0],
                               [1.5, 1, 1],
                               [2.2, 0.2, 1],
                               [3, 0, 0]])  # , 0]])

    knot_vector_curve1 = np.array([0, 0, 0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1, 1])

    return knot_vector_curve1, curve1_cpoints

def make_surface_biquadratic():
    cp = np.asarray([[[0,0],[-1,0],[-2,0]],
                     [[0,1], [-1,2], [-2,2]],
                     [[1, 1.5], [1,4],[1,5]],
                     [[3, 1.5],[3, 4],[3, 5]]])
    #cp = np.asarray([[[0, 0], [0, 1], [1, 1.5], [3, 1.5]],
    #                 [[-1,0], [-1,2], [1,4], [3,4]],
    #                 [[-2, 0], [-2,2], [1,5], [3,5]]])

    shape = np.asarray(cp.shape)
    shape[-1] = 3  # To include the weights in the last term
    B = np.ones(shape)
    B[..., :2] = cp
    V = np.asarray([0, 0, 0, 1, 1, 1])
    U = np.asarray([0, 0, 0, 0.5, 1, 1, 1])

    return [U,V], B


def quarter_disk(r_inn=5, r_out=10):

    knots = np.asarray([[0,0,0,1,1,1],[0,0,0,1,1,1]])

    p1 = np.asarray([0, r_inn])
    p2 = np.asarray([r_inn, r_inn])
    p3 = np.asarray([r_inn, 0])
    p4 = np.asarray([0, r_out])
    p5 = np.asarray([r_out, r_out])
    p6 = np.asarray([r_out, 0])

    B = np.zeros((3, 3, 3));
    B[0, 0, :2] = p1
    B[1, 0, :2]=p2
    B[2, 0, :2]=p3
    B[0, 1, :2]=(p1 + p4) / 2
    B[1, 1, :2]=(p2 + p5) / 2
    B[2, 1, :2]=(p3 + p6) / 2
    B[0, 0, 2] = 1
    B[0, 1, 2] = 1
    B[1, 0, 2] = 1 / np.sqrt(2)
    B[1, 1, 2] = 1 / np.sqrt(2)
    B[2, 0, 2] = 1
    B[2, 1, 2] = 1
    B[0, 2, :2]=p4
    B[1, 2, :2]=p5
    B[2, 2, :2]=p6
    B[0, 2, 2] = 1
    B[1, 2, 2] = 1 / np.sqrt(2)
    B[2, 2, 2] = 1

    return knots, B

def make_curved_surface():
    middle_c = np.array([[[0., 100.], [0., 200.], [0., 300.]],
                         [[250., 360.], [250., 380.], [250., 400.]],
                         [[500., 100.], [500., 200.], [500., 300.]]])
    knot_m = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    shape = np.asarray(middle_c.shape)
    shape[-1] = 3  # To include the weights in the last term
    B = np.ones(shape)
    B[..., :2] = middle_c
    return knot_m, middle_c
###############Multipatch geometries############
def make_3_layer_patches(refine=False, knot_ins=[np.arange(0.1,1,0.1), np.arange(0.1,1,0.1)]):
    bottom_c = np.array([[[0., 0.], [0.,  50.], [0., 100.]],
                         [[250., 0.], [250., 180.], [250., 250.]],
                         [[500.,   0.], [500.,  50.], [500., 100.]]])
    knot_b = [[0, 0, 0, 1, 1, 1],[0, 0, 0, 1, 1, 1]]

    middle_c=np.array([[[0., 100.], [0., 200.], [0., 300.]],
                       [[250., 250.], [250., 350.], [250., 400.]],
                       [[500., 100.], [500., 200.], [500., 300.]]])
    knot_m = [[0, 0, 0, 1, 1, 1],[0, 0, 0, 1, 1, 1]]

    upper_c = np.array([[[0., 300.], [0., 400.], [0., 500.]],
                        [[250., 400.], [250., 450.], [250., 500.]],
                        [[500., 300.], [500., 400.], [500., 500.]]])
    knot_u = [[0, 0, 0, 1, 1, 1],[0, 0, 0, 1, 1, 1]]

    cpoints = [bottom_c, middle_c, upper_c]
    knots = [knot_b, knot_m, knot_u]
    B = []
    for i, cp in enumerate(cpoints):
        shape = np.asarray(cp.shape)
        shape[-1] = 3  # To include the weights in the last term
        B.append(np.ones(shape))
        B[i][..., :2] = cp

    if refine:
        from pygeoiga.nurb.refinement import knot_insertion
        for i in range(len(B)):
            knots_ins_0 = knot_ins[0]
            B[i], knots[i] = knot_insertion(B[i], degree=(2, 2), knots=knots[i], knots_ins=knots_ins_0, direction=0)
            knots_ins_1 = knot_ins[1]
            B[i], knots[i] = knot_insertion(B[i], degree=(2, 2), knots=knots[i], knots_ins=knots_ins_1, direction=1)

    geometry = OrderedDict({})
    name = ["granite", "mudstone", "sandstone"]
    color = ["red", "blue", "green"]
    kappa = [3.1, 0.9, 3]
    position = [(1,1),(2,1),(3,1)]
    for i, mat in enumerate(B):
        geometry[name[i]] = {"B": mat, "knots": knots[i], "kappa": kappa[i], 'color': color[i], "position": position[i]}

    # BOUNDARIES - faces of the patch in contact
    # 0: down
    # 1: right
    # 2: up
    # 3: left
    geometry["granite"]["patch_faces"] = {2:"mudstone"}
    geometry["mudstone"]["patch_faces"] = {0:"granite", 2:"sandstone"}
    geometry["sandstone"]["patch_faces"] = {0:"mudstone"}

    # For the meshing part. Specify the face that have a boundary condition
    geometry["granite"]["BC"] = {0: "bot_bc"}
    geometry["sandstone"]["BC"] = {2: "top_bc"}
    return geometry

def make_L_shape(refine = False, knot_ins=(np.arange(0.1,1,0.1), np.arange(0.1,1,0.1))):
    c_d1 = np.array([[[0, 0], [0, 100], [0, 200]],
                     [[50, 0], [85.5, 100], [121, 200]],
                     [[100, 0], [171, 100], [242, 200]]])
    k_d1 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c_d2 = np.array([[[100, 0], [171, 100], [242, 200]],
                     [[225, 0], [585.5/2, 100], [360, 200]],
                     [[500, 0], [500, 100], [500, 200]]])
    k_d2 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c_d3 = np.array([[[0, 200], [0, 250], [0, 300]],
                     [[121, 200], [138, 250], [155, 300]],
                     [[242, 200], [276, 250], [310, 300]]])
    k_d3 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    cpoints = [c_d1, c_d2, c_d3]
    knots = [k_d1, k_d2, k_d3]

    B = []
    for i, cp in enumerate(cpoints):
        shape = np.asarray(cp.shape)
        shape[-1] = 3  # To include the weights in the last term
        B.append(np.ones(shape))
        B[i][..., :2] = cp

    if refine:
        from pygeoiga.nurb.refinement import knot_insertion
        for i in range(len(B)):
            knots_ins_0 = knot_ins[0]
            B[i], knots[i] = knot_insertion(B[i], degree=(2,2), knots=knots[i], knots_ins=knots_ins_0, direction=0)
            knots_ins_1 = knot_ins[1]
            B[i], knots[i] = knot_insertion(B[i], degree=(2,2), knots=knots[i], knots_ins=knots_ins_1, direction=1)

    color = ["red", "blue", "purple"]

    red = 3  # kappa
    blue = 2  # kappa
    yellow = 5  # kappa

    geometry = OrderedDict({})
    name = ["bottom_L", "bottom_R","top"]
    kappa = [red, blue, yellow]
    position = [(1,1),(1,2), (2,1)]
    for i, mat in enumerate(B):
        geometry[name[i]] = {"B": mat, "knots": knots[i], "kappa": kappa[i], 'color': color[i], "position": position[i]}

    geometry["bottom_L"]["patch_faces"] = {1: "bottom_R", 2: "top"}#{2: "top"}#
    geometry["bottom_R"]["patch_faces"] = {3: "bottom_L"}
    geometry["top"]["patch_faces"] = {0: "bottom_L"}

    # For the meshing part. Specify the face that have a boundary condition
    geometry["bottom_L"]["BC"] = {0: "bot_bc"}
    geometry["bottom_R"]["BC"] = {0: "bot_bc"}
    geometry["top"]["BC"] = {2: "top_bc"}

    return geometry

def make_fault_model(refine = False, knot_ins=(np.arange(0.1,1,0.1), np.arange(0.1,1,0.1))):
    c_d1 = np.array([[[0, 0], [0, 100], [0, 200]],
                          [[50, 0], [85.5, 100], [121, 200]],
                          [[100, 0], [171, 100], [242, 200]]])
    k_d1 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c_d2 = np.array([[[100, 0], [171, 100], [242, 200]],
                     [[550, 0], [585.5, 100], [621, 200]],
                     [[1000, 0], [1000,  100], [1000,  200]]])
    k_d2 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c_d3 = np.array([[[0, 200], [0, 250], [0, 300]],
                     [[121, 200], [138, 250], [155, 300]],
                     [[242, 200],  [276, 250], [310, 300]]])
    k_d3 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c_d4 = np.array([[[ 242,  200], [ 276.,  250.], [ 310.,  300]],
                     [[ 621.,  200.], [ 638.,  250], [ 655.,  300.]],
                     [[1000.,  200.], [1000.,  250.], [1000.,  300.]]])
    k_d4 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c_d5 = np.array([[[  0., 300], [  0., 500], [  0., 700.]],
                     [[155., 300.], [225., 500.], [295., 700.]],
                     [[310., 300.], [450., 500], [590., 700]]])
    k_d5 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c_d6 = np.array([[[ 310.,  300], [ 450.,  500], [ 590.,  700]],
                     [[ 655.,  300], [ 725.,  500], [ 795.,  700]],
                     [[1000.,  300], [1000.,  500], [1000.,  700]]])
    k_d6 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c_d7 = np.array([[[  0. , 700], [  0. , 750], [  0. , 800.]],
                     [[295. , 700], [312.5, 750], [330. , 800]],
                     [[590. , 700], [625. , 750], [660. , 800]]])
    k_d7 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c_d8 = np.array([[[ 590. ,  700], [ 625. ,  750], [ 660. ,  800]],
                  [[ 795. ,  700], [ 812.5,  750], [ 830. ,  800]],
                  [[1000. ,  700], [1000. ,  750], [1000. ,  800]]])
    k_d8 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c_d9 = np.array([[[   0.,  800], [   0.,  900], [ 0., 1000]],
                     [[ 330.,  800], [ 365.,  900], [ 400., 1000]],
                      [[660.,  800], [ 730.,  900], [ 800., 1000]]])
    k_d9 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c_d10 = np.array([[[ 660.,  800.], [ 730.,  900], [ 800., 1000]],
                      [[ 830.,  800], [ 865.,  900], [ 900., 1000]],
                      [[1000.,  800], [1000.,  900], [1000., 1000]]])
    k_d10 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]

    cpoints = [c_d1, c_d2, c_d3, c_d4, c_d5, c_d6, c_d7, c_d8, c_d9, c_d10]
    knots = [k_d1, k_d2, k_d3, k_d4, k_d5, k_d6, k_d7, k_d8, k_d9, k_d10]

    B = []
    for i, cp in enumerate(cpoints):
        shape = np.asarray(cp.shape)
        shape[-1] = 3  # To include the weights in the last term
        B.append(np.ones(shape))
        B[i][..., :2] = cp

    if refine:
        from pygeoiga.nurb.refinement import knot_insertion
        for i in range(len(B)):
            knots_ins_0 = knot_ins[0]
            B[i], knots[i] = knot_insertion(B[i], degree=(2,2), knots=knots[i], knots_ins=knots_ins_0, direction=0)
            knots_ins_1 = knot_ins[1]
            B[i], knots[i] = knot_insertion(B[i], degree=(2,2), knots=knots[i], knots_ins=knots_ins_1, direction=1)


    red = 3.1  # kappa
    blue = 0.9 # salt
    yellow = 3  # New green

    geometry = OrderedDict({})
    name = ["bottom_L", "bottom_R", "D3", "D4", "D5", "D6", "D7", "D8", "top_L", "top_R"]
    color = ["red", "red", "red", "blue", "red", "green", "blue", "green", "green", "green"]
    # position refer to the location in x,y - x is columns, y rows - from top left to top right
    position = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (5, 2)]
    kappa = [red, red, red, blue, red, yellow, blue, yellow, yellow, yellow]
    for i, mat in enumerate(B):
        geometry[name[i]] = {"B": mat, "knots": knots[i], "kappa": kappa[i], 'color': color[i], "position": position[i]}

    geometry["bottom_L"]["patch_faces"] = {1: "bottom_R", 2: "D3"}
    geometry["bottom_R"]["patch_faces"] = {2: "D4", 3: "bottom_L"}
    geometry["D3"]["patch_faces"] = {0: "bottom_L", 1: "D4", 2: "D5"}
    geometry["D4"]["patch_faces"] = {0: "bottom_R", 2: "D6", 3: "D3"}
    geometry["D5"]["patch_faces"] = {0: "D3", 1: "D6", 2: "D7"}
    geometry["D6"]["patch_faces"] = {0: "D4", 2: "D8", 3: "D5"}
    geometry["D7"]["patch_faces"] = {0: "D5", 1: "D8", 2: "top_L"}
    geometry["D8"]["patch_faces"] = {0: "D6", 2: "top_R", 3: "D7"}
    geometry["top_L"]["patch_faces"] = {0: "D7", 1: "top_R"}
    geometry["top_R"]["patch_faces"] = {0: "D8", 3: "top_L"}

    # For the meshing part. Specify the face that have a boundary condition
    geometry["bottom_L"]["BC"] = {0: "bot_bc"}
    geometry["bottom_R"]["BC"] = {0: "bot_bc"}
    geometry["top_L"]["BC"] = {2: "top_bc"}
    geometry["top_R"]["BC"] = {2: "top_bc"}

    return geometry

def make_salt_dome(show=False, refine=False, knot_ins=[np.arange(0.1,1,0.1), np.arange(0.1,1,0.1)]):
    U1_1 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c1_1= np.asarray([[[ 0, 0], [0,  250], [0,  500]],
                    [[1200, 0], [1200,  250], [1200,  500]],
                    [[2400, 0], [2400,  250], [2400,  500]]])
    U1_2 = [[0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1],[0, 0, 0, 1, 1, 1]]
    c1_2 = np.asarray([[[2400, 0], [2400, 250], [2400, 500]],
                     [[2580, 0], [2640, 250], [2700, 500]],
                     [[2880, 0], [2790, 250], [2700, 500]],
                     [[3180, 0], [3040, 450], [2900, 900]],
                     [[3480, 0], [3190, 450], [2900, 900]],
                     [[3600, 0], [3600, 450], [3600, 900]]])
    U1_3 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c1_3 = np.asarray([[[3600, 0], [3600, 450], [3600, 900]],
                       [[4800, 0], [4800, 450],[4800, 900]],
                       [[6000, 0], [6000, 450],[6000, 900]]])
    U2_1 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c2_1 = np.asarray([[[0, 500], [0, 650], [0, 800]],
                       [[1200,  500], [1600, 500], [2000, 500]],
                       [[2400, 500], [2400, 566], [2400, 632]]])
    U2_2 = [[0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c2_2 = np.asarray([[[2400, 500], [2400, 566], [2400, 632]],
                       [[2700, 500], [2650, 600], [2700, 650]],
                       [[2700, 500], [2650, 700], [2750, 750]],
                       [[2900, 900], [2900, 950], [2900, 1000]],
                       [[2900, 900], [3300, 1000], [3300, 1100]],
                       [[3600, 900], [3600, 1050], [3600, 1200]]])
    U2_3 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c2_3 = np.asarray([[[3600, 900], [3600, 1050], [3600, 1200]],
                     [[4800, 900], [4200, 900], [4100, 1000]],
                     [[6000, 900], [6000, 905], [6000, 910]]])
    U3_1 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c3_1 = np.asarray([[[0, 800], [0, 1100], [0, 1400]],
                       [[2000, 500], [1400, 800], [2000, 700]],
                       [[2400, 632], [2600, 700], [2800, 1250]]])
    U3_2 = [[0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c3_2 = np.asarray([[[2400, 632], [2600, 700], [2800, 1250]],
                       [[2700, 650], [2700, 800], [2850, 1250]],
                       [[2750, 750], [2750, 900], [2950, 1400]],
                       [[2900, 1000], [2950, 1250], [3000, 1500]],
                       [[3300, 1100], [3150, 1400], [3200, 1700]],
                       [[3600, 1200], [3250, 1600], [3300, 1800]]])
    U3_3 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c3_3 = np.asarray([[[3600, 1200], [3250, 1600], [3300, 1800]],
                       [[4100, 1000], [4500, 1000], [4000, 1300]],
                       [[6000, 910], [6000, 1100], [6000, 1300]]])
    U4_1 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c4_1 = np.asarray([[[0, 1400], [0, 1800], [0, 2200]],
                       [[2000, 700], [2000, 1200], [2400, 1800]],
                       [[2800, 1250], [2900, 1600], [2750, 2200]]])
    U4_2 = [[0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c4_2 = np.asarray([[[2800, 1250], [2900, 1600], [2750, 2200]],
                       [[2850, 1250], [2900, 2000], [2900, 2200]],
                       [[2950, 1400], [3050, 2000], [3000, 2200]],
                       [[3000, 1500], [3100, 2100], [3200, 2300]],
                       [[3200, 1700], [3150, 2100], [3300, 2350]],
                       [[3300, 1800], [3250, 2000], [3400, 2400]]])
    U4_3 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c4_3 = np.asarray([[[3300, 1800], [3250, 2000], [3400, 2400]],
                       [[4000, 1300], [4000, 1200], [4000, 1800]],
                       [[6000, 1300], [6000, 1600], [6000, 1800]]])
    U5_1 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c5_1 = np.asarray([[[0, 2200], [0, 2350], [0, 2500]],
                       [[2400, 1800], [2400, 2100], [2400, 2200]],
                       [[2750, 2200], [2700, 2300], [2700, 2500]]])
    U5_2 = [[0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c5_2 = np.asarray([[[2750, 2200], [2700, 2300], [2700, 2500]],
                       [[2900, 2200], [2800, 2500], [2800, 2700]],
                       [[3000, 2200], [3000, 2500], [2900, 2750]],
                       [[3200, 2300], [3150, 2500], [3000, 2790]],
                       [[3300, 2350], [3280, 2500], [3190, 2790]],
                       [[3400, 2400], [3400, 2500], [3300, 2670]]])
    U5_3 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c5_3 = np.asarray([[[3400, 2400], [3400, 2500], [3300, 2670]],
                       [[4000, 1800], [4000, 2100], [4000, 2200]],
                       [[6000, 1800], [6000, 2000], [6000, 2200]]])
    U6_1 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c6_1 = np.asarray([[[0, 2500], [0, 2750], [0, 3000]],
                       [[2400, 2200], [2400, 2600], [2400, 3000]],
                       [[2700, 2500], [2700, 2750], [2700, 3000]]])
    U6_2 = [[0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c6_2 = np.asarray([[[2700, 2500], [2700, 2750], [2700, 3000]],
                       [[2800, 2700], [2800, 2850], [2800, 3000]],
                       [[2900, 2750], [2900, 2900], [2900, 3000]],
                       [[3000, 2790], [3000, 2900], [3000, 3000]],
                       [[3190, 2790], [3150, 2900], [3150, 3000]],
                       [[3300, 2670], [3300, 2800], [3300, 3000]]])
    U6_3 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c6_3 = np.asarray([[[3300, 2670], [3300, 2800], [3300, 3000]],
                       [[4000, 2200], [4000, 2600], [4000, 3000]],
                       [[6000, 2200], [6000, 2600], [6000, 3000]]])

    cpoints = [c1_1, c1_2, c1_3, c2_1, c2_2, c2_3, c3_1, c3_2, c3_3, c4_1, c4_2, c4_3, c5_1, c5_2, c5_3, c6_1, c6_2,
               c6_3]
    knots = [U1_1, U1_2, U1_3, U2_1, U2_2, U2_3, U3_1, U3_2, U3_3, U4_1, U4_2, U4_3, U5_1, U5_2, U5_3, U6_1, U6_2,
             U6_3]

    B = []
    for i, cp in enumerate(cpoints):
        shape = np.asarray(cp.shape)
        shape[-1] = 3  # To include the weights in the last term
        B.append(np.ones(shape))
        B[i][..., :2] = cp


    if show:
        def show_plots():
            from pygeoiga.plot.nrbplotting_mpl import p_surface, p_cpoints, p_knots
            import matplotlib
            import matplotlib.pyplot as mpl

            color = None
            mpl.close("all")
            figa, axa = mpl.subplots()
            org=False
            if org:
                image = mpl.imread(
                    "/home/danielsk78/GitProjects/master_thesis_IGA-Geothermal-Modeling/IGA/pygeoiga/nurb/data/salt_diapir.png")
                extent = [0, 6000, 0, 3000]
                axa.imshow(np.flipud(image), extent=extent, origin="lower", aspect="auto")
                major_ticks_x = np.arange(0, extent[1] + 1, 1000)
                major_ticks_y = np.arange(0, extent[3] + 1, 1000)
                minor_ticks_x = np.arange(0, extent[1] + 1, 200)
                minor_ticks_y = np.arange(0, extent[3] + 1, 200)

                axa.set_xticks(major_ticks_x)
                axa.set_xticks(minor_ticks_x, minor=True)
                axa.set_yticks(major_ticks_y)
                axa.set_yticks(minor_ticks_y, minor=True)
                axa.grid(which='both')

            for i, surf in enumerate(cpoints):
                col = color[i] if color is not None else np.random.rand(3, )
                p_surface(knots[i], surf, ax=axa, dim=2, color=col, alpha=0.5)
                p_cpoints(surf, ax = axa, line=False, point=True, color =col, dim=2)
                p_knots(knots[i], surf, ax =axa,dim=2,point=False, color = col)
            mpl.show()

            color = ["red", "red", "red", "blue", "blue", "blue", "brown", "blue", "brown","yellow", "blue","yellow","gray","blue","gray", "green","green","green"]
            figs, axs = mpl.subplots()
            for i, surf in enumerate(cpoints):
                col = color[i] if color is not None else np.random.rand(3, )
                p_surface(knots[i], surf, ax=axs, dim=2, color=col)
                #p_cpoints(surf, ax=axs, line=False, point=True, color=col, dim=2)
                #p_knots(knots[i], surf, ax=axs, dim=2, point=False, color=col)
            mpl.show()
        show_plots()
    if refine:
        from pygeoiga.nurb.refinement import knot_insertion
        for i in range(len(B)):
            knots_ins_0 = knot_ins[0]
            B[i], knots[i] = knot_insertion(B[i], degree=(2,2), knots=knots[i], knots_ins=knots_ins_0, direction=0)
            knots_ins_1 = knot_ins[1]
            B[i], knots[i] = knot_insertion(B[i], degree=(2,2), knots=knots[i], knots_ins=knots_ins_1, direction=1)

    if show:
        show_plots()
    red = 3.1  # W/mK Granite
    blue = 7.5  #W/mK Salt
    brown = 1.2  # 1.05–1.45 W/mK, shale
    yellow = 3  # 2.50–4.20 W/mK Sandstone
    gray = 0.9 #0.80–1.25 W/mK Claystone-Siltstone
    green = 3.2 # 2.50–4.20 W/mK Sandstone

    geometry = OrderedDict({})
    name = ["bottom_L", "bottom_C", "bottom_R", "D2_1", "D2_2", "D2_3", "D3_1", "D3_2", "D3_3", "D4_1", "D4_2", "D4_3",
            "D5_1", "D5_2", "D5_3", "top_L", "top_C", "top_R"]
    color = ["red", "red", "red", "blue", "blue", "blue", "brown", "blue", "brown", "yellow", "blue", "yellow", "gray",
             "blue", "gray", "green", "green", "green"]
    kappa = [red, red, red, blue, blue,blue, brown, blue,brown, yellow, blue, yellow, gray, blue, gray,
             green, blue, green]
    position = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3),(4, 1), (4, 2), (4, 3), (5, 1),
                (5, 2), (5, 3), (6, 1), (6, 2), (6, 3)]
    for i, mat in enumerate(B):
        geometry[name[i]] = {"B": mat, "knots": knots[i], "kappa": kappa[i], 'color': color[i], "position":position[i]}

    geometry["bottom_L"]["patch_faces"] = {1: "bottom_C", 2: "D2_1"}
    geometry["bottom_C"]["patch_faces"] = {1: "bottom_R", 2: "D2_2", 3:"bottom_L"}
    geometry["bottom_R"]["patch_faces"] = {2: "D2_3", 3: "bottom_C"}
    geometry["D2_1"]["patch_faces"] = {0: "bottom_L", 1: "D2_2", 2: "D3_1"}
    geometry["D2_2"]["patch_faces"] = {0: "bottom_C", 1: "D2_3", 2: "D3_2", 3:"D2_1"}
    geometry["D2_3"]["patch_faces"] = {0: "bottom_R", 2: "D3_3", 3: "D2_2"}
    geometry["D3_1"]["patch_faces"] = {0: "D2_1", 1: "D3_2", 2: "D4_1"}
    geometry["D3_2"]["patch_faces"] = {0: "D2_2", 1: "D3_3", 2: "D4_2", 3:"D3_1"}
    geometry["D3_3"]["patch_faces"] = {0: "D2_3", 2: "D4_3", 3: "D3_2"}
    geometry["D4_1"]["patch_faces"] = {0: "D3_1", 1: "D4_2", 2: "D5_1"}
    geometry["D4_2"]["patch_faces"] = {0: "D3_2", 1: "D4_3", 2: "D5_2", 3: "D4_1"}
    geometry["D4_3"]["patch_faces"] = {0: "D3_3", 2: "D5_3", 3: "D4_2"}
    geometry["D5_1"]["patch_faces"] = {0: "D4_1", 1: "D5_2", 2: "top_L"}
    geometry["D5_2"]["patch_faces"] = {0: "D4_2", 1: "D5_3", 2: "top_C", 3: "D5_1"}
    geometry["D5_3"]["patch_faces"] = {0: "D4_3", 2: "top_R", 3: "D5_2"}
    geometry["top_L"]["patch_faces"] = {0: "D5_1", 1: "top_C"}
    geometry["top_C"]["patch_faces"] = {0: "D5_2", 1: "top_R", 3: "top_L"}
    geometry["top_R"]["patch_faces"] = {0: "D5_3", 3: "top_C"}

    # For the meshing part. Specify the face that have a boundary condition
    geometry["bottom_L"]["BC"] = {0: "bot_bc"}
    geometry["bottom_C"]["BC"] = {0: "bot_bc"}
    geometry["bottom_R"]["BC"] = {0: "bot_bc"}
    geometry["top_L"]["BC"] = {2: "top_bc"}
    geometry["top_C"]["BC"] = {2: "top_bc"}
    geometry["top_R"]["BC"] = {2: "top_bc"}

    return geometry
#%%
def _make_unconformity_model(refine=False, knot_ins=(np.arange(0.1,1,0.1), np.arange(0.1,1,0.1))):
    c_e1 = np.array([[[0, 0], [50, 350], [500, 500]],
                    [[150, 0], [300, 300], [650, 500]],
                    [[300, 0], [300, 300], [800, 500]]])
    k_e1 = [[0, 0, 0, 1, 1, 1],[0, 0, 0, 1, 1, 1]]
    c_e2 = np.array([[[300, 0], [300, 300], [800,500]],
                      [[750, 0], [600, 250], [900, 500]],
                      [[1000, 0], [1000, 250], [1000,500]]])
    k_e2 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c_e3 = np.array([[[0, 0], [0, 250], [0, 500]],
                     [[50, 350], [100, 400], [250, 500]],
                     [[500, 500], [500,500], [500, 500]]])
    k_e3 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c_e4 = np.array([[[0, 500], [0, 600], [0, 700]],
                     [[250, 500], [250, 600], [250, 700]],
                     [[500, 500], [500,600], [500, 700]]])
    k_e4 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c_e5 = np.array([[[500, 500], [500,600], [500, 700]],
                     [[650, 500], [650, 600], [650, 700]],
                     [[800, 500], [800, 600], [800, 700]]])
    k_e5 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c_e6 = np.array([[[800, 500], [800, 600], [800, 700]],
                     [[900, 500], [900, 600], [900, 700]],
                     [[1000, 500], [1000, 600], [1000, 700]]])
    k_e6 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]

    cpoints = [c_e1, c_e2, c_e3, c_e4, c_e5, c_e6]
    knots = [k_e1, k_e2, k_e3, k_e4, k_e5, k_e6]
    B = []
    for i, cp in enumerate(cpoints):
        shape = np.asarray(cp.shape)
        shape[-1] = 3  # To include the weights in the last term
        B.append(np.ones(shape))
        B[i][..., :2] = cp

    if refine:
        from pygeoiga.nurb.refinement import knot_insertion
        for i in range(len(B)):
            knots_ins_0 = knot_ins[0]
            B[i], knots[i] = knot_insertion(B[i], degree=(2, 2), knots=knots[i], knots_ins=knots_ins_0, direction=0)
            knots_ins_1 = knot_ins[1]
            B[i], knots[i] = knot_insertion(B[i], degree=(2, 2), knots=knots[i], knots_ins=knots_ins_1, direction=1)

    red = 3#3.1  # W/mK Granite
    blue = 3#7.5  # W/mK Salt
    brown = 3#1.2  # 1.05–1.45 W/mK, shale
    yellow = 3  # 2.50–4.20 W/mK Sandstone
    gray = 3#0.9  # 0.80–1.25 W/mK Claystone-Siltstone
    green = 3  # 2.50–4.20 W/mK Sandstone

    geometry = OrderedDict({})
    name = ["E1", "E2", "E3", "E4", "E5", "E6"]
    color = ["red", "blue", "yellow", "gray", "gray", "gray" ]
    kappa = [red, blue, yellow, gray, gray, gray]
    position = [(1, 1), (1,3), (2,1), (3,1), (3,2), (3,3)]
    for i, mat in enumerate(B):
        geometry[name[i]] = {"B": mat, "knots": knots[i], "kappa": kappa[i], 'color': color[i], "position": position[i]}

    geometry["E1"]["patch_faces"] = {1: "E2", 2: "E5", 3: "E3"}
    geometry["E2"]["patch_faces"] = {2:"E6", 3: "E1"}
    geometry["E3"]["patch_faces"] = {0: "E1", 3: "E4"}
    geometry["E4"]["patch_faces"] = {0: "E3", 1: "E5"}
    geometry["E5"]["patch_faces"] = {0: "E1", 1: "E6", 3: "E4"}
    geometry["E6"]["patch_faces"] = {0: "E2", 3: "E5"}
    return geometry

def make_unconformity_model(refine=False, knot_ins=(np.arange(0.1,1,0.1), np.arange(0.1,1,0.1))):
    c_e1 = np.array([[[0, 0], [0, 250], [0, 500]],
                     [[0, 0], [100, 300], [250, 500]],
                     [[0, 0], [50, 350], [500, 500]]])
    k_e1 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c_e2 = np.array([[[0, 0], [50, 350], [500, 500]],
                    [[150, 0], [300, 300], [650, 500]],
                    [[300, 0], [300, 300], [800, 500]]])
    k_e2 = [[0, 0, 0, 1, 1, 1],[0, 0, 0, 1, 1, 1]]
    c_e3 = np.array([[[300, 0], [300, 300], [800,500]],
                      [[750, 0], [600, 250], [900, 500]],
                      [[1000, 0], [1000, 250], [1000,500]]])
    k_e3 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]

    c_e4 = np.array([[[0, 500], [0, 600], [0, 700]],
                     [[250, 500], [250, 600], [250, 700]],
                     [[500, 500], [500,600], [500, 700]]])
    k_e4 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c_e5 = np.array([[[500, 500], [500,600], [500, 700]],
                     [[650, 500], [650, 600], [650, 700]],
                     [[800, 500], [800, 600], [800, 700]]])
    k_e5 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    c_e6 = np.array([[[800, 500], [800, 600], [800, 700]],
                     [[900, 500], [900, 600], [900, 700]],
                     [[1000, 500], [1000, 600], [1000, 700]]])
    k_e6 = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]

    cpoints = [c_e1, c_e2, c_e3, c_e4, c_e5, c_e6]
    knots = [k_e1, k_e2, k_e3, k_e4, k_e5, k_e6]
    B = []
    for i, cp in enumerate(cpoints):
        shape = np.asarray(cp.shape)
        shape[-1] = 3  # To include the weights in the last term
        B.append(np.ones(shape))
        B[i][..., :2] = cp

    if refine:
        from pygeoiga.nurb.refinement import knot_insertion
        for i in range(len(B)):
            knots_ins_0 = knot_ins[0]
            B[i], knots[i] = knot_insertion(B[i], degree=(2, 2), knots=knots[i], knots_ins=knots_ins_0, direction=0)
            knots_ins_1 = knot_ins[1]
            B[i], knots[i] = knot_insertion(B[i], degree=(2, 2), knots=knots[i], knots_ins=knots_ins_1, direction=1)

    red = 3.1  # W/mK Granite
    blue = 7.5  # W/mK Salt
    brown = 1.2  # 1.05–1.45 W/mK, shale
    yellow = 3  # 2.50–4.20 W/mK Sandstone
    gray = 0.9  # 0.80–1.25 W/mK Claystone-Siltstone
    green = 3  # 2.50–4.20 W/mK Sandstone

    geometry = OrderedDict({})
    name = ["E1", "E2", "E3", "E4", "E5", "E6"]
    color = ["red", "blue", "yellow", "gray", "gray", "gray" ]
    kappa = [red, blue, yellow, gray, gray, gray]
    position = [(1, 1), (1,2), (1,3), (2,1), (2,2), (2,3)]
    for i, mat in enumerate(B):
        geometry[name[i]] = {"B": mat, "knots": knots[i], "kappa": kappa[i], 'color': color[i], "position": position[i]}

    geometry["E1"]["patch_faces"] = {1: "E2", 2: "E4"}
    geometry["E2"]["patch_faces"] = {1:"E3", 2:"E5", 3: "E1"}
    geometry["E3"]["patch_faces"] = {2: "E6", 3:"E2"}
    geometry["E4"]["patch_faces"] = {0: "E1", 1: "E5"}
    geometry["E5"]["patch_faces"] = {0: "E2", 1: "E6", 3: "E4"}
    geometry["E6"]["patch_faces"] = {0: "E3", 3: "E5"}

    # For the meshing part. Specify the face that have a boundary condition
    #geometry["E1"]["BC"] = {0: "bot_bc"}
    geometry["E2"]["BC"] = {0: "bot_bc"}
    geometry["E3"]["BC"] = {0: "bot_bc"}
    geometry["E4"]["BC"] = {2: "top_bc"}
    geometry["E5"]["BC"] = {2: "top"}
    geometry["E6"]["BC"] = {2: "top"}

    return geometry
