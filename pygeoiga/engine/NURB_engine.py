import numpy as np

def NURB_construction(knot_vector: list, cpoints, resolution=30, weight=None):
    """
    Calculate a NURB curve, surface of volume by passing the control, points and knot vector
    Args:
        knot_vector:
        cpoints:
        resolution:
        weight:

    Returns:
        positions: x, y ,z
    """
    dim = len(knot_vector)
    assert dim == len(cpoints.shape)-1

    if dim == 1:
        positions = NURBS_Curve(cpoints, knot_vector[0], resolution, weight)
    elif dim == 2:
        positions = NURBS_Surface(cpoints, knot_vector, resolution, weight)
        #NURBS_surface
    elif dim == 3:
        positions = NURB_Volume(cpoints, knot_vector, resolution, weight)
        #NURB_volume
    else:
        print("Not possible to construct NURB of dimension: ", dim)
        raise AttributeError
    return positions

def find_span(u, knot_vector):
    """
    Find the vector span in which the point is located U[i]<=u<U[i+1].
    Taking into account that we need to ignore the multiples at the beginning and at the end.

    Args:
        knot_vector: knot vector
        degree: degree
    Returns:
        index of the knot vector
    """
    index = np.searchsorted(knot_vector, u, side="right") - 1
    return index

def basis_function_point(u, knot_vector, degree, index = None):
    """
    Calculates the basis function on the point u of the curve and its derivative
    N[i,degree] where i is over all the possible spans
    Args:
        u: point on the curve to look on
        knot_vector: knot_vector
        degree: degree
    Returns:
        returns the basis functions when it finds the span
    """
    if index is None:
        index = find_span(u, knot_vector)

    N = np.zeros((len(knot_vector) + 1, degree + 1))
    dN = np.zeros((len(knot_vector) + 1, degree + 1))
    N[index, 0] = 1
    for p in range(1, degree + 1, 1):
        n = len(knot_vector) - p - 1
        for i in range(n):
            if knot_vector[i + p] == knot_vector[i]:  # avoid dividing by 0
                left = 0
                left_d = 0
            else:
                left = N[i, p - 1] * ((u - knot_vector[i]) /
                                      (knot_vector[i + p] - knot_vector[i]))
                left_d = degree * N[i, p - 1] / (knot_vector[i + p] - knot_vector[i])

            if knot_vector[i + p + 1] == knot_vector[i + 1]:
                right = 0
                right_d = 0
            else:
                right = N[i + 1, p - 1] * ((knot_vector[i + p + 1] - u) /
                                           (knot_vector[i + p + 1] - knot_vector[i + 1]))
                right_d = degree * N[i + 1, p - 1] / (knot_vector[i + p + 1] - knot_vector[i + 1])

            N[i, p] = left + right
            dN[i, p] = left_d - right_d

    return N[:, degree], dN[:, degree]


def basis_function_array_spline(knot_vector, degree, resolution):
    """
    calculates the basis functions for an spline given a desired resolution
    Args:
        knot_vector: knot_vector
        degree: degree
        resolution: amount of points (np.linspace(knot_vector[0], knot_vector[-1], resolution))

    Returns:
        basis_functions
        derivatives
    """
    points = np.linspace(knot_vector[0], knot_vector[-1], resolution)
    # number of basis functions according to the degree
    n = len(knot_vector) - degree - 1
    # The basis functions of all the knot vectors according to the resolution where to save
    N_spline = np.zeros((len(points), len(knot_vector) + 1))  # N[i,p] to multiply with the control points
    dN_spline = np.zeros((len(points), len(knot_vector) + 1))

    for i in range(len(points)):
        N_spline[i, :], dN_spline[i, :] = basis_function_point(points[i], knot_vector, degree)
    N_spline[-1, n - 1] = 1
    return N_spline, dN_spline

def basis_function_array_nurbs(knot_vector, degree, resolution, weight=None):
    """
    Function to calculate basis function of nurbs taking into account the splines and the weights
    If no weight is passed or all the values are 1, it returns the basis functions and derivatives of the splines
    Args:
        knot_vector: knot vector
        degree: degree
        resolution:
        weight: amount of points (np.linspace(knot_vector[0], knot_vector[-1], resolution))
    Returns:
        basis_functions
        derivatives

    """
    N_spline, dN_spline = basis_function_array_spline(knot_vector, degree, resolution)
    n = len(knot_vector) - degree - 1
    # Deleting the excess of columns
    N_spline = N_spline[:, :n]
    dN_spline = dN_spline[:, :n]
    if np.all(weight == 1.) or weight is None:
        return N_spline, dN_spline

    points = len(N_spline)
    assert len(weight) == n

    N_nurbs = np.zeros((points, n), dtype=float)
    dN_nurbs = np.zeros((points, n), dtype=float)

    W = np.zeros(points, dtype=float)
    dW = np.zeros(points, dtype=float)  # N[i,p] to multiply with the control points

    for i in range(points):
        W[i] = N_spline[i, :] @ weight
        dW[i] = dN_spline[i, :] @ weight

    for i in range(n):
        N_nurbs[:, i] = weight[i] * \
                        np.divide(N_spline[:, i], W, out=np.zeros_like(N_spline[:, i]), where=W != 0)
        temp1 = dW * N_spline[:, i]
        temp2 = W ** 2
        dN_nurbs[:, i] = weight[i] * \
                         (np.divide(dN_spline[:, i], W, out=np.zeros_like(dN_spline[:, i]), where=W != 0) -
                          np.divide(temp1, temp2, out=np.zeros_like(temp1), where=temp2 != 0)
                          )
    return N_nurbs, dN_nurbs

def curve_point(u, degree, knot, cpoints, weight):
    """
    Plot the position of te knot u in the curve
    Args:
        u: knot to know the position
        degree: degree of the curve
        knot: knot vector
        cpoints: control points
        weight: weight

    Returns:
        position x,y and if z
    """

    #points = np.linspace(knot_vector[0], knot_vector[-1], resolution)
    # number of basis functions according to the degree
    n = len(knot) - degree - 1
    # The basis functions of all the knot vectors according to the resolution where to save
    #N_spline = np.zeros(len(knot) + 1) # N[i,p] to multiply with the control points
    #dN_spline = np.zeros(len(knot) + 1)

    N_spline, _ = basis_function_point(u, knot, degree)
    # Deleting the excess of columns
    N_spline = N_spline[:n]
    if u==knot[-1]:
        N_spline[n - 1] = knot[-1]


    #dN_spline = dN_spline[:n]
    if weight is None:
        weight = np.ones(cpoints.shape[0])

    assert len(weight) == n

    x = N_spline @ (cpoints[:, 0]*weight[:])
    y = N_spline @ (cpoints[:, 1]*weight[:])

    try:
        z = N_spline @ (cpoints[:, 2]*weight)
        positions = (x,y,z)
    except:
        positions = (x, y)
    return positions

def NURBS_Curve(cpoints, knot_vector, resolution, weight=None):
    """
    Calculate the positions of the curve
    Args:
        cpoints: control points
        knot_vector: knot vector
        resolution:
        weight:

    Returns:


    """
    degree = len(np.where(knot_vector == 0.)[0]) - 1
    basis_fun, derivatives = basis_function_array_nurbs(knot_vector, degree, resolution, weight)
    x = np.zeros(resolution)
    y = np.zeros(resolution)

    for i in range(resolution):
        x[i] = basis_fun[i, :] @ cpoints[:, 0]
        y[i] = basis_fun[i, :] @ cpoints[:, 1]

    try:
        z = np.zeros(resolution)
        for i in range(resolution):
            z[i] = basis_fun[i, :] @ cpoints[:, 2]
        positions = (x,y,z)
    except:
        positions = (x, y)
    return positions
################ Calculate surfaces ################
def NURBS_Surface(cpoints, knot_vector: list, resolution, weight=None, **kwargs):
    knot1 = np.asarray(knot_vector[0])
    knot2 = np.asarray(knot_vector[1])

    _deg = kwargs.get('degree',None)
    if _deg is not None:
        degree1 = _deg[0]
        degree2 = _deg[1]
    else:
        degree1 = len(np.where(knot1 == 0.)[0]) - 1
        degree2 = len(np.where(knot2 == 0.)[0]) - 1

    if weight is None:
        weight = np.ones((cpoints.shape[0], cpoints.shape[1], 1))

    points1 = np.linspace(knot1[0], knot1[-1], resolution)
    points2 = np.linspace(knot2[0], knot2[-1], resolution)

    positions = [surface_point(u,
                              v,
                              degree1,
                              degree2,
                              knot1,
                              knot2,
                              cpoints,
                              weight) for u in points1 for v in points2]

    return np.asarray(positions)

def surface_point(u, v, degree1, degree2, knot1, knot2, cpoints, weight):
    index1 = find_span(u, knot1)
    index2 = find_span(v, knot2)

    basis_fun1, dN1 = basis_function_point(u, knot1, degree1, index1)
    basis_fun2, dN2 = basis_function_point(v, knot2, degree2, index2)

    n1 = len(knot1) - degree1 - 1
    n2 = len(knot2) - degree2 - 1

    basis_fun1= basis_fun1[:n1]
    if u == knot1[-1]:
        basis_fun1[-1] = 1
    basis_fun2 = basis_fun2[:n2]
    if v == knot2[-1]:
        basis_fun2[-1] = 1

    temp_x = basis_fun1 @ (cpoints[..., 0] * weight[..., 0])
    temp_y = basis_fun1 @ (cpoints[..., 1] * weight[..., 0])
    x = basis_fun2 @ temp_x
    y = basis_fun2 @ temp_y
    try:
        temp_z = basis_fun1 @ (cpoints[..., 2] * weight[..., 0])
        z = basis_fun2 @ temp_z
        positions = (x,y,z)
    except:
        positions = (x, y)

    return positions

##### Volume NURBS #########
def volume_point(u, v, w, degree1, degree2, degree3, knot1, knot2, knot3, cpoints, weight):
    index1 = find_span(u, knot1)
    index2 = find_span(v, knot2)
    index3 = find_span(w, knot3)

    basis_fun1, dN1 = basis_function_point(u, knot1, degree1, index1)
    basis_fun2, dN2 = basis_function_point(v, knot2, degree2, index2)
    basis_fun3, dN3 = basis_function_point(w, knot3, degree3, index3)

    n1 = len(knot1) - degree1 - 1
    n2 = len(knot2) - degree2 - 1
    n3 = len(knot3) - degree3 - 1

    basis_fun1 = basis_fun1[:n1]
    basis_fun2 = basis_fun2[:n2]
    basis_fun3 = basis_fun3[:n3]

    temp_x = basis_fun1 @ cpoints[..., 0]
    temp_y = basis_fun1 @ cpoints[..., 1]
    temp_z = basis_fun1 @ cpoints[..., 2]

    temp_x2 = basis_fun2 @ temp_x
    temp_y2 = basis_fun2 @ temp_y
    temp_z2 = basis_fun2 @ temp_z

    x = basis_fun3 @ temp_x2
    y = basis_fun3 @ temp_y2
    z = basis_fun3 @ temp_z2

    return x, y, z

def NURB_Volume(cpoints, knot_vector: list, resolution, weight=None):
    knot1 = np.asarray(knot_vector[0])
    knot2 = np.asarray(knot_vector[1])
    knot3 = np.asarray(knot_vector[1])

    degree1 = len(np.where(knot1 == 0.)[0]) - 1
    degree2 = len(np.where(knot2 == 0.)[0]) - 1
    degree3 = len(np.where(knot3 == 0.)[0]) - 1

    if weight is not None:
        weight1 = np.asarray(weight[:, :, :, 0][0])
        weight2 = np.asarray(weight[:, :, :, 0][1])
        weight3 = np.asarray(weight[:, :, :, 0][1])
    else:
        weight1 = weight2 = weight3 = weight
    points1 = np.linspace(knot1[0], knot1[-1], resolution)
    points2 = np.linspace(knot2[0], knot2[-1], resolution)
    points3 = np.linspace(knot3[0], knot3[-1], resolution)

    positions = [volume_point(u,
                              v,
                              w,
                              degree1,
                              degree2,
                              degree3,
                              knot1,
                              knot2,
                              knot3,
                              cpoints,
                              weight)
                 for u in points1
                 for v in points2
                 for w in points3]

    return np.asarray(positions)

