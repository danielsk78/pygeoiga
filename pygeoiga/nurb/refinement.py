from pygeoiga.nurb import NURB
import numpy as np
from tqdm.autonotebook import tqdm


def knot_insertion(B, degree, knots: list, knots_ins: list, direction = 0, leave = True):
    """
    Manage the knot insertion to refine the nurb control point mesh
    Direction eta or xi
    Args:
        B: Control points + weights
        degree: degree
        knots: current knot to refine
        knots_ins: knots to insert. Can be single knot or a list of knots
        direction: parametric direction to insert the knots. xi (0), eta(1)
    Returns:
        nurb object

    """
    B_new = B
    if isinstance(knots, tuple):
        knots = list(knots)
    if not isinstance(knots, list):
        if isinstance(knots, np.ndarray):
            knots = knots.tolist()
        else:
            knots = [knots]
    knot_new = knots[direction]
    if isinstance(degree, int):
        degree=[degree]
    for knot_ins in tqdm(knots_ins, desc="Inserting single knot", leave=leave):
        B_new, knot_new = single_knot_insertion(cpoint_old=B_new,
                                                       knot_old=knot_new,
                                                       knot_ins=knot_ins,
                                                       degree = degree[direction],
                                                       direction=direction)
    knots[direction] = knot_new
    return B_new, knots

def single_knot_insertion(cpoint_old, knot_old, knot_ins, degree, direction):
    """
    Function that generates new control point when one new knot is inserted
    Args:
        cpoint_old: include the wheights in last column
        knot_old:
        knot_ins:
        degree:
        direction:

    Returns:

    """
    knot = np.append(knot_old, knot_ins)
    sort = np.argsort(knot)
    k_index = np.where(sort == len(knot_old) )[0][0]  # index at it was inserted
    # Project out to d + 1 dimension control points with the weights
    for i in range(len(cpoint_old.shape)-1):
        cpoint_old[..., i] = cpoint_old[..., i] * cpoint_old[..., -1]
    # Evaluate the new control points according to the inserted knot
    _shape = np.asarray(cpoint_old.shape, dtype=object)
    _shape[direction] = _shape[direction] + 1
    B = np.zeros(_shape)
    if direction == 0:
        B[:(k_index - degree), :] = cpoint_old[:(k_index - degree), :]
        for i in range(k_index - degree, k_index, 1):
            alpha = (knot_ins - knot_old[i]) / (knot_old[i + degree] - knot_old[i])
            B[i, :] = alpha * cpoint_old[i, :] + (1 - alpha) * cpoint_old[i - 1, :]

        B[k_index:, :] = cpoint_old[k_index - 1:, :]

    elif direction == 1:
        B[:, :(k_index - degree)] = cpoint_old[:, :(k_index - degree)]
        for i in range(k_index - degree, k_index, 1):
            alpha = (knot_ins - knot_old[i]) / (knot_old[i + degree] - knot_old[i])
            B[:, i] = alpha * cpoint_old[:, i] + (1 - alpha) * cpoint_old[:, i - 1]

        B[:, k_index:] = cpoint_old[:, k_index-1:]

    # Project out to d + 1 dimension control points with the weights
    for i in range(len(cpoint_old.shape)-1):
        B[..., i] = B[..., i] / B[..., -1]

    knot_sorted = knot[sort]

    return B, np.asarray(knot_sorted)

def degree_elevation(B, knots: list, times=1, direction = 0):
    try:
        from igakit.nurbs import NURBS
        nrb = NURBS(knots, B)
        nrb.elevate(axis=direction, times=times)
        B = nrb.control
        knots = nrb.knots
        return B, knots

    except:
        from geomdl import NURBS
        from geomdl import helpers
        if not isinstance(knots, list):
            knots = [knots]
        degree = np.asarray([len(np.where(knots[x] == 0.)[0]) - 1 for x in range(len(knots))])
        dim = len(degree)
        if dim==1:
            nrb = NURBS.Curve()
            nrb.degree = degree[0]
            nrb.ctrlpts = B[...,:-1]
            nrb.knotvector = knots[0]
            nrb = helpers.degree_elevation(degree[0], )

        else:
            nrb = NURBS.Surface()
            nrb.ctrlpts2d = B
            nrb.knotvector = knots




