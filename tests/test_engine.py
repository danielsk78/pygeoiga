import matplotlib.pyplot as plt
import numpy as np
import pygeoiga as gn
#import gempyExplicit
import matplotlib
from pygeoiga.engine.NURB_engine import *

def test_knot_seeker_index():
    u = 1.8 #position to find in the knot
    knot_vector = np.array([0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5])
    index = find_span(u, knot_vector)
    assert index == 3

    u2 = 4.674  # position to find in the knot
    index2 = find_span(u2, knot_vector)
    assert index2 == 7

def test_show_basis_func():
    weight = None
    knot_vector = np.array([0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5])
    resolution = 1000
    degree = len(np.where(knot_vector == 0.)[0]) - 1
    N, der = basis_function_array_nurbs(knot_vector,
                                                       degree,
                                                       resolution,
                                                       weight)

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(N)
    fig.show()

    knot_vector1 = np.array([0, 0, 0, 1,1,2,2,3,3, 4,4, 5, 5,5])

    degree1 = len(np.where(knot_vector == 0.)[0]) - 1
    N1, der1 = basis_function_array_nurbs(knot_vector1,
                                                       degree1,
                                                       resolution,
                                                       weight)

    fig1 = plt.figure()
    ax1 = plt.gca()
    ax1.plot(N1)
    fig1.show()


def test_basis_func_and_der():
    weight = None
    knot_vector = np.array([0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5])
    resolution = 10
    degree = len(np.where(knot_vector == 0.)[0]) - 1
    N, der = basis_function_array_nurbs(knot_vector,
                                                       degree,
                                                       resolution)
    ans = np.array([[1., 0., 0., 0., 0.,0., 0., 0.],
                    [0.19753086, 0.64814815, 0.15432099, 0., 0.,0., 0., 0.],
                    [0., 0.39506173, 0.59876543, 0.00617284, 0., 0., 0., 0.]
                    ])

    ans2_der = np.array([[-2.,2.,0.,0.,0.,0., 0.,0.,],
                         [-0.88888889,0.33333333,0.55555556,0.,0.,0.,0.,0.],
                         [ 0.,-0.88888889,0.77777778,0.11111111,0.,0.,0.,0.]])
    assert np.allclose(N[:3], ans)
    assert np.allclose(der[:3], ans2_der)

    fig, ax = plt.subplots()
    ax.plot(N)
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(der)
    fig.show()

    weight = np.array([2, 3, 0, 2, 1, 2, 1, 1])
    N_n, der_n = basis_function_array_nurbs(knot_vector,
                                                         degree,
                                                         resolution,
                                                         weight)



    print(der[:3])
    fig, ax = plt.subplots()
    ax.plot(N_n)
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(der_n)
    fig.show()

def test_basis_derivatives_control():
    weight = None
    knot_vector = np.array([0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5])
    resolution = 100
    degree = len(np.where(knot_vector == 0.)[0]) - 1
    N, der = basis_function_array_nurbs(knot_vector,
                                                       degree,
                                                       resolution)

    fig, ax = plt.subplots()
    ax.plot(N)
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(der)
    fig.show()

    weight = np.array([1, 1/2, 1/2, 1, 1, 1, 1, 1])
    N_n, der_n = basis_function_array_nurbs(knot_vector,
                                                         degree,
                                                         resolution,
                                                         weight)
    fig, ax = plt.subplots()
    ax.plot(N_n)
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(der_n)
    fig.show()

def test_wrapper_1d():
    curve1_cpoints = np.array([[0, 0, 0],
                               [1, 0, 0],
                               [2, 1, 0],
                               [2, 2, 0]])
    knot_vector_curve1 = np.array([0, 0, 0, 0.5, 1, 1, 1])
    resolution = 1000
    x, y, z = NURB_construction([knot_vector_curve1], curve1_cpoints)
    plt.close('all')
    fig = plt.figure("Curve")
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    ax.plot(curve1_cpoints[:, 0], curve1_cpoints[:, 1], color='red', marker='s')  # , linestyle='None')
    fig.show()

    urve1_cpoints = np.array([[0, 0],
                              [1, 0],
                              [2, 1],
                              [2, 2]])
    knot_vector_curve1 = np.array([0, 0, 0, 0.5, 1, 1, 1])
    resolution = 1000

    x, y, z = NURB_construction([knot_vector_curve1], curve1_cpoints)
    plt.close('all')
    fig = plt.figure("Curve")
    ax = plt.gca()
    ax.plot(x, y)
    ax.plot(curve1_cpoints[:, 0], curve1_cpoints[:, 1], color='red', marker='s')  # , linestyle='None')
    fig.show()

def test_wrapper_2d():
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
    U = [0, 0, 0, 1 / 3., 2 / 3., 1, 1, 1 ]
    V = [0, 0, 0, 1 / 3., 2 / 3., 1, 1, 1 ]
    resolution = 20
    weight = np.ones((C.shape[0],C.shape[1],1))
    weight[...,0][3] =0.5 #TODO. Test the weight if they are correct
    positions = NURB_construction([U,V], C, resolution, weight)
    plt.close('all')
    fig = plt.figure("Curve")
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(positions[:, 0], positions[:, 1], positions[:, 2])
    ax.plot_wireframe(C[:, :, 0], C[:, :, 1], C[:, :, 2], color='red')
    fig.show()

def test_2d_otherpoints():
    points2D = np.array([[[150, 0, 50], [150, 25, 50], [150, 50, 50], [140, 50, 50], [140, 25, 50], [140, 0, 50]],
                         [[150, 0, 30], [150, 25, 30], [150, 50, 30], [140, 50, 30], [140, 25, 30], [140, 0, 30]],
                         [[150, 0, 15], [150, 25, 15], [150, 50, 15], [140, 50, 15], [140, 25, 15], [140, 0, 15]],
                         [[150, 0, 0], [150, 25, 0], [150, 50, 0], [140, 50, 0], [140, 25, 0], [140, 0, 0]]])

    knot1 = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # D1
    knot2 = np.array([0, 0, 0, 0.3, 0.5, 0.7, 1, 1, 1])  # D2
    resolution = 20

    positions = NURB_construction([knot1, knot2], points2D, resolution)
    plt.close('all')
    fig = plt.figure("Curve")
    ax = fig.add_subplot(111, projection='3d')
    #ax.plot_trisurf(positions[:, 0], positions[:, 1], positions[:, 2])
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],linestyle='None', marker ='.', color='blue')
    ax.plot_wireframe(points2D[:, :, 0], points2D[:, :, 1], points2D[:, :, 2], color='red')
    fig.show()

def test_wrapper_3d(): #TODO
    C = np.zeros((3, 2, 3, 4), dtype='d')
    C[0, 0, :, 0:2] = [0.0, 0.5]
    C[1, 0, :, 0:2] = [0.5, 0.5]
    C[2, 0, :, 0:2] = [0.5, 0.0]
    C[0, 1, :, 0:2] = [0.0, 1.0]
    C[1, 1, :, 0:2] = [1.0, 1.0]
    C[2, 1, :, 0:2] = [1.0, 0.0]
    C[:, :, 0, 2] = 0.0
    C[:, :, 1, 2] = 0.5
    C[:, :, 2, 2] = 1.0
    C[:, :, :, 3] = 1.0
    C[1, :, :, :] *= np.sqrt(2) / 2
    U = [0, 0, 0, 1, 1, 1]
    V = [0, 0, 1, 1]
    W = [0, 0, 0.5, 1, 1]

    resolution = 20
    positions = NURB_construction([U, V, W], C, resolution)

    plt.close('all')
    fig = plt.figure("Curve")
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='.')
    #ax.plot_wireframe(C[:, :, 0], C[:, :, 1], C[:, :, 2], color='red')
    fig.show()

def test_circular_beam():
    C = np.zeros((3, 3, 3))
    C[:, :, 0] = [[0, 0.5, 0.5],
                  [0.5, 0.5, 0.0],
                  [0,0,0]]
    C[:, :, 1] = [[0, 0.75, 75.0],
                  [0.75, 0.75, 0.0],
                  [0,0,0]]
    C[:, :, 2] = [[0, 1, 1],
                  [1, 1, 0.0],
                  [0,0,0]]
    C = C.transpose()
    #W = np.zeros((1, 3, 3))
    #W[:, :, 0] = [[1, np.sqrt(2)/2, 1],
    #              [1, np.sqrt(2)/2, 1],
    #              [1, np.sqrt(2)/2, 1]]
    #W = W.transpose()
    U = [0, 0, 0, 1, 1, 1 ]
    V = [0, 0, 0, 1, 1, 1 ]
    resolution = 20
    #W = np.ones((C.shape[0],C.shape[1],1))
    #W[]
    W=None

    positions = NURB_construction([U,V], C, resolution, W)
    plt.close('all')
    fig = plt.figure("Curve")
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(positions[:, 0], positions[:, 1], positions[:, 2])
    ax.plot_wireframe(C[:, :, 0], C[:, :, 1], C[:, :, 2], color='red')
    fig.show()
