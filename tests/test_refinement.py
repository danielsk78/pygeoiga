#%%
import numpy as np
import matplotlib.pyplot as plt
from pygeoiga.nurb.refinement import knot_insertion
from pygeoiga.nurb import NURB


def test_knot_refinement_1D():
    points1D = np.array([[0,0,0],
                         [0,1,0],
                         [1,1,0],
                         #[1,0,0]
                         ])
    knot10=np.array([0,0,0,1,1,1])#D1
    weight1D=np.ones((points1D.shape[0], 1))

    nurb1 = NURB(points1D,[knot10], weight1D, engine="python")
    print(nurb1.B)
    fig1 = plt.figure("first")
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot(nurb1.model[0], nurb1.model[1], nurb1.model[2])
    ax1.plot(nurb1.cpoints[..., 0], nurb1.cpoints[..., 1], nurb1.cpoints[..., 2], color='red', marker='s')  # , linestyle='None')
    fig1.show()

    knot_ins = np.asarray([0.5, 0.6])
    B_new, knots = knot_insertion(nurb1.B, nurb1.degree, nurb1.knots, knot_ins, direction = 0)
    nurb1.cpoints = B_new[..., :3]
    nurb1.weight = B_new[..., 3, np.newaxis]
    nurb1.B = B_new
    nurb1.knots = knots

    nurb1.create_model()
    #nurb1 = gn.nurb.knot_insertion(nurb1, knot_ins, direction=0)

    print(nurb1.B)
    print(nurb1.B.shape)
    print(nurb1.degree)

    fig2 = plt.figure("first")
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot(nurb1.model[0], nurb1.model[1], nurb1.model[2])
    ax2.plot(nurb1.cpoints[..., 0], nurb1.cpoints[..., 1], nurb1.cpoints[..., 2], color='red',
            marker='s')  # , linestyle='None')
    fig2.show()

    knot_ins = np.asarray([0.4, 0.5, 0.7])
    B_new, knots = knot_insertion(nurb1.B, nurb1.degree, nurb1.knots, knot_ins, direction=0)
    nurb1.cpoints = B_new[..., :3]
    nurb1.weight = B_new[..., 3, np.newaxis]
    nurb1.B = B_new
    nurb1.knots = knots

    nurb1.create_model()
    #nurb1 = gn.nurb.knot_insertion(nurb1, knot_ins, direction=0)
    print(nurb1.B)
    print(nurb1.B.shape)
    print(nurb1.degree)
    fig3 = plt.figure("first")
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.plot(nurb1.model[0], nurb1.model[1], nurb1.model[2])
    ax3.plot(nurb1.cpoints[..., 0], nurb1.cpoints[..., 1], nurb1.cpoints[..., 2], color='red',
            marker='s')  # , linestyle='None')
    fig3.show()

def test_single_knot_2d():
    C = np.zeros((3, 3, 3))
    C[:, :, 0] = [[0.0, 3.0, 5.0],
                  [0.0, 0.0, 0.0],
                  [2.0, 2.0, 7.0]]
    C[:, :, 1] = [[0.0, 3.0, 5.0],
                  [3.0, 3.0, 3.0],
                  [0.0, 0.0, 5.0]]
    C[:, :, 2] = [[0.0, 3.0, 5.0],
                  [5.0, 5.0, 5.0],
                  [0.0, 0.0, 5.0]]

    C = C.transpose()
    U = np.array([0, 0, 0, 1, 1, 1])
    V = np.array([0, 0, 0, 1, 1, 1])
    resolution = 20

    nurb2 = NURB(C, [U, V], resolution=resolution, engine="python")

    print(nurb2.B)
    print(nurb2.B.shape)
    print(nurb2.degree)
    print(nurb2.knots)

    fig1 = plt.figure("first")
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot(nurb2.model[:, 0], nurb2.model[:, 1], nurb2.model[:, 2], linestyle='None', marker='.', color='blue')
    ax1.plot_wireframe(nurb2.cpoints[..., 0], nurb2.cpoints[..., 1], nurb2.cpoints[..., 2], color='red')
    fig1.show()

    knot_ins = np.asarray([0.5])

    direction = 1
    B_new, knots = knot_insertion(nurb2.B, nurb2.degree, nurb2.knots, knot_ins, direction=direction)
    nurb2.cpoints = B_new[..., :3]
    nurb2.weight = B_new[..., 3, np.newaxis]
    nurb2.B = B_new
    nurb2.knots = knots

    nurb2.create_model()

    print(nurb2.B)
    print(nurb2.B.shape)
    print(nurb2.degree)
    print(nurb2.knots)

    fig2 = plt.figure("second")
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot(nurb2.model[:, 0], nurb2.model[:, 1], nurb2.model[:, 2], linestyle='None', marker='.', color='blue')
    ax2.plot_wireframe(nurb2.cpoints[..., 0], nurb2.cpoints[..., 1], nurb2.cpoints[..., 2], color='red')
    fig2.show()

    direction = 0
    B_new, knots = knot_insertion(nurb2.B, nurb2.degree, nurb2.knots, knot_ins, direction=direction)
    nurb2.cpoints = B_new[..., :3]
    nurb2.weight = B_new[..., 3, np.newaxis]
    nurb2.B = B_new
    nurb2.knots = knots

    nurb2.create_model()
    print(nurb2.B)
    print(nurb2.B.shape)
    print(nurb2.degree)
    print(nurb2.knots)

    fig3 = plt.figure("second")
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.plot(nurb2.model[:, 0], nurb2.model[:, 1], nurb2.model[:, 2], linestyle='None', marker='.', color='blue')
    ax3.plot_wireframe(nurb2.cpoints[..., 0], nurb2.cpoints[..., 1], nurb2.cpoints[..., 2], color='red')
    fig3.show()

def test_knot_refinement_2d():

    points2D = np.array([[[150,0,50],[150,25,50],[150,50,50],[140,50,50],[140,25,50],[140,0,50]],
                             [[150,0,30],[150,25,30],[150,50,30],[140,50,30],[140,25,30],[140,0,30]],
                             [[150,0,15],[150,25,15],[150,50,15],[140,50,15],[140,25,15],[140,0,15]],
                             [[150,0,0],[150,25,0],[150,50,0],[140,50,0],[140,25,0],[140,0,0]]])

    knot1 = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # D1
    knot2 = np.array([0,0,0,0.3,0.5,0.7,1,1,1])#D2
    weight2D=np.ones((points2D.shape[0],points2D.shape[1], 1))

    nurb2 = NURB(points2D, [knot1, knot2], weight2D, engine="python")

    print(nurb2.B)
    print(nurb2.B.shape)
    print(nurb2.degree)

    fig1 = plt.figure("first")
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot(nurb2.model[:, 0], nurb2.model[:, 1], nurb2.model[:, 2], linestyle='None', marker='.', color='blue')
    ax1.plot_wireframe(nurb2.cpoints[..., 0], nurb2.cpoints[..., 1], nurb2.cpoints[..., 2], color='red')
    fig1.show()

    knot_ins = np.asarray([0.5])
    direction = 0
    B_new, knots = knot_insertion(nurb2.B,nurb2.degree,nurb2.knots, knot_ins, direction=direction)
    nurb2.cpoints = B_new[..., :3]
    nurb2.weight = B_new[..., 3, np.newaxis]
    nurb2.B = B_new
    nurb2.knots = knots

    nurb2.create_model()

    print(nurb2.B)
    print(nurb2.B.shape)
    print(nurb2.degree)

    fig2 = plt.figure("second")
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot(nurb2.model[:, 0], nurb2.model[:, 1], nurb2.model[:, 2], linestyle='None', marker='.', color='blue')
    ax2.plot_wireframe(nurb2.cpoints[..., 0], nurb2.cpoints[..., 1], nurb2.cpoints[..., 2], color='red')
    fig2.show()



def test_knot_refinement_3d():
    #TODO
    pass
