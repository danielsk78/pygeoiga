import matplotlib.pyplot as plt
import numpy as np
import pygeoiga as gn


def test_square_2d():
    XY = np.asarray([[0, -10], [100, -10], [0, 10], [100, 10] ])
    ### Degree 1
    resolution = 50
    U, V, B = gn.nurb.make_surface_square(XY, 1)
    positions = gn.engine.NURB_construction([U, V], B[:, :, :2], resolution=resolution, weight=B[:, :, 2])
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(positions[:, 0], positions[:, 1], linestyle='None', marker='.', color='blue')
    ax.plot(B[:, :, 0], B[:, :, 1], linestyle='None', marker='.', color='red')  # , C[:, :, 2], color='red')
    fig.show()
    ### Degree 2
    U, V, B = gn.nurb.make_surface_square(XY, 2)

    ans = np.asarray([[[  0.,   0.,   0.],
                      [ 50.,  50.,  50.],
                      [100., 100., 100.]],

                     [[-10.,   0.,  10.],
                      [-10.,   0.,  10.],
                      [-10.,   0.,  10.]],

                     [[  1.,   1.,   1.],
                      [  1.,   1.,   1.],
                      [  1.,   1.,   1.]]])

    assert np.allclose(ans, B.T)
    resolution = 50
    positions = gn.engine.NURB_construction([U,V], B[:,:,:2],resolution=resolution,weight=B[:,:,2])

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(positions[:,0],positions[:,1],linestyle='None', marker ='.', color='blue')
    ax.plot(B[:, :, 0], B[:, :, 1], linestyle='None', marker ='.', color='red')#, C[:, :, 2], color='red')
    fig.show()
    ### Degree 3
    U, V, B = gn.nurb.make_surface_square(XY, 3)
    positions = gn.engine.NURB_construction([U, V], B[:, :, :2], resolution=resolution, weight=B[:, :, 2])
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(positions[:, 0], positions[:, 1], linestyle='None', marker='.', color='blue')
    ax.plot(B[:, :, 0], B[:, :, 1], linestyle='None', marker='.', color='red')  # , C[:, :, 2], color='red')
    fig.show()

def test_surface_3d():
    U, V, C, weight, B = gn.nurb.make_surface_3d()
    print(B)
    print(B[:,:,3])
    resolution = 100
    positions = gn.engine.NURB_construction([U, V], B[:, :, :3], resolution=resolution, weight=B[:, :, 3])
    fig = plt.figure("Curve")
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], linestyle='None', marker='.', color='blue')
    ax.plot_wireframe(B[:, :, 0], B[:, :, 1], B[:, :, 2], color='red')
    fig.show()