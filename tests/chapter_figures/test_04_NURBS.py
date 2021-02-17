import pygeoiga as gn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('text', usetex=True)
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
fig_folder=gn.myPath+'/../../manuscript/Thesis/figures/04_NURBS/'
kwargs_savefig=dict(transparent=True, box_inches='tight', pad_inches=0)
save_all=False
#Generate Bezier curve figure
dpi=200
def test_fig_basis_functions():
    knot_vector = np.array([0, 0, 0, 0.5, 1, 1, 1])
    resolution = 1000
    points = np.linspace(knot_vector[0], knot_vector[-1], resolution)
    degree = 0
    N0,_ = gn.NURB_engine.basis_function_array_spline(knot_vector,
                                                      degree,
                                                      resolution)
    N0=N0[:,:4]
    fig, ax=plt.subplots(N0.shape[-1],1,dpi=200)#,figsize=(5,6))
    fig.canvas.draw()
    for i in range (N0.shape[-1]):
        ax[i].plot((-1,0),(0,0),'b')
        ax[i].plot(points,N0[:,i],'b')
        ax[i].plot((1,2), (0,0),'b')
        ax[i].set_xticks(np.arange(-1,2.5,0.5))
        ax[i].set_yticks([0,0.5,1])
        ax[i].set_xticklabels([0,0,0,0.5,1,1,1])
        ax[i].set_ylim((0,1))
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].annotate(r"$N_{%i,0}$" % i, xy=(-0.2, 0.5))
        if i==2:
            ax[i].plot((0,0),(0,1),'b')

    ax[N0.shape[-1] - 1].set_xlabel("knot index ($u_{i}$)")
    fig.show()
    #fig.save("")

    degree = 1
    N1, _= gn.NURB_engine.basis_function_array_spline(knot_vector,
                                                      degree,
                                                      resolution)
    N1 = N1[:, :4]
    fig1, ax1 = plt.subplots(N1.shape[-1], 1, dpi=200)
    for i in range(N1.shape[-1]):
        ax1[i].plot((-1, 0), (0, 0), 'b')
        ax1[i].plot(points, N1[:, i], 'b')
        ax1[i].plot((1, 2), (0, 0), 'b')
        ax1[i].set_xticks(np.arange(-1, 2.5, 0.5))
        ax1[i].set_yticks([0, 0.5, 1])
        ax1[i].set_xticklabels([0, 0, 0, 0.5, 1, 1, 1])
        ax1[i].set_ylim((0, 1))
        ax1[i].spines["top"].set_visible(False)
        ax1[i].spines["right"].set_visible(False)
        if i==1:
            ax1[i].plot((0,0),(0,1),'b')
        ax1[i].annotate(r"$N_{%i,1}$" % i, xy=(-0.2, 0.5))
    ax1[N0.shape[-1] - 1].set_xlabel("knot index ($u_{i}$)")
    fig1.show()

    degree = 2
    N2, der2= gn.NURB_engine.basis_function_array_spline(knot_vector,
                                                      degree,
                                                      resolution)
    N2_ = N2[:, :4]
    fig2, ax2 = plt.subplots(N2_.shape[-1], 1, dpi=200)
    for i in range(N2_.shape[-1]):
        ax2[i].plot((-1, 0), (0, 0), 'b')
        ax2[i].plot(points, N2[:, i], 'b')
        ax2[i].plot((1, 2), (0, 0), 'b')
        ax2[i].set_xticks(np.arange(-1, 2.5, 0.5))
        ax2[i].set_yticks([0, 0.5, 1])
        ax2[i].set_xticklabels([0, 0, 0, 0.5, 1, 1, 1])
        ax2[i].set_ylim((0, 1))
        ax2[i].spines["top"].set_visible(False)
        ax2[i].spines["right"].set_visible(False)
        ax2[i].annotate(r"$N_{%i,2}$"%i, xy=(-0.2,0.5))

        if i==0:
            ax2[i].plot((0,0),(0,1),'b')
        if i == 3:
            ax2[i].plot((1, 1), (1, 0), 'b')

    ax2[N0.shape[-1] - 1].set_xlabel("knot index ($u_{i}$)")
    fig2.show()

    figall, axall = plt.subplots(dpi=200)
    axall.plot(points, N2)
    axall.set_xticks([0,0.5,1])
    axall.spines["top"].set_visible(False)
    axall.spines["right"].set_visible(False)
    axall.set_ylim((0,1))
    axall.annotate(r"$N_{0,2}$", xy=(0.1, 0.8), fontsize=15)
    axall.annotate(r"$N_{1,2}$", xy=(0.3, 0.7), fontsize=15)
    axall.annotate(r"$N_{2,2}$", xy=(0.6, 0.7), fontsize=15)
    axall.annotate(r"$N_{3,2}$", xy=(0.8, 0.8), fontsize=15)
    figall.show()

    figder, axder = plt.subplots(dpi=200)
    axder.plot(points, der2)
    axder.set_xticks([0, 0.5, 1])
    axder.spines["top"].set_visible(False)
    axder.spines["right"].set_visible(False)
    axder.set_ylim((0, 1))
    #axder.annotate(r"$N_{0,2}$", xy=(0.2, 0.8), fontsize=15)
    #axder.annotate(r"$N_{1,2}$", xy=(0.6, 0.7), fontsize=15)
    #axder.annotate(r"$N_{2,2}$", xy=(1.2, 0.7), fontsize=15)
    #axder.annotate(r"$N_{3,2}$", xy=(1.7, 0.8), fontsize=15)
    figder.show()

    save = False
    if save or save_all:
        fig.savefig(fig_folder+"0_degree_basis.pdf", **kwargs_savefig)
        fig1.savefig(fig_folder + "1_degree_basis.pdf", **kwargs_savefig)
        fig2.savefig(fig_folder + "2_degree_basis.pdf", **kwargs_savefig)
        figall.savefig(fig_folder+ "all_join_degree_basis.pdf", **kwargs_savefig)

def test_curve_control_points():
    resolution = 1000
    points = np.linspace(0, 1, resolution)
    figsize=(15,5)
    curve1_cpoints = np.array([[0, 0],#, 0],
                               [0.5, 0.2],# 0],
                               [1, 1],#, 0],
                               [0.2, 1.5],
                               [1, 2.2],
                               [2,2.5],
                               [2.5,2.2],
                               [3,1.5],
                               [1.5,1],
                               [2.2, 0.2],
                               [3,0]])#, 0]])

    knot_vector_curve1 = np.array([0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1])

    x, y = gn.engine.NURB_construction([knot_vector_curve1], curve1_cpoints, resolution=resolution)
    degree = len(np.where(knot_vector_curve1 == 0.)[0]) - 1
    basis_fun, _ = gn.engine.basis_function_array_nurbs(knot_vector_curve1, degree, resolution)
    fig, (ax,bas) = plt.subplots(1,2, dpi=200,figsize=figsize)
    ax.plot(x, y)
    ax.plot(curve1_cpoints[:, 0], curve1_cpoints[:, 1], color='red', marker='s',
            linestyle='None')  # , linestyle='None')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i in range(len(curve1_cpoints)):
        ax.annotate("$P_{%i}$" % i, xy=(curve1_cpoints[i, 0], curve1_cpoints[i, 1]), xytext=(5, 0),
                    textcoords="offset points", fontsize=15)
    bas.plot(points,basis_fun)
    bas.set_ylim(0,1)
    bas.set_xlim(0, 1)
    #bas.set_xlabel("knot vector", ha="right")
    fig.show()

    knot_vector_curve2 = np.array([0, 0, 0,  0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1, 1])
    x2, y2 = gn.engine.NURB_construction([knot_vector_curve2], curve1_cpoints, resolution = resolution)
    degree = len(np.where(knot_vector_curve2 == 0.)[0]) - 1
    basis_fun, _ = gn.engine.basis_function_array_nurbs(knot_vector_curve2, degree, resolution)
    fig2, (ax2, bas) = plt.subplots(1, 2, dpi=200, figsize=figsize)
    ax2.plot(x2, y2)
    ax2.plot(curve1_cpoints[:, 0], curve1_cpoints[:, 1], color='red', marker='s', linestyle='None')  # , linestyle='None')
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for i in range(len(curve1_cpoints)):
        ax2.annotate("$P_{%i}$"%i, xy=(curve1_cpoints[i,0], curve1_cpoints[i,1]), xytext =(5,0),
                     textcoords="offset points", fontsize=15)
    bas.plot(points, basis_fun)
    bas.set_ylim(0, 1)
    bas.set_xlim(0, 1)
    fig2.show()

    knot_vector_curve3 = np.array([0, 0, 0, 0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 1, 1, 1])
    x3, y3 = gn.engine.NURB_construction([knot_vector_curve3], curve1_cpoints, resolution=resolution)
    degree = len(np.where(knot_vector_curve3 == 0.)[0]) - 1
    basis_fun, _ = gn.engine.basis_function_array_nurbs(knot_vector_curve3, degree, resolution)
    fig3, (ax3, bas) = plt.subplots(1, 2, dpi=200, figsize=figsize)
    ax3.plot(x3, y3)
    ax3.plot(curve1_cpoints[:, 0], curve1_cpoints[:, 1], color='red', marker='s',
             linestyle='None')  # , linestyle='None')
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    for i in range(len(curve1_cpoints)):
        ax3.annotate("$P_{%i}$" % i, xy=(curve1_cpoints[i, 0], curve1_cpoints[i, 1]), xytext=(5, 0),
                     textcoords="offset points", fontsize=15)
    bas.plot(points, basis_fun)
    bas.set_ylim(0, 1)
    bas.set_xlim(0, 1)
    fig3.show()

    knot_vector_curve5 = np.array([0, 0, 0, 0, 0, 0, 0.3, 0.4, 0.5, 0.6, 0.7, 1, 1, 1, 1, 1, 1])
    resolution = 1000
    x5, y5 = gn.engine.NURB_construction([knot_vector_curve5], curve1_cpoints, resolution=resolution)
    degree = len(np.where(knot_vector_curve5 == 0.)[0]) - 1
    basis_fun, _ = gn.engine.basis_function_array_nurbs(knot_vector_curve5, degree, resolution)
    fig5, (ax5, bas) = plt.subplots(1, 2, dpi=200, figsize=figsize)
    ax5.plot(x5, y5)
    ax5.plot(curve1_cpoints[:, 0], curve1_cpoints[:, 1], color='red', marker='s',
             linestyle='None')  # , linestyle='None')
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)
    for i in range(len(curve1_cpoints)):
        ax5.annotate("$P_{%i}$" % i, xy=(curve1_cpoints[i, 0], curve1_cpoints[i, 1]), xytext=(5, 0),
                     textcoords="offset points", fontsize=15)
    bas.plot(points, basis_fun)
    bas.set_ylim(0, 1)
    bas.set_xlim(0, 1)
    fig5.show()

    save = False
    if save or save_all:
        fig.savefig(fig_folder + "1_degree_curve.pdf", **kwargs_savefig)
        fig2.savefig(fig_folder + "2_degree_curve.pdf",**kwargs_savefig)
        fig3.savefig(fig_folder + "3_degree_curve.pdf", **kwargs_savefig)
        fig5.savefig(fig_folder + "5_degree_curve.pdf", **kwargs_savefig)

def test_surface():
    import matplotlib
    matplotlib.use('Qt5Agg')
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
    resolution = 50
    weight = np.ones((C.shape[0], C.shape[1], 1))
    weight[..., 0][3] = 0.5  # TODO. Test the weight if they are correct
    positions = gn.NURB_engine.NURB_construction([U, V], C, resolution, weight)
    fig = plt.figure("Curve")
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(positions[:, 0], positions[:, 1], positions[:, 2])
    ax.plot_wireframe(C[:, :, 0], C[:, :, 1], C[:, :, 2], color='red')
    plt.show()


def make_plot(control, basis_fun, resol, positions):
    dpi=200
    figsize=(5,6)
    fig, (ax, bas) = plt.subplots(2, 1, dpi=dpi, figsize=figsize)
    bas.plot(resol, basis_fun)
    bas.spines["top"].set_visible(False)
    bas.spines["right"].set_visible(False)
    bas.set_xticks([])
    bas.set_xlim(0,1)
    bas.set_ylim(0,1)
    # bas.set_xticklabels([])
    bas.set_yticks([])
    bas.set_aspect(0.2)

    ax.plot(positions[..., 0], positions[..., 1], 'b')
    ax.plot(control[..., 0], control[..., 1], color='red', marker='s',
            linestyle='--')
    #ax.set_ylim(top=1)
    ax.axis('off')
    ax.set_aspect(1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i in range(len(control)):
        ax.annotate("$P_{%i}$" % i, xy=(control[i, 0], control[i, 1]), xytext=(5, 0),
                    textcoords="offset points", fontsize=15)
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    fig.show()
    return fig

def test_refinement():
    points_ori = np.array([[0, 0, 0],
                         [1, 1, 0],
                         [2,0 , 0],
                         # [1,0,0]
                         ])
    knot = np.array([0, 0, 0, 1, 1, 1])  # D1
    resolution = 500
    reso = np.linspace(0, 1, resolution)

    nr = gn.NURB(points_ori, [knot])
    points = nr.cpoints
    B=nr.B
    weight=nr.weight

    x, y, z = gn.engine.NURB_construction([knot],points, resolution, weight)
    basis_fun, derivatives = gn.NURB_engine.basis_function_array_nurbs(knot, 2, resolution, weight)
    fig = make_plot(B, basis_fun, reso, np.asarray([x, y]).T)

    knot_ins1 = np.asarray([0.5])
    points1, knot1 = gn.nurb.knot_insertion(B, [2], [knot], knot_ins1)

    basis_fun1, derivatives1 = gn.NURB_engine.basis_function_array_nurbs(knot1[0], 2, resolution, weight)
    x1, y1, z1 = gn.engine.NURB_construction(knot1, points1, resolution, weight)
    fig1 = make_plot(points1, basis_fun1, reso, np.asarray([x1, y1]).T)

    nr2 = gn.NURB(points1, knot1)
    points2 = nr2.cpoints
    B2 = nr2.B
    weight = nr.weight

    knot_ins2 = np.asarray([0.3])
    points2, knot2 = gn.nurb.knot_insertion(B2, [2], knot1, knot_ins2)

    basis_fun2, derivatives2 = gn.NURB_engine.basis_function_array_nurbs(knot2[0], 2, resolution, weight)
    x2, y2, z2 = gn.engine.NURB_construction(knot2, points2, resolution, weight)
    fig2 = make_plot(points2, basis_fun2, reso, np.asarray([x2, y2]).T)

    knot_ins3 = np.asarray([0.3, 0.5, 0.7])
    points3, knot3 = gn.nurb.knot_insertion(B, [2], [knot], knot_ins3)

    basis_fun3, derivatives3 = gn.NURB_engine.basis_function_array_nurbs(knot3[0], 2, resolution, weight)
    x3, y3, z3 = gn.engine.NURB_construction(knot3, points3, resolution, weight)
    fig3 = make_plot(points3, basis_fun3, reso, np.asarray([x3, y3]).T)

    save = True
    if save or save_all:
        fig.savefig(fig_folder + "h_refine_ori.pdf", **kwargs_savefig)
        fig1.savefig(fig_folder + "h_refine_1.pdf", **kwargs_savefig)
        fig2.savefig(fig_folder + "h_refine_2.pdf", **kwargs_savefig)
        fig3.savefig(fig_folder + "h_refine_3.pdf", **kwargs_savefig)



def test_refinement_iga():
    from igakit.nurbs import NURBS
    points = np.array([[0, 0, 0],  # , 0],
                      [1, 1, 0],  # 0],
                      [2, 0,0],  # 0],
                      # [1,0,0]
                      ])
    knot = np.array([0, 0, 0, 1, 1, 1])  # D1
    resolution = 500
    reso = np.linspace(0, 1, resolution)
    weight = np.ones((points.shape[0], 1))
    crv = NURBS([knot], points)
    crv.refine(0,[0.5, 0.6])

    fig1, ax1 = plt.subplots()
    ax1.plot(crv(reso)[...,0], crv(reso)[...,1], 'b')
    ax1.plot(crv.control[..., 0], crv.control[..., 1], color='red', marker='s',
             linestyle='None')
    ax1.set_ylim(0, 1)
    fig1.show()


def test_refinement_order():
    try:
        from igakit.nurbs import NURBS
        active = True
    except:
        _0= np.asarray([[0., 0., 0., 1.], [1., 1., 0., 1.], [2., 0., 0., 1.]])
        _1=np.asarray([[0.,         0.,         0.,         1.,        ],
                      [0.66666667, 0.66666667, 0.,         1.,        ],
                      [1.33333333, 0.66666667, 0.,         1.,        ],
                      [2.,         0.,         0.,         1.,        ]])
        _2=np.asarray([[0.,         0.,         0.,         1.        ],
                       [0.5,        0.5,        0.,         1.,        ],
                       [1.,         0.66666667, 0.,         1.,        ],
                       [1.5,        0.5,        0.,         1.,        ],
                       [2.,         0.,         0.,         1.,        ]])
        active = False

    points = np.array([[0, 0],  # , 0],
                      [1, 1],  # 0],
                      [2, 0],  # 0],
                      # [1,0,0]
                      ])
    knot = np.array([0, 0, 0, 1, 1, 1])  # D1
    resolution = 500
    reso = np.linspace(0, 1, resolution)

    if active:
        crv = NURBS([knot], points)
        basis_fun, _ = gn.NURB_engine.basis_function_array_nurbs(crv.knots[0], crv.degree[0], resolution, crv.weights)
        _0 = crv.control
        fig = make_plot(crv.control, basis_fun, reso, crv(reso))

        crv.elevate(axis=0, times=1)
        basis_fun1, _ = gn.NURB_engine.basis_function_array_nurbs(crv.knots[0], crv.degree[0], resolution, crv.weights)
        _1 = crv.control
        fig1 = make_plot(crv.control, basis_fun1, reso, crv(reso))

        crv.elevate(axis=0, times=1)
        basis_fun2, _ = gn.NURB_engine.basis_function_array_nurbs(crv.knots[0], crv.degree[0], resolution, crv.weights)
        _2 = crv.control
        fig2 = make_plot(crv.control, basis_fun1, reso, crv(reso))

    else:
        x, y, z = gn.engine.NURB_construction([knot], _0, resolution, None)
        basis_fun, _ = gn.NURB_engine.basis_function_array_nurbs(knot, 2, resolution, None)
        fig = make_plot(_0, basis_fun, reso, np.asarray([x, y]).T)

        knot1=np.asarray([0,0, 0, 0,1, 1, 1, 1])
        x1, y1, z1 = gn.engine.NURB_construction([knot1], _1, resolution, None)
        basis_fun1, _ = gn.NURB_engine.basis_function_array_nurbs(knot1, 3, resolution, None)
        fig1 = make_plot(_1, basis_fun1, reso, np.asarray([x1, y1]).T)

        knot2 = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        x2, y2, z2 = gn.engine.NURB_construction([knot2], _2, resolution, None)
        basis_fun2, _ = gn.NURB_engine.basis_function_array_nurbs(knot2, 4, resolution, None)
        fig2 = make_plot(_2, basis_fun2, reso, np.asarray([x2, y2]).T)

    save = False
    if save or save_all:
        fig.savefig(fig_folder + "p_refine_2.pdf", **kwargs_savefig)
        fig1.savefig(fig_folder + "p_refine_3.pdf", **kwargs_savefig)
        fig2.savefig(fig_folder + "p_refine_4.pdf", **kwargs_savefig)

def test_k_refinement():
    try:
        from igakit.nurbs import NURBS
        active = True
    except:
        raise ImportError

    points = np.array([[0, 0],  # , 0],
                       [1, 1],  # 0],
                       [2, 0],  # 0],
                       # [1,0,0]
                       ])
    knot = np.array([0, 0, 0, 1, 1, 1])  # D1
    resolution = 500
    reso = np.linspace(0, 1, resolution)

    crv = NURBS([knot], points)
    basis_fun, _ = gn.NURB_engine.basis_function_array_nurbs(crv.knots[0], crv.degree[0], resolution, crv.weights)
    _0 = crv.control
    fig = make_plot(crv.control, basis_fun, reso, crv(reso))
    print(crv.knots)
    crv.insert(axis = 0, value=0.3).insert(axis = 0, value=0.5)
    basis_fun1, _ = gn.NURB_engine.basis_function_array_nurbs(crv.knots[0], crv.degree[0], resolution, crv.weights)
    _1 = crv.control
    fig1 = make_plot(crv.control, basis_fun1, reso, crv(reso))
    print(crv.knots)
    crv.elevate(axis=0, times=1)
    basis_fun2, _ = gn.NURB_engine.basis_function_array_nurbs(crv.knots[0], crv.degree[0], resolution, crv.weights)
    _2 = crv.control
    fig2 = make_plot(crv.control, basis_fun2, reso, crv(reso))
    print(crv.knots)
    save = False
    if save or save_all:
        fig.savefig(fig_folder + "k_refine_ori.pdf",**kwargs_savefig )
        fig1.savefig(fig_folder + "k_refine_1.pdf", **kwargs_savefig)
        fig2.savefig(fig_folder + "k_refine_2.pdf", **kwargs_savefig)


def test_projective_control_points():
    import matplotlib
    #matplotlib.use('Qt5Agg')
    points_ori = np.array([[1, 0],
                           [0, 1],
                           [1,1],
                           [2, 1],
                           [-2,-1]])

    points_circle = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]])
    points_ori=points_circle
    weight = np.array([[1],
                       [1/2],
                       [1],
                       [1/2],
                       [1]])
    weight_circle = np.array([[1], [1/np.sqrt(2)], [1], [1/np.sqrt(2)],[1], [1/np.sqrt(2)], [1], [1/np.sqrt(2)], [1]])
    weight=weight_circle

    knot = np.array([0, 0, 0, 0.5, 0.5, 1, 1, 1])  # D1
    knot_circle = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4])  # D1

    knot = knot_circle
    resolution = 500
    reso = np.linspace(0, 1, resolution)

    #nr = gn.NURB(points_ori, [knot], weight, resolution)

    N_spline, dN_spline = gn.NURB_engine.basis_function_array_spline(knot, 2, resolution)
    n = len(knot) - 2 - 1
    # Deleting the excess of columns
    N_spline = N_spline[:, :n]
    dN_spline = dN_spline[:, :n]

    points = len(N_spline)

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


        x = np.zeros(resolution)
        y = np.zeros(resolution)

        for i in range(resolution):
            x[i] = N_nurbs[i, :] @ points_ori[:, 0]
            y[i] = N_nurbs[i, :] @ points_ori[:, 1]

    def xy_from_z(ori, p2, z):
        vect = p2 - ori
        t = (z - ori[-1])/vect[-1]
        r = ori + (t*vect)
        return r

    origin = np.asarray([np.mean(points_ori[...,0]), np.mean(points_ori[...,1]) ,0])
    projected_control_points = np.asarray([xy_from_z(ori=origin, p2 = np.asarray([points_ori[i,0], points_ori[i,1], 1]), z=weight[i][0]+1) for i in range(len(weight))])
    projected_points = np.asarray([xy_from_z(ori=origin, p2 = np.asarray([x[i], y[i], 1]), z=W[i]+1) for i in range(resolution)])

    fig = plt.figure("projective Curve", dpi=dpi)#, figsize=plt.figaspect(0.5)*1.5)
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot_trisurf(positions[:, 0], positions[:, 1], positions[:, 2])
    #ax.plot(x, y, W+1, 'g')#linestyle='None', marker='.', color='blue')
    ax.plot(x,y, np.ones(points), 'b', label='$C(u)$')
    ax.plot(projected_points[...,0], projected_points[...,1], projected_points[...,2], 'g', label='$C^{w}(u)$')

    ax.plot(points_ori[...,0], points_ori[...,1], np.ones(weight.shape[0]), label='$\mathbf{P_{i}}$', color = 'red', marker = 's', markerfacecolor='red', linestyle = '--', alpha=0.5)
    ax.plot(projected_control_points[..., 0], projected_control_points[..., 1], weight[...,0]+1,label='$\mathbf{P^{w}_{i}}$', color='red', marker='o', markerfacecolor='red', linestyle='--', alpha=0.5)

    for i in projected_control_points:
        ax.plot((origin[0], i[0]), (origin[1], i[1]), (origin[2], i[2]), color='black', alpha=0.5)

    #for i in projected_points[::10]:
    #    ax.plot((origin[0], i[0]), (origin[1], i[1]), (origin[2], i[2]), color='black', alpha=0.1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="lower center")
    #plt.rcParams['grid.alpha'] = 0.1
    ax.grid(False)
    ax.view_init(elev=25,azim=45)
    x_pos, y_pos, z_pos = projected_control_points[3]
    ax.plot((x_pos, x_pos),(y_pos,y_pos), (0, z_pos), 'k')
    ax.text(x_pos, y_pos,  1, '$w_{i}$')
    ax.set_xticks([-2, -1 ,0, 1, 2])
    ax.set_yticks([-2,-1, 0, 1, 2])
    ax.set_zticks([0, 0.5, 1,1.5, 2])
    fig.show()
   # plt.show()

    save = False
    if save or save_all:
        fig.savefig(fig_folder + "projective_NURS.pdf", **kwargs_savefig)

def test_NURBS_vs_Bspline():

    #control = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]])

    #weight = np.array(
     #   [[1], [1 / np.sqrt(2)], [1], [1 / np.sqrt(2)], [1], [1 / np.sqrt(2)], [1], [1 / np.sqrt(2)], [1]])


    #knot = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4])  # D1
    control = np.array([[0, 0],
                        [1,0],
                           [1, 1],
                           [2, 1],
                           [2,0]])
    weight = np.array([[1],
                       [0.1],
                       [0.1],
                       [0.1],
                       [1]])

    knot = np.array([0, 0, 0, 0.5, 0.5, 1, 1, 1])  # D1
    resolution = 500
    resol = np.linspace(0, 1, resolution)

    spline_curve = gn.NURB(control, [knot], resolution=resolution)
    NURB_curve = gn.NURB(control, [knot], weight, resolution)

    dpi = 200
    figsize = (5, 3)
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    positions_Spline = np.asarray(spline_curve.model).T
    positions_NURBS = np.asarray(NURB_curve.model).T

    ax.plot(positions_NURBS[..., 0], positions_NURBS[..., 1], 'b', label="NURBS curve")
    ax.plot(positions_Spline[..., 0], positions_Spline[..., 1], 'g', label="B-spline curve")

    ax.plot(control[..., 0], control[..., 1], color='red', marker='s',
            linestyle='--')
    # ax.set_ylim(top=1)
    #ax.axis('off')
    #ax.set_aspect(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i in range(len(control)):
        #if i==0: xytext=(-5, 0)
        #else:
        xytext=(5, 0)
        ax.annotate("$P_{%i}$" % i, xy=(control[i, 0], control[i, 1]), xytext=xytext,
                    textcoords="offset points", fontsize=15)
    #fig.subplots_adjust(wspace=0, hspace=0)
    #fig.tight_layout()
    #ax.grid(linestyle="--")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    fig.show()

    #Basis-functions
    basis_fun_spline, derivatives_spline = gn.NURB_engine.basis_function_array_nurbs(knot, 2, resolution)
    basis_fun_NURBS, derivatives_NURBS = gn.NURB_engine.basis_function_array_nurbs(knot, 2, resolution, weight)

    fig2, (bas, bas2) = plt.subplots(1,2, dpi=dpi, figsize=(15,5))
    bas.plot(resol, basis_fun_spline)
    bas.spines["top"].set_visible(False)
    bas.spines["right"].set_visible(False)
    bas.set_xlim(0, 1)
    bas.set_ylim(0, 1)
    bas.set_xlabel("B-spline basis functions", fontsize=15)

    bas2.plot(resol, basis_fun_NURBS)
    bas2.spines["top"].set_visible(False)
    bas2.spines["right"].set_visible(False)
    #bas2.set_xticks([])
    bas2.set_xlim(0, 1)
    bas2.set_ylim(0, 1)
    bas2.set_xlabel("NURBS basis functions, $w_{0} = w_{4} = 1$ and $w_{1} = w_{2} = w_{3}= 0.1$",fontsize=15)

    fig2.show()

    save = False
    if save or save_all:
        fig.savefig(fig_folder + "NURBS_vs_BSpline.pdf", **kwargs_savefig)
        fig2.savefig(fig_folder + "NURBS_vs_BSpline_basis_functions.pdf",**kwargs_savefig)


def test_make_NURB_quarter_disk():
    from pygeoiga.nurb.cad import quarter_disk
    knots, B = quarter_disk()

    from pygeoiga.plot.nrbplotting_mpl import create_figure, p_cpoints, p_knots, p_curve, p_surface
    fig, [ax2, ax] = plt.subplots(1,2, sharey=True)
    ax = p_knots(knots,B,  ax=ax, dim=2, point=False, line=True, color ="k")
    ax = p_surface(knots,B, ax=ax, dim=2, color="blue", border = False, alpha=0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("x")
    ax.set_aspect("equal")


    ax2 = p_cpoints(B, ax=ax2, dim=2, color="blue", linestyle="-", point=False, line=True)
    ax2 = p_cpoints(B, ax=ax2, dim=2, color="red", marker="o", point=True, line=False)
    n, m = B.shape[0], B.shape[1]

    P = np.asarray([(B[x, y, 0], B[x, y, 1]) for y in range(m) for x in range(n)])
    for count, point in enumerate(P):
        ax2.annotate(str(count), point, fontsize=8, xytext=(3, 7), textcoords="offset points")

    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect("equal")

    fig.show()

    save = False
    if save or save_all:
        fig.savefig(fig_folder+"NURBS_surface.pdf", **kwargs_savefig)

def test_make_NURB_quarter_disk_refined():
    from pygeoiga.nurb.cad import quarter_disk
    knots, B = quarter_disk()

    from pygeoiga.nurb.refinement import knot_insertion
    knots_ins_0 = [0.3, 0.6]
    B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_0, direction=0)
    knots_ins_1 = [0.5]
    B, knots = knot_insertion(B, degree=(2, 2), knots=knots, knots_ins=knots_ins_1, direction=1)

    from pygeoiga.plot.nrbplotting_mpl import create_figure, p_cpoints, p_knots, p_curve, p_surface
    fig, [ax2, ax] = plt.subplots(1,2, sharey=True)
    ax = p_knots(knots,B,  ax=ax, dim=2, point=False, line=True, color ="k")
    ax = p_surface(knots,B, ax=ax, dim=2, color="blue", border = False, alpha=0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("x")
    ax.set_aspect("equal")


    ax2 = p_cpoints(B, ax=ax2, dim=2, color="blue", linestyle="-", point=False, line=True)
    ax2 = p_cpoints(B, ax=ax2, dim=2, color="red", marker="o", point=True, line=False)
    n, m = B.shape[0], B.shape[1]

    P = np.asarray([(B[x, y, 0], B[x, y, 1]) for y in range(m) for x in range(n)])
    for count, point in enumerate(P):
        ax2.annotate(str(count), point, fontsize=8, xytext=(3, 7), textcoords="offset points")

    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect("equal")
    fig.show()

    save = True
    if save or save_all:
        fig.savefig(fig_folder+"NURBS_surface.pdf", **kwargs_savefig)

def test_make_NURB_biquadratic():
    from pygeoiga.nurb.cad import make_surface_biquadratic
    knots, B = make_surface_biquadratic()

    from pygeoiga.plot.nrbplotting_mpl import create_figure, p_cpoints, p_knots, p_curve, p_surface
    fig, [ax2, ax] = plt.subplots(1,2, sharey=True, figsize=(7,3))
    ax = p_knots(knots,B,  ax=ax, dim=2, point=False, line=True, color ="k")
    ax = p_surface(knots,B, ax=ax, dim=2, color="blue", border = False, alpha=0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("x")

    #ax.annotate("1", (-1, 1), fontsize=20)
    #ax.annotate("2", (1.6, 3), fontsize=20)
    ax.set_aspect(0.8)

    ax2 = p_cpoints(B, ax=ax2, dim=2, color="blue", linestyle="-", point=False, line=True)
    ax2 = p_cpoints(B, ax=ax2, dim=2, color="red", marker="s", point=True, line=False)
    n, m = B.shape[0], B.shape[1]

    P = np.asarray([(B[x, y, 0], B[x, y, 1]) for x in range(n) for y in range(m)])
    for count, point in enumerate(P):
        ax2.annotate("$P_{%s}$"%str(count), point, fontsize=8, xytext=(3, 7), textcoords="offset points")

    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect(0.8)
    fig.show()

    from pygeoiga.engine.NURB_engine import basis_function_array_nurbs
    fig3 = plt.figure(constrained_layout=True)
    gs = fig3.add_gridspec(2, 2, hspace=0, wspace=0,
                           width_ratios=[0.2, 1],
                           height_ratios=[1, 0.2])
    (ax_v, ax3), (no, ax_u) = gs.subplots(sharex=True, sharey=True)
    no.remove()
    N_spline_u, _ = basis_function_array_nurbs(knot_vector=knots[0], degree=2, resolution=100)
    N_spline_v, _ = basis_function_array_nurbs(knot_vector=knots[1], degree=2, resolution=100)
    resol = np.linspace(0, 1, 100)

    ax_u.plot(resol, N_spline_u)
    ax_u.spines["top"].set_visible(False)
    ax_u.spines["right"].set_visible(False)

    ax_u.set_xlim(0, 1)
    ax_u.set_ylim(0, 1)

    ax_v.plot(N_spline_v, resol)
    ax_v.spines["top"].set_visible(False)
    ax_v.spines["right"].set_visible(False)
    ax_v.set_yticks(knots[2:-2])
    ax_v.set_xlim(1, 0)
    ax_v.set_ylim(0, 1)

    for i in knots[0]:
        ax3.vlines(i, 0, 1, 'k')
    for j in knots[1]:
        ax3.hlines(j, 0, 1, 'k')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_axis_off()

    ax_u.set_xlabel("$u$")
    ax_v.set_ylabel("$v$")
    ax_v.set_yticks(knots[1][2:-2])
    ax_u.set_xticks(knots[0][2:-2])
    for ax in ax_u, ax_v, ax3, no:
        ax.label_outer()
    fig3.show()

    save = True
    if save or save_all:
        fig.savefig(fig_folder+"B-spline_biquadratic.pdf", **kwargs_savefig)
        fig3.savefig(fig_folder+"B-spline_biquadratic_parameter.pdf", **kwargs_savefig)