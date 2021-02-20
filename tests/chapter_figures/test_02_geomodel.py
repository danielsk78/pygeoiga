#%%
import pygeoiga as gn
import numpy as np
import matplotlib.pyplot as plt
fig_folder=gn.myPath+'/../../manuscript_IGA_MasterThesis/Thesis/figures/02_Geomodeling/'
kwargs_savefig=dict(transparent=True, box_inches='tight', pad_inches=0)
save_all=False
#%%

def test_plot_circle():
    # implicit equation
    r = 1
    x = np.linspace(-r, r, 1000)
    y = np.sqrt(-x ** 2 + r ** 2)
    fig, ax = plt.subplots()
    ax.plot(x, y, 'b')
    ax.plot(x, -y, 'b')
    ax.set_aspect('equal')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(ls="dashed")
    fig.show()

    # parametric equation
    # theta goes from 0 to 2pi
    theta = np.linspace(0, 2 * np.pi, 1000)
    # the radius of the circle
    r = 1
    # compute x1 and x2
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)

    # create the figure
    fig, ax = plt.subplots()
    ax.plot(x1, x2)
    ax.set_aspect("equal")
    ax.grid(ls="dashed")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()

    save = False
    if save or save_all:
        fig.savefig(fig_folder + "implicit_explicit_circle.pdf", **kwargs_savefig)

def test_plot_gempy_explicit_implicit():
    # Importing GemPy
    import gempy as gp
    import matplotlib

    geo_model = gp.create_model('Model1')
    geo_model = gp.init_data(geo_model, extent=[0, 791, 0, 200, -582, 0], resolution=[100, 10, 100])
    gp.set_interpolator(geo_model, theano_optimizer='fast_run', verbose=[])

    geo_model.set_default_surfaces()
    geo_model.add_surfaces(['surface3', 'basement'])

    geo_model.add_surface_points(X=225, Y=0, Z=-94, surface='surface1')
    geo_model.add_surface_points(X=464, Y=0, Z=-107, surface='surface1')
    geo_model.add_surface_points(X=620, Y=0, Z=-14, surface='surface1')

    geo_model.add_orientations(X=350, Y=0, Z=-300, surface='surface1', pole_vector=(0, 0, 1))

    geo_model.add_surface_points(X=225, Y=0.01, Z=-269, surface='surface2')
    geo_model.add_surface_points(X=464, Y=0, Z=-279, surface='surface2')
    geo_model.add_surface_points(X=620, Y=0, Z=-123, surface='surface2')

    geo_model.add_surface_points(X=225, Y=0, Z=-439, surface='surface3')
    geo_model.add_surface_points(X=464, Y=0, Z=-446, surface='surface3')
    geo_model.add_surface_points(X=620, Y=0, Z=-433, surface='surface3')

    geo_model.surfaces.df.color[0] = '#015482'
    geo_model.surfaces.df.color[1] = '#ffbe00'
    geo_model.surfaces.df.color[2] = '#728f02'
    geo_model.surfaces.df.color[3] = '#ff0000'

    gp.compute_model(geo_model)
    # Outcrop
    outcrop = gp.plot.Plot2D(geo_model)
    outcrop.create_figure((6, 5))
    ax_o = outcrop.add_section(cell_number=1, direction='y')
    img = matplotlib.image.imread('Picture 1.png')
    ax_o.imshow(img, origin='upper', alpha=.8, extent=(0, 791, -582, 0))
    outcrop.plot_data(ax_o)
    ax_o.set_title("Data")
    outcrop.fig.show()

    gp.compute_model(geo_model)

    # scalar field
    scalar = gp.plot.Plot2D(geo_model)
    scalar.create_figure((6, 5))
    ax_s = scalar.add_section(cell_number=1, direction='y')
    scalar.plot_data(ax_s)
    scalar.plot_scalar_field(ax_s)
    ax_s.set_title("Scalar field")
    scalar.fig.show()

    gp.compute_model(geo_model)

    # cross section
    cross = gp.plot.Plot2D(geo_model)
    cross.create_figure((6, 5))
    ax_c = cross.add_section(cell_number=1, direction='y')
    cross.plot_data(ax_c)
    cross.plot_lith(ax_c)
    cross.plot_contacts(ax_c)
    ax_c.set_title("Geomodel")
    cross.fig.show()

    plt.figlegend()

    save = False
    if save or save_all:
        outcrop.fig.savefig(fig_folder + "outcrop_image.pdf", **kwargs_savefig)
        scalar.fig.savefig(fig_folder + "scalar_field.pdf", **kwargs_savefig)
        cross.fig.savefig(fig_folder + "geomodel.pdf", **kwargs_savefig)

def test_explicit_model():
    import matplotlib
    from pygeoiga.plot.nrbplotting_mpl import create_figure, p_cpoints, p_knots, p_curve, p_surface

    img = matplotlib.image.imread('dike.jpg')
    fig, ax = plt.subplots()
    ax.imshow(img, alpha=.8, extent=(0, 300, 0, 178))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ### Big dike
    cp_1=np.array([[[223, 0], [145,60], [108,90], [71,149]],
                 [[300, 0], [300, 4.5], [141, 111], [104, 159]]
                 ]
                )
    kn1_1 = [0,0,1,1]
    kn1_2 = [0,0,0.3,0.6,1,1]
    knots_1=[kn1_1,kn1_2]

    ax = p_cpoints(cp_1, ax=ax, dim=2, color="black", marker="s", point=True, line=False)
    ax = p_knots(knots_1, cp_1, ax=ax, dim=2, point=True, line=True)

    ### middle dike
    cp_2 = np.array([[[0, 40], [92, 86], [197,154], [300,178]],
                     [[0, 94], [80, 120], [91, 144], [204, 178]]]
                    )
    kn2_1 = [0, 0, 1, 1]
    kn2_2 = [0, 0, 0.3, 0.6, 1, 1]
    knots_2 = [kn2_1, kn2_2]

    ax = p_cpoints(cp_2, ax=ax, dim=2, color="black", marker="s", point=True, line=False)
    ax = p_knots(knots_2, cp_2, ax=ax, dim=2, point=True, line=True)

    ### Weathered dike
    cp_3 = np.array([[[0, 94], [90, 100], [91, 144], [204, 178]],
                     [[0, 100], [80, 150], [91, 160], [204, 178]],
                    [[0, 178], [80, 178], [142, 178], [204, 178]]
                    ]
                    )
    kn3_1 = [0, 0,0,1, 1, 1]
    kn3_2 = [0, 0, 0, 0.5,1, 1, 1]
    knots_3 = [kn3_1, kn3_2]

    ax = p_cpoints(cp_3, ax=ax, dim=2, color="black", marker="s", point=True, line=False)
    ax = p_knots(knots_3, cp_3, ax=ax, dim=2, point=True, line=True)

    ### Bottom
    cp_4 = np.array([[[0, 40], [92, 86], [197,154], [300,178]],
                     [[0,0], [92, 0], [197, 0], [300, 0]]
                     ]
                    )
    kn4_1 = [0, 0, 1, 1]
    kn4_2 = [0, 0, 0.3, 0.6, 1, 1]
    knots_4 = [kn4_1, kn4_2]

    ax = p_cpoints(cp_4, ax=ax, dim=2, color="black", marker="s", point=True, line=False)
    ax = p_knots(knots_4, cp_4, ax=ax, dim=2, point=True, line=True)

    ax.set_xlim(0, 300)
    ax.set_ylim(0, 178)
    fig.show()

    fig2, ax2 =create_figure("2d")

    p_surface(knots_4, cp_4, ax=ax2, dim=2, color="red", border=False, label="Dike1")
    p_surface(knots_2, cp_2, ax=ax2, dim=2, color="blue", border=False, label="Dike2")
    p_surface(knots_1, cp_1, ax=ax2, dim=2, color="yellow", border=False, label="Dike3")
    p_surface(knots_3, cp_3, ax=ax2, dim=2, color="gray", border=False, label="Unknown")

    ax2.set_xlim(0, 300)
    ax2.set_ylim(0, 178)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.legend(loc="center right", facecolor = 'white', framealpha = 0.8)

    fig2.show()

    fig3, ax3 = create_figure("2d")
    ax3.imshow(img, alpha=.8, extent=(0, 300, 0, 178))

    #p_surface(knots_4, cp_4, ax=ax3, dim=2, color="red", fill=False, label="Dike1")
    #p_surface(knots_2, cp_2, ax=ax3, dim=2, color="blue", fill=False, label="Dike2")
    #p_surface(knots_1, cp_1, ax=ax3, dim=2, color="yellow", fill=False, label="Dike3")
    #p_surface(knots_3, cp_3, ax=ax3, dim=2, color="gray", fill=False, label="Unknown")

    ax3.set_xlim(0, 300)
    ax3.set_ylim(0, 178)
    ax3.set_aspect("equal")
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")
    #ax3.legend(loc="center right")
    fig3.show()

    save = False
    if save or save_all:
        fig2.savefig(fig_folder + "model_explicit.pdf", **kwargs_savefig)
        fig3.savefig(fig_folder + "original_explicit.pdf", **kwargs_savefig)

def test_show_modifysimple_mesh():
    from pygeoiga.plot.nrbplotting_mpl import create_figure, p_cpoints, p_knots, p_curve, p_surface

    cp = np.array([[[0, 0], [1, 0], [2, 0], [3, 0]],
                   [[0, 1], [1, 1], [2, 1], [3, 1]],
                   [[0, 2], [1, 2], [2, 2], [3, 2]]]
                  )
    kn1 = [0, 0, 0.5, 1, 1]
    kn2= [0, 0, 1/3, 2/3, 1, 1]

    fig, ax = create_figure("2d")
    #ax.set_axis_off()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = p_knots([kn1, kn2], cp, ax=ax, dim=2, point=False, line=True, color="black")
    ax = p_cpoints(cp, ax=ax, dim=2, color="red", marker="o", point=True, line=False)
    n, m = cp.shape[0], cp.shape[1]
    P = np.asarray([(cp[x, y, 0], cp[x, y, 1]) for x in range(n) for y in range(m)])

    for count, point in enumerate(P):
        ax.annotate(str(count), point, xytext =(5,5), textcoords="offset points")

    ax.annotate("$\Omega_1$", (0.5, 0.5), fontsize=20, xytext =(-5,-5), textcoords="offset points")
    ax.annotate("$\Omega_2$", (1.5, 0.5), fontsize=20, xytext=(-5, -5), textcoords="offset points")
    ax.annotate("$\Omega_3$", (2.5, 0.5), fontsize=20, xytext=(-5, -5), textcoords="offset points")
    ax.annotate("$\Omega_4$", (0.5, 1.5), fontsize=20, xytext=(-5, -5), textcoords="offset points")
    ax.annotate("$\Omega_5$", (1.5, 1.5), fontsize=20, xytext=(-5, -5), textcoords="offset points")
    ax.annotate("$\Omega_6$", (2.5, 1.5), fontsize=20, xytext=(-5, -5), textcoords="offset points")

    fig.show()

    cp_2 = np.array([[[0. , 0. ],
        [0.5, 0. ],
        [1. , 0. ],
        [1.5, 0. ],
        [2. , 0. ],
        [2.5, 0. ],
        [3. , 0. ]],

       [[0. , 0.5],
        [0.5, 0.5],
        [1. , 0.5],
        [1.5, 0.5],
        [2. , 0.5],
        [2.5, 0.5],
        [3. , 0.5]],

       [[0. , 1. ],
        [0.5, 1. ],
        [1. , 1. ],
        [1.5, 1. ],
        [2. , 1. ],
        [2.5, 1. ],
        [3. , 1. ]],

       [[0. , 1.5],
        [0.5, 1.5],
        [1. , 1.5],
        [1.5, 1.5],
        [2. , 1.5],
        [2.5, 1.5],
        [3. , 1.5]],

       [[0. , 2. ],
        [0.5, 2. ],
        [1. , 2. ],
        [1.5, 2. ],
        [2. , 2. ],
        [2.5, 2. ],
        [3. , 2. ]]])

    knots = (np.array([0. , 0. , 0. , 0.5, 0.5, 1. , 1. , 1. ]), np.array([0.        , 0.        , 0.        , 0.33333333, 0.33333333,
       0.66666667, 0.66666667, 1.        , 1.        , 1.        ]))

    fig2, ax2 = create_figure("2d")
    #ax2.set_axis_off()
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax2 = p_knots(knots, cp_2, ax=ax2, dim=2, point=False, line=True, color="black")
    ax2 = p_cpoints(cp_2, ax=ax2, dim=2, color="red", marker="o", point=True, line=False)
    n, m = cp_2.shape[0], cp_2.shape[1]
    P = np.asarray([(cp_2[x, y, 0], cp_2[x, y, 1]) for x in range(n) for y in range(m)])

    for count, point in enumerate(P):
        ax2.annotate(str(count), point, xytext =(5,5), textcoords="offset points")

    disp = (28,30)
    ax2.annotate("$\Omega_1$", (0.5, 0.5), fontsize=20, xytext=disp, textcoords="offset points")
    ax2.annotate("$\Omega_2$", (1.5, 0.5), fontsize=20, xytext=disp, textcoords="offset points")
    ax2.annotate("$\Omega_3$", (2.5, 0.5), fontsize=20, xytext=disp, textcoords="offset points")
    ax2.annotate("$\Omega_4$", (0.5, 1.5), fontsize=20, xytext=disp, textcoords="offset points")
    ax2.annotate("$\Omega_5$", (1.5, 1.5), fontsize=20, xytext=disp, textcoords="offset points")
    ax2.annotate("$\Omega_6$", (2.5, 1.5), fontsize=20, xytext=disp, textcoords="offset points")

    fig2.show()

    save = True
    if save or save_all:
        fig.savefig(fig_folder + "mesh_1degree.pdf",  **kwargs_savefig)
        fig2.savefig(fig_folder + "mesh_2degree.pdf", **kwargs_savefig)

def test_gempy_block_model():
    import gempy as gp
    import pandas as pd
    pd.set_option('precision', 2)
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/jan_models/"

    def create_model(resolution=[50, 50, 50]):
        geo_data = gp.create_data('fault', extent=[0, 1000, 0, 1000, 0, 1000], resolution=resolution,
                                  path_o=path_to_data + "model5_orientations.csv",
                                  path_i=path_to_data + "model5_surface_points.csv")

        geo_data.get_data()

        gp.map_stack_to_surfaces(geo_data, {"Fault_Series": 'fault',
                                            "Strat_Series": ('rock2', 'rock1')})
        geo_data.set_is_fault(['Fault_Series'])

        interp_data = gp.set_interpolator(geo_data, theano_optimizer='fast_compile')

        sol = gp.compute_model(geo_data)

        geo = gp.plot_2d(geo_data,
                         direction='y',
                         show_data=True,
                         show_lith=True,
                         show_boundaries=False)
        geo.axes[0].set_title("")
        plt.tight_layout()
        plt.close()
        return geo.axes[0].figure

    fig_coarse = create_model([20, 20, 20])
    fig_fine = create_model([70, 70, 70])

    fig_coarse.show()
    fig_fine.show()

    save = False
    if save or save_all:
        fig_coarse.savefig(fig_folder + "gempy_fault_coarse.pdf", **kwargs_savefig)
        fig_fine.savefig(fig_folder + "gempy_fault_fine.pdf", **kwargs_savefig)
