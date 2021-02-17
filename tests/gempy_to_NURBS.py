#%%
import numpy as np
import matplotlib.pyplot as plt
import gempy as gp
from igakit.nurbs import NURBS
import pandas as pd
pd.set_option('precision', 2)
#%%

data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
path_to_data = data_path + "/data/input_data/jan_models/"
extent = [0, 1000, 0, 1000, 0, 1000]
resolution = [50, 50, 50]
geo_data = gp.create_data('fold', extent=extent, resolution=resolution,
                          path_o=path_to_data + "model2_orientations.csv",
                          path_i=path_to_data + "model2_surface_points.csv")
gp.map_stack_to_surfaces(geo_data, {"Strat_Series": ('rock2', 'rock1'), "Basement_Series": ('basement')})


interp_data = gp.set_interpolator(geo_data, theano_optimizer='fast_compile')
sol = gp.compute_model(geo_data)

gp.plot_2d(geo_data, cell_number=15,
           direction='y', show_data=True)

gp.plot_2d(geo_data, cell_number=25,
           direction='x', show_data=True)

# %%
edges = geo_data.surfaces.df['edges'].copy()
vertices = geo_data.surfaces.df['vertices'].copy()
dx,dy,dz = geo_data._grid.regular_grid.get_dx_dy_dz(rescale=False)
#vertices=geo_data.solutions.vertices
#geo_data.surfaces.df['vertices']

# %%
fig = plt.figure("Curve")
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(vertices[0][:,0], vertices[0][:,1], vertices[0][:,2])
ax.plot_trisurf(vertices[1][:,0], vertices[1][:,1], vertices[1][:,2])

plt.show()

#%% md
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


#%%
df = pd.DataFrame({'col1':vertices[0][:,0],
                  'col2':vertices[0][:,1],
                  'col3':vertices[0][:,2]
                  })
#df_sort = df.sort_values(by=['col2','col1'])
df_sort=df
l_y = df_sort['col2'].values.tolist()
val_y=np.sort(list(set([x for x in l_y if l_y.count(x) > 1])))

l_x = df_sort['col1'].values.tolist()
val_x=np.sort(list(set([x for x in l_x if l_x.count(x) > 1])))
new_=[]
new_shape = []
for i in val_y:
    temp=df_sort.loc[df_sort['col2'] == i].to_numpy()
    shape_temp = temp.shape

    if new_ == []:
        new_.append(temp)
        new_shape.append(shape_temp[0])
    elif new_[-1].shape[0] - 5 < shape_temp[0]  < new_[-1].shape[0] + 5 :
        new_.append(temp)
        new_shape.append(shape_temp[0])
#%%
max_shape = np.argmax(np.bincount(new_shape))
for i in range(len(new_)):
    if new_[i].shape[0] != max_shape:
        to_add = max_shape - new_[i].shape[0]
        to_stack = np.asarray([new_[i][-1] for j in range(to_add)])
        new_[i] = np.vstack((new_[i], to_stack))
#%%
control_points = np.asarray(new_)

#%%
fig = plt.figure("Curve")
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(control_points[:, :, 0], control_points[:, :, 1], control_points[:, :, 2], color='red')
plt.show()
# %%
degree = 2
tot_U, tot_V, _ = control_points.shape

# %%%%%%%%%%%%%
# TO nurbs
#knot vector is always to m=n+p+1
def make_knot_vector(degree, len_cp):
    U = np.ones(len_cp+degree+1)
    num_add = len(U[degree+1:-degree-1])/degree
    vec_add=np.asarray([])
    if num_add.is_integer():
        for i in range(degree):
            vec_add = np.concatenate((vec_add, np.linspace(0.01, 0.09, int(num_add))))
        U[degree + 1:-degree - 1] = vec_add
    else:
        for i in range(degree):
            vec_add = np.concatenate((vec_add, np.linspace(0.01, 0.09, int(np.floor(num_add)))))
        U[degree + 1:-degree - 2] = vec_add

    U = np.sort(U)
    U[-degree-1:]=1
    U[:degree+1]=0
    return U

#%%
U = make_knot_vector(degree=2, len_cp=tot_U)
V = make_knot_vector(degree=2, len_cp=tot_V)

#%%%%
#to igakit
from igakit.plot import plt as plt_surf
surface = NURBS([U,V],control_points)

plt_surf.figure()
plt_surf.cpoint(surface)
plt_surf.cwire(surface)
plt_surf.curve(surface)
plt.show()

# %% anti-refinement
new_surf = surface.clone()
deviation= 10000000
for i in U:
    new_surf = new_surf.remove(0, i, deviation=deviation)
for j in V:
    new_surf = new_surf.remove(1, i, deviation=deviation)
plt_surf.figure()
plt_surf.cpoint(new_surf)
plt.show()
print(surface.shape, new_surf.shape)
#%%
import pygeoiga as gn
resolution = 250
kwargs_={'degree':[20,20]}
positions = gn.engine.NURBS_Surface(control_points,[surface.knots[0], surface.knots[1]], resolution=resolution)
fig = plt.figure("surface1")
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(positions[:, 0], positions[:, 1],positions[:, 2])#, linestyle='None', marker='.', color='blue')
#ax.plot(B[:, :, 0], B[:, :, 1], linestyle='None', marker='.', color='red')  # , C[:, :, 2], color='red')
fig.show()

positions = gn.engine.NURB_construction([new_surf.knots[0], new_surf.knots[1]], new_surf.control, resolution=resolution)
fig = plt.figure("surface2")
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(positions[:, 0], positions[:, 1],positions[:, 2])#, linestyle='None', marker='.', color='blue')
#ax.plot(B[:, :, 0], B[:, :, 1], linestyle='None', marker='.', color='red')  # , C[:, :, 2], color='red')
fig.show()
#%%
positions=surface(np.linspace(0,1,50), np.linspace(0,1,50))
fig = plt.figure("Curve")
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(positions[..., 0], positions[..., 1],positions[..., 2], linestyle='None', marker='.', color='blue')
#ax.plot(B[:, :, 0], B[:, :, 1], linestyle='None', marker='.', color='red')  # , C[:, :, 2], color='red')
fig.show()
#%%

fig = plt.figure("surface1")
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(surface.control[...,0],surface.control[...,1],surface.control[..., 2])#, linestyle='None', marker='.', color='blue')
#ax.plot(B[:, :, 0], B[:, :, 1], linestyle='None', marker='.', color='red')  # , C[:, :, 2], color='red')
fig.show()

fig = plt.figure("surface1")
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(new_surf.control[...,0],new_surf.control[...,1],new_surf.control[..., 2])#, linestyle='None', marker='.', color='blue')
ax.set_zlim(200)
#ax.plot(B[:, :, 0],
fig.show()

#%%

