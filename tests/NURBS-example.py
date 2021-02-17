#%%
import sys,os
print(os.getcwd())

#sys.path.append('GempyExplicit/python/')
#sys.path.append('/home/danielsk78/Daniel/GempyExplicit/python/')

import gempyExplicit
import pyvista
import numpy as np
#%%
points_gempy2 = np.array(

                     [[[9,0,15],[9,5,15],[9,10,15],[8,10,15],[8,5,15],[8,0,15]],
                     [[10,0,10],[10,5,10],[10,10,10],[9,10,10],[9,5,10],[9,0,10]],
                     [[11,0,5],[11,5,5],[11,10,5],[10,10,5],[10,5,5],[10,0,5]],
                     [[12,0,0],[12,5,0],[12,10,0],[11,10,0],[11,5,0],[11,0,0]]])

weighttt = np.array(

                 [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]])


knot11 = np.array([0, 0, 0, 0, 1, 1, 1, 1])

knot22=np.array([0,0,0,0.3,0.5,0.7,1,1,1])

model2= gempyExplicit.NURBS_Surface(3, 2, knot11, knot22, points_gempy2, weighttt, 50, 50, "auto")[0]
re_points=np.reshape(points_gempy2.ravel(),(24,3))
#model3 = gempyExplicit.NURBS_Surface(3, 2, knot11, knot22, re_points, weighttt, 50, 50, "auto")
print(model2)
cloud2 = pyvista.PolyData(model2)
#%%
#cloud2.points.shape
model2.shape
#%%
def update_surface(point, i):
    re_points[i] = point
    temp= re_points
    to_model = np.reshape(temp.ravel(),(4,6,3))
    model_p = gempyExplicit.NURBS_Surface(3,2, knot11, knot22, to_model, weighttt, 50, 50, "auto")
    cloud2.points = model_p
    return
#%%
surf2 = cloud2.delaunay_3d()

p = pyvista.Plotter()
p.add_sphere_widget(update_surface, center=re_points)
#p.add_points(re_points)
p.add_mesh(surf2, color="red")

p.set_background("blue")

p.show_axes()

p.show_grid()

p.show()

if __name__ =='__main__':
    pass