import gempyExplicit
import pyvista
import numpy as np
#import skfmm
#import pymesh
import matplotlib.pyplot as plt
from pyvista import examples
p=pyvista.Plotter(notebook= False)
##############################################################
points1D = np.array([[30,0,15],[30,5,15],[30,10,15],[25,10,15],[25,5,15],[25,0,15]])

knot10=np.array([0,0,0,0.3,0.5,0.7,1,1,1])#D1
weight1D=np.array([[[1],[1],[1],[1],[1],[1]]])
model1D= gempyExplicit.NURBS_Curve(2, knot10, points1D, weight1D, 50,"auto")
cloud1D = pyvista.PolyData(model1D)

p.add_mesh(cloud1D, color="red")


#################################################################################
points2D = np.array([[[150,0,50],[150,25,50],[150,50,50],[140,50,50],[140,25,50],[140,0,50]],
                         [[150,0,30],[150,25,30],[150,50,30],[140,50,30],[140,25,30],[140,0,30]],
                         [[150,0,15],[150,25,15],[150,50,15],[140,50,15],[140,25,15],[140,0,15]],
                         [[150,0,0],[150,25,0],[150,50,0],[140,50,0],[140,25,0],[140,0,0]]])


knot1 = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # D1
knot2=np.array([0,0,0,0.3,0.5,0.7,1,1,1])#D2


weight2D=np.array([[[1],[1],[1],[1],[1],[1]],
                    [[1],[1],[1],[1],[1],[1]],
                    [[1],[1],[1],[1],[1],[1]],
                    [[1], [1], [1], [1], [1],[1]]])

model2D= gempyExplicit.NURBS_Surface(3, 2, knot1, knot2,points2D, weight2D, 20, 20,"auto")[0]
cloud2D = pyvista.PolyData(model2D)
surf2D = cloud2D.delaunay_3d()
#p.add_mesh(cloud2D, color="red")

triangles=gempyExplicit.NURBS_Surface(3, 2, knot1, knot2,points2D, weight2D, 20, 20,"auto")[1]
#print(triangles)



SurfaceMesh=preceduralMesh(model2D,triangles)
print(SurfaceMesh)
p.add_mesh(SurfaceMesh,scalars=np.arange(722))
#np.savetxt('2D_test.text', model2D, delimiter=',')
#############################################
#points3D = np.array([[[[15,0,15],[15,10,15],[15,20,15],[1,20,15],[1,10,15],[1,0,15]],
#                    [[15,0,10],[15,10,10],[15,20,10],[1,20,10],[1,10,10],[1,0,10]],
#                     [[15, 0, 5], [15, 10, 5], [15, 20, 5], [1, 20, 5], [1, 10, 5], [1, 0, 5]],
#                     [[15, 0, 0], [15, 10, 0], [15, 20, 0], [1, 20, 0], [1, 10, 0], [1, 0, 0]]],

#                     [[[13,0,15],[13,10,15],[13,20,15],[3,20,15],[3,10,15],[3,0,15]],
#                    [[13,0,10],[13,10,10],[13,20,10],[3,20,10],[3,10,10],[3,0,10]],
#                     [[13, 0, 5], [13, 10, 5], [13, 20, 5], [3, 20, 5], [3, 10, 5], [3, 0, 5]],
#                     [[13, 0, 0], [13, 10, 0], [13, 20, 0], [3, 20, 0], [3, 10, 0], [3, 0, 0]]],

#                     [[[11, 0, 15], [11, 10, 15], [11, 20, 15], [5, 20, 15], [5, 10, 15], [5, 0, 15]],
#                      [[11, 0, 10], [11, 10, 10], [11, 20, 10], [5, 20, 10], [5, 10, 10], [5, 0, 10]],
#                      [[11, 0, 5], [11, 10, 5], [11, 20, 5], [5, 20, 5], [5, 10, 5], [5, 0, 5]],
#                      [[11, 0, 0], [11, 10, 0], [11, 20, 0], [5, 20, 0], [5, 10, 0], [5, 0, 0]]],

#                     [[[9, 0, 15], [9, 10, 15], [9, 20, 15], [7, 20, 15], [7, 10, 15], [7, 0, 15]],
#                      [[9, 0, 10], [9, 10, 10], [9, 20, 10], [7, 20, 10], [7, 10, 10], [7, 0, 10]],
#                      [[9, 0, 5], [9, 10, 5], [9, 20, 5], [7, 20, 5], [7, 10, 5], [7, 0, 5]],
#                      [[9, 0, 0], [9, 10, 0], [9, 20, 0], [7, 20, 0], [7, 10, 0], [7, 0, 0]]],
#
#                                                                                             ])

points3D = np.array([[[[100, 0, 50],[100, 50, 50],[50, 50, 50],[45, 20, 50],[15, 20, 50],[15, 50, 50]],
                      [[100, 0, 30],[100, 50, 30],[50, 50, 30],[45, 20, 30],[15, 20, 30],[15, 50, 30]],
                      [[100, 0, 15],[100, 50, 15],[50, 50, 15],[45, 20, 15],[15, 20, 15], [15, 50, 15]],
                      [[100, 0, 0], [100, 50, 0], [50, 50, 0],[ 45, 20, 0], [15, 20, 0], [ 15, 50, 0]]],

                     [[[95, 0, 50], [95, 45, 50], [55, 45, 50], [50, 15, 50], [10, 15, 50], [10, 50, 50]],
                      [[95, 0, 30], [95, 45, 30], [55, 45, 30], [50, 15, 30], [10, 15, 30], [10, 50, 30]],
                      [[95, 0, 15], [95, 45, 15], [55, 45, 15], [50, 15, 15], [10, 15, 15], [10, 50, 15]],
                      [[95, 0, 0],  [95, 45, 0],  [55, 45, 0],  [50, 15, 0], [10, 15, 0], [10, 50, 0]]],

                     [[[90, 0, 50], [90, 40, 50], [60, 40, 50], [55, 10, 50], [5, 10, 50], [5, 50, 50]],
                      [[90, 0, 30], [90, 40, 30], [60, 40, 30], [55, 10, 30], [5, 10, 30], [5, 50, 30]],
                      [[90, 0, 15], [90, 40, 15], [60, 40, 15], [55, 10, 15], [5, 10, 15], [5, 50, 15]],
                      [[90, 0, 0], [90, 40, 0],  [60, 40, 0],   [55, 10, 0],  [5, 10, 0],  [5, 50, 0]]],

                     [[[85, 0, 50], [85, 35, 50], [65, 35, 50], [60, 5, 50], [0, 5, 50], [0, 50, 50]],
                      [[85, 0, 30], [85, 35, 30], [65, 35, 30], [60, 5, 30], [0, 5, 30], [0, 50, 30]],
                      [[85, 0, 15], [85, 35, 15], [65, 35, 15], [60, 5, 15], [0, 5, 15], [0, 50, 15]],
                      [[85, 0, 0], [85, 35, 0], [65, 35, 0],  [60, 5, 0],  [0, 5, 0], [0, 50, 0]]],

                                                                                             ])

points3D_2 = np.array([[[[-50, 50, 50],[-50, 50, 50],[-100, 50, 50],[-100, 50, 50],[-150, 50, 50],[-150, 50, 50]],
                      [[-50, 50, 30],[-50, 50, 30],[-100, 50, 30],[-100, 50, 30],[-150, 50, 30],[-150, 50, 30]],
                      [[-50, 50, 15],[-50, 50, 15],[-100, 50, 15],[-100, 50, 15],[-150, 50, 15], [-150, 50, 15]],
                      [[-50, 50, 0], [-50, 50, 0], [-100, 50, 0],[ -100, 50, 0], [-150, 50, 0], [ -150, 50, 0]]],

                     [[[-80, 30, 50], [-80, 40, 50], [-120, 40, 50], [-120, 35, 50], [-180, 35, 50], [-180, 40, 50]],
                      [[-80, 30, 30], [-80, 40, 30], [-120, 40, 30], [-120, 35, 30], [-180, 35, 30], [-180, 40, 30]],
                      [[-80, 30, 15], [-80, 40, 15], [-120, 40, 15], [-120, 35, 15], [-180, 35, 15], [-180, 40, 15]],
                      [[-80, 30, 0],  [-80, 40, 0],  [-120, 40, 0],  [-120, 35, 0], [-180, 35, 0], [-180, 40, 0]]],

                     [[[-30, 20, 50], [-30, 30, 50], [-70, 30, 50], [-70, 25, 50], [-120, 25, 50], [-120, 30, 50]],
                      [[-30, 20, 30], [-30, 30, 30], [-70, 30, 30], [-70, 25, 30], [-120, 25, 30], [-120, 30, 30]],
                      [[-30, 20, 15], [-30, 30, 15], [-70, 30, 15], [-70, 25, 15], [-120, 25, 15], [-120, 30, 15]],
                      [[-30, 20, 0],  [-30, 30, 0],  [-70, 30, 0],  [-70, 25, 0], [-120, 25, 0], [-120, 30, 0]]],

                       [[[-50, 0, 50], [-50, 0, 50], [-100, 0, 50], [-100, 0, 50], [-150, 0, 50], [-150, 0, 50]],
                        [[-50, 0, 30], [-50, 0, 30], [-100, 0, 30], [-100, 0, 30], [-150, 0, 30], [-150, 0, 30]],
                        [[-50, 0, 15], [-50, 0, 15], [-100, 0, 15], [-100, 0, 15], [-150, 0, 15], [-150, 0, 15]],
                        [[-50, 0, 0], [-50, 0, 0], [-100, 0, 0], [-100, 0, 0], [-150, 0, 0], [-150, 0, 0]]],

                                                                                             ])


knot11 = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # D1
knot22 = np.array([0, 0, 0, 0, 1, 1, 1, 1]) #D2
weight3D=np.array([[[[1],[1],[1],[1],[1],[1]],
                    [[1],[1],[1],[1],[1],[1]],
                     [[1], [1], [1], [1], [1], [1]],
                     [[1], [1], [1], [1], [1], [1]]],

                   [[[1], [1], [1], [1], [1], [1]],
                    [[1], [1], [1], [1], [1], [1]],
                    [[1], [1], [1], [1], [1], [1]],
                    [[1], [1], [1], [1], [1], [1]]],

                   [[[1], [1], [1], [1], [1], [1]],
                    [[1], [1], [1], [1], [1], [1]],
                    [[1], [1], [1], [1], [1], [1]],
                    [[1], [1], [1], [1], [1], [1]]],

                   [[[1], [1], [1], [1], [1], [1]],
                    [[1], [1], [1], [1], [1], [1]],
                    [[1], [1], [1], [1], [1], [1]],
                    [[1], [1], [1], [1], [1], [1]]],

                                                                                             ])

knot33=np.array([0,0,0,0.3,0.5,0.7,1,1,1])#D3



model3D= gempyExplicit.NURBS_Volume(3,3, 2, knot11, knot22,knot33, points3D, weight3D, 30, 30,30, "auto")[0]
model3D_2= gempyExplicit.NURBS_Volume(3,3, 2, knot11, knot22,knot33, points3D_2, weight3D, 30, 30,30, "auto")[0]
triangles3D=gempyExplicit.NURBS_Volume(3,3, 2, knot11, knot22,knot33, points3D, weight3D, 30, 30,30, "auto")[1]
triangles3D_2=gempyExplicit.NURBS_Volume(3,3, 2, knot11, knot22,knot33, points3D_2, weight3D, 30, 30,30, "auto")[1]
surface_triangles=gempyExplicit.NURBS_Volume(3,3, 2, knot11, knot22,knot33, points3D, weight3D, 30, 30,30, "auto")[2]
surface_triangles_2=gempyExplicit.NURBS_Volume(3,3, 2, knot11, knot22,knot33, points3D_2, weight3D, 30, 30,30, "auto")[2]
#print(model3D)

voxels=gempyExplicit.NURBS_Voxels(20, 20,20)



np.savetxt('3D_test.text', surface_triangles, delimiter=',')

VolumeMesh=preceduralMesh(model3D,triangles3D)
VolumeMeshSUrfaces=preceduralMesh(model3D,surface_triangles)
VolumeMesh_2=preceduralMesh(model3D_2,triangles3D_2)

p.add_mesh(VolumeMesh,scalars=np.arange(151380))
p.add_mesh(VolumeMesh_2,scalars=np.arange(151380))

point3DforWidget=gempyExplicit.Conv1(points3D)
point3DforWidget_2=gempyExplicit.Conv1(points3D_2)

def update_volune_NURBS(point, i):
    point3DforWidget[i] = point
    points3DNew= point3DforWidget.reshape(points3D.shape[0],points3D.shape[1],points3D.shape[2],points3D.shape[3])
    model3DNew = gempyExplicit.NURBS_Volume(3, 3, 2, knot11, knot22, knot33, points3DNew, weight3D, 30, 30, 30, "auto")[0]
    VolumeMesh.points=model3DNew

    return

#p.add_sphere_widget(update_volune_NURBS, center=point3DforWidget, radius=0.8, color="green")

def update_volune_NURBS_2(point, i):
    point3DforWidget_2[i] = point
    points3DNew_2= point3DforWidget_2.reshape(points3D_2.shape[0],points3D_2.shape[1],points3D_2.shape[2],points3D_2.shape[3])
    model3DNew_2 = gempyExplicit.NURBS_Volume(3, 3, 2, knot11, knot22, knot33, points3DNew_2, weight3D, 30, 30, 30, "auto")[0]
    VolumeMesh_2.points=model3DNew_2

    return

p.add_sphere_widget(update_volune_NURBS_2, center=point3DforWidget_2, radius=0.8, color="green")


#p.set_background("royalblue", top="aliceblue")
p.show_axes()
p.show_grid()
p.show()


