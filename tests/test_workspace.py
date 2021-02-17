#%%
import pygeoiga as gn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('QT5Agg')


#%%
knot_vector=np.array([0,0,0,1,2,3,4,4,5,5,5])
knot_vector1=np.array([0,0,0,0,1,5,6,8,8,8,8])
cpoints =  np.array([[0,0,0],
                         [1,0,0],
                         [2,1,0],
                         [2, 2, 0],
                         [3, 2, 0],
                         [3, 1, 0],
                         [4, 1, 0],
                         [4, 0, 0]
                         ])
weight=np.array([1,0.5,0.5,1,1,1,1,1])
weight1 = np.ones(8)
degree = len(np.where(knot_vector == 0.)[0]) - 1
degree1 = len(np.where(knot_vector == 0.)[0]) - 1
resolution = 100

#%%
curve1_cpoints = np.array([[0,0,0],
                         [1,0,0],
                         [2,1,0],
                         [0, 2,0]])
knot_vector_curve1 =np.array([0,0,0,0.5,1,1,1])


#%%
def make_srf():
    C = np.zeros((3,5,5))
    C[:,:,0] = [[ 0.0,  3.0,  5.0,  8.0, 10.0],
                [ 0.0,  0.0,  0.0,  0.0,  0.0],
                [ 2.0,  2.0,  7.0,  7.0,  8.0],]
    C[:,:,1] = [[ 0.0,  3.0,  5.0,  8.0, 10.0],
                [ 3.0,  3.0,  3.0,  3.0,  3.0],
                [ 0.0,  0.0,  5.0,  5.0,  7.0],]
    C[:,:,2] = [[ 0.0,  3.0,  5.0,  8.0, 10.0],
                [ 5.0,  5.0,  5.0,  5.0,  5.0],
                [ 0.0,  0.0,  5.0,  5.0,  7.0],]
    C[:,:,3] = [[ 0.0,  3.0,  5.0,  8.0, 10.0],
                [ 8.0,  8.0,  8.0,  8.0,  8.0],
                [ 5.0,  5.0,  8.0,  8.0, 10.0],]
    C[:,:,4] = [[ 0.0,  3.0,  5.0,  8.0, 10.0],
                [10.0, 10.0, 10.0, 10.0, 10.0],
                [ 5.0,  5.0,  8.0,  8.0, 10.0],]
    C = C.transpose()
    U = np.asarray([0, 0, 0, 1/3., 2/3., 1, 1, 1])
    V = np.asarray([0, 0, 0, 1/3., 2/3., 1, 1, 1])

    return C, U, V
#%%
cpoints, knot1, knot2 = make_srf()
#%%
from igakit.nurbs import NURBS
from igakit.plot import plt
import matplotlib

matplotlib.use('Qt5Agg')

C1 = [[-1.5, 0],
      [-1, 0.5],
      [-1, 1]]

U = [0, 0, 0, 1, 1, 1]

crv = NURBS([U], C1)
plt.figure()
plt.cpoint(crv)
plt.cwire(crv)
plt.plot(crv)
plt.show()

#%%
C1 = [[-1, 1],
      [-1.2, 2],
      [-1, 3]]


U1 = [0, 0, 0, 1, 1, 1]
crv1 = NURBS([U1], C1)
plt.figure()
plt.cpoint(crv1)
plt.cwire(crv1)
plt.plot(crv1)
plt.show()

#%%
from igakit.cad import *
crv_diapir = join(crv,crv1, axis =0)
plt.figure()
plt.cpoint(crv_diapir)
plt.cwire(crv_diapir)
plt.plot(crv_diapir)
ax = plt.backend.gca()
ax.set_xlim((-3,3))
ax.set_ylim((0,6))
plt.show()
#%%
C2 = [[-1, 3],
      [0, 4],
      [1, 3]]
U2 = [0, 0, 0, 1, 1, 1]
cr2 = NURBS([U2], C2)

crv_diapir = join(crv_diapir, cr2, axis = 0)
plt.figure()
plt.cpoint(crv_diapir)
plt.cwire(crv_diapir)
plt.plot(crv_diapir)
ax = plt.backend.gca()
ax.set_xlim((-3,3))
ax.set_ylim((0,6))
plt.show()
#%%
C3 = [[1, 3],
      [1.2, 2],
      [1, 1],
      [1, 0.5],
      [1.5,0]]
U3 = [0,0,0,1,1,2,2,2]
cr3 = NURBS([U3], C3)

crv_diapir = join(crv_diapir, cr3, axis = 0)
plt.figure()
plt.cpoint(crv_diapir)
plt.cwire(crv_diapir)
plt.plot(crv_diapir)
ax = plt.backend.gca()
ax.set_xlim((-3,3))
ax.set_ylim((0,6))
plt.show()

#%%
c_bot = [[-1.5,0],
         [0,0],
         [1.5,0]]
U = [0,0,0,1,1,1]
cr_bot = NURBS([U], c_bot)
#%%
diapir = ruled(crv_diapir, cr_bot)
plt.figure()
#plt.cpoint(diapir)
plt.cwire(diapir)
plt.plot(diapir, resolution=100)
plt.kpoint(diapir)
ax = plt.backend.gca()
ax.set_xlim((-3,3))
ax.set_ylim((0,6))
plt.show()
#%%
C_1 = [[-1.5, 0],
      [-1, 0.5],
      [-1, 1],
      [-1.2, 2],
      [-1, 3],
       [-0.5, 3.8],
      [0, 4],]
U_1 = [0,0,0,1,1,2,2,3,3,3]
cr_1 = NURBS([U_1], C_1)

plt.cwire(cr_1)
plt.cpoint(cr_1)
plt.plot(cr_1)
plt.kpoint(cr_1)
ax = plt.backend.gca()
ax.set_xlim((-3,3))
ax.set_ylim((0,6))
plt.show()
#%%
#C_2 = [[0, 4],
#       [0.5, 3.8],
#       [1, 3],
#       [1.2, 2],
##       [1, 1],
 #      [1, 0.5],
 #      [1.5, 0],]
C_2 = [[1.5, 0],
      [1, 0.5],
      [1, 1],
      [1.2, 2],
      [1, 3],
       [0.5, 3.8],
      [0, 4],]
U_2 = [0,0,0,1,1,2,2,3,3,3]
cr_2 = NURBS([U_2], C_2)

plt.cwire(cr_2)
plt.cpoint(cr_2)
plt.plot(cr_2)
plt.kpoint(cr_2)
ax = plt.backend.gca()
ax.set_xlim((-3,3))
ax.set_ylim((0,6))
plt.show()

#%%
diapir2 = ruled(cr_1, cr_2)
diapir2.elevate(axis=1, times=1)
#plt.cwire(diapir2)
plt.cpoint(diapir2)
plt.plot(diapir2, alpha=0.5)
plt.kpoint(diapir2)
plt.kwire(diapir2)
plt.ksurf(diapir2)
ax = plt.backend.gca()
ax.set_xlim((-3,3))
ax.set_ylim((0,6))
plt.show()
#%%
left= [[-3,0],
       [-3, 0.5],
       [-3,1]]
right = [[-1.5, 0],
      [-1, 0.5],
      [-1, 1]]
u = [0,0,0,1,1,1]
_1 = NURBS([u], left)
_2 = NURBS([u], right)
bottom = ruled(_1, _2)
bottom.elevate(axis=1,times=1)
#plt.cwire(diapir2)
plt.cpoint(bottom)
plt.plot(bottom, alpha=0.5)
plt.kpoint(bottom)
plt.kwire(bottom)
plt.ksurf(bottom)
ax = plt.backend.gca()
ax.set_xlim((-3,3))
ax.set_ylim((0,6))
plt.show()

#%%
possible = join(bottom, diapir2, axis=1)
plt.cpoint(possible)
plt.plot(possible, alpha=0.5)
plt.kpoint(possible)
plt.kwire(possible)
plt.ksurf(possible)
ax = plt.backend.gca()
ax.set_xlim((-3,3))
ax.set_ylim((0,6))
plt.show()
#%%
plt.figure()
plt.cpoint(bottom)
plt.plot(bottom, alpha=0.2)
plt.kpoint(bottom)
plt.ksurf(bottom)
plt.cpoint(diapir2)
plt.plot(diapir2, alpha=0.2, color="green")
plt.kwire(diapir2, color="black")
plt.ksurf(diapir2)
ax = plt.backend.gca()
ax.set_xlim((-3,3))
ax.set_ylim((0,6))
plt.show()

#%%
crv = circle()
rf = revolve(crv, point=0, axis=2)
plt.figure()
plt.plot(crv)
plt.show()
plt.figure()
plt.plot(rf)
plt.show()

#%%
from igakit.nurbs import NURBS
from igakit.plot import plt
import matplotlib
matplotlib.use('Qt5Agg')

C1 = [[0, 0],
      [250, 0],
      [500, 0]]
U = [0, 0, 0, 1, 1, 1]

C2 = [[0, 100],
      [250, 360],
      [500, 100]]

C3 = [[0, 300],
      [250, 400],
      [500, 300]]
C4 = [[0, 500],
      [250, 500],
      [500, 500]]

bot = NURBS([U], C1)
bot_lay = NURBS([U], C2)
top_lay = NURBS([U], C3)
top = NURBS([U], C4)

plt.figure()
plt.plot(bot)
plt.plot(bot_lay)
plt.plot(top_lay)
plt.plot(top)
plt.show()

#%%
from igakit.cad import ruled
bottom = ruled(bot, bot_lay)
middle = ruled(bot_lay, top_lay)
upper = ruled(top_lay, top)

plt.figure()
color = ["red", "blue", "yellow"]
for i, surf in enumerate([bottom, middle, upper]):
    plt.plot(surf, color =color[i], alpha =0.2)
    plt.kplot(surf)
    plt.cplot(surf)
plt.show()

#%%
value = [0.2,0.4,0.6,0.8]
bottom.elevate(axis=1, times=1)
middle.elevate(axis=1, times=1)
upper.elevate(axis=1, times=1)
bottom.refine(axis=0, values=value).refine(axis=1, values=value)
middle.refine(axis=0, values=value).refine(axis=1, values=value)
upper.refine(axis=0, values=value).refine(axis=1, values=value)

plt.figure()
color = ["red", "blue", "yellow"]
for i, surf in enumerate([bottom, middle, upper]):
    plt.plot(surf, color =color[i], alpha =0.2)
    plt.kplot(surf)
    #plt.cplot(surf)
plt.show()

