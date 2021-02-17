#from pygeoiga import NURB
import pyvista
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pyvista import examples

def plot_cpoints(B, ax = None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')




class Plotter:
    """
    Class that manage the ploting of a nurb object and shows it in a pyvista and/or matplotlib figure
    """
    def __init(self, nurb):# NURB):
        """

        Args:
            nurb: give a nurb that contains all the information of degree, knots, control points...
        Returns:
        """
        self.nurb = nurb
        self.p = pyvista.Plotter(notebook=False)
        self.use_backend('Qt5Agg')
        self.fig = None
        self.ax = None

    def create_figure(self):
        """
        creates a pyvista and /or matplotlib object
        Returns:
        """
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.close()
        return self.fig, self.ax

    def preceduralMesh(self, model, triangles):
        add4 = []  # for surf.faces
        for e in triangles:
            ee = np.array(e)
            l = np.append(3, ee)
            add4.append(l)

        MESHH = pyvista.PolyData()
        MESHH.points = np.array(model, dtype=float)
        MESHH.faces = np.array(add4)
        return MESHH

    def simple_plot(self):
        self.ax.plot(self.nurb.model[:,0],self.nurb.model[:,1],self.nurb.model[:,2])

    def plot_cpoint(self, ax):
        if self.nurb.dim == 1:
            ax.plot(self.nurb.cpoints[:, 0],
                    self.nurb.cpoints[:, 1],
                    self.nurb.cpoints[:, 2],
                    color='red', marker='s', linestyle='None')
        elif self.nurb.dim == 2:
            pass
        elif self.nurb.dim == 3:
            pass
        return ax

    def use_backend(self, backend: str):
        matplotlib.use(backend)
