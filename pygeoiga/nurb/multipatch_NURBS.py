import traceback
import numpy as np
from collections import OrderedDict
from pygeoiga.nurb.nurb_creation import NURB
from pygeoiga.plot.nrbplotting_mpl import p_cpoints, p_knots, p_surface, create_figure
from pygeoiga.plot.solution_mpl import p_temperature
import matplotlib.colors as mcolors
from tqdm.autonotebook import tqdm

class Multipatch(object):
    """Manage Multiple NURBS patches"""
    def __init__(self):
        self.geometry = OrderedDict({})

    def add_patch(self, nurb: NURB, name: str, kappa: float = 1, color: str = None, position: tuple = (1, 1)):
        """
        Add a nurb patch to the
        Args:
            nurb: nurb object
            name: name of the patch
            kappa: Material property -> Thermal conductivity of pacth
            color: (Optional) Color for plotting
            position: Position of patch in a grid

        Returns:

        """
        if color is None: # add random color
            opt = mcolors.CSS4_COLORS.keys()
            color = list(opt)[np.random.randint(0, len(opt))]
        self.geometry[name] = {"B": nurb.B,
                               "knots": nurb.knots,
                               "weight": nurb.weight,
                               "kappa": kappa,
                               'color': color,
                               "position": position,
                               "nrb": nurb,
                               "patch_faces": {}}

    def plot(self, ax=None, kwargs_surface={}, kwargs_knots={}, kwargs_cpoints={}):
        if ax is None:
            fig, ax = create_figure("2d", figsize=(10, 20))

        ax = self.plot_surfaces(ax, **kwargs_surface)
        ax = self.plot_knots(ax, **kwargs_knots)
        ax = self.plot_cpoints(ax, **kwargs_cpoints)
        return ax

    def plot_knots(self, ax=None, **kwargs_knots):
        if ax is None:
            fig, ax = create_figure("2d", figsize=(10, 20))
        for patch_name in tqdm(self.geometry.keys(), desc="Plotting knots"):
            ax = p_knots(self.geometry[patch_name].get("knots"),
                         self.geometry[patch_name].get("B"),
                         weight=self.geometry[patch_name].get("weight"),
                         ax=ax,
                         **kwargs_knots)
        return ax

    def plot_cpoints(self, ax=None, **kwargs_cpoints):
        if ax is None:
            fig, ax = create_figure("2d", figsize=(10, 20))
        for patch_name in tqdm(self.geometry.keys(), desc="Plotting control points"):
            ax = p_cpoints(self.geometry[patch_name].get("B"),
                           ax=ax,
                           **kwargs_cpoints)
        return ax

    def plot_surfaces(self, ax=None, **kwargs_surface):
        if ax is None:
            fig, ax = create_figure("2d", figsize=(10, 20))

        for patch_name in tqdm(self.geometry.keys(), desc="Plotting surfaces"):
            ax = p_surface(self.geometry[patch_name].get("knots"),
                           self.geometry[patch_name].get("B"),
                           weight=self.geometry[patch_name].get("weight"),
                           ax=ax,
                           color=self.geometry[patch_name].get("color"),
                           **kwargs_surface)
        return ax

    def plot_solution(self, ax = None, **kwargs_temperature):
        if ax is None:
            fig, ax = create_figure("2d", figsize=(10, 20))
        tmin = 200
        tmax = -200
        for patch_name in self.geometry.keys():
            t = self.geometry[patch_name].get("t_sol")
            tmin = t.min() if t.min() < tmin else tmin
            tmax = t.max() if t.max() > tmax else tmax

        for patch_name in tqdm(self.geometry.keys(), desc="Plotting solutions"):
            x = self.geometry[patch_name].get("x_sol")
            y = self.geometry[patch_name].get("y_sol")
            t = self.geometry[patch_name].get("t_sol")
            ax = p_temperature(x,
                               y,
                               t,
                               vmin=tmin,
                               vmax=tmax,
                               ax=ax,
                               **kwargs_temperature)
        return ax

    def global_knot_insertion(self, knot_ins: list, direction: int = 0):
        """
        Refinement by knot insertion
        Args:
            knot_ins: list of knots to insert
            direction: parametric direction to insert them

        Returns:

        """
        for patch_name in tqdm(self.geometry.keys(), desc="Inserting knots by patch"):
            self.knot_insertion_by_name(patch_name, knot_ins, direction)
        return True

    def knot_insertion_by_name(self, name: str, knot_ins: list, direction: int = 0):
        """
        Refine specific patch by knot insertion
        Args:
            name: 
            knot_ins: list of knots to insert
            direction: parametric direction to insert them

        Returns:

        """
        if name in self.geometry.keys():
            self.geometry[name].get("nrb").knot_insert(knot_ins, direction, leave=False)
            self.geometry[name]["B"] = self.geometry[name].get("nrb").B
            self.geometry[name]["knots"] = self.geometry[name].get("nrb").knots
            self.geometry[name]["weight"] = self.geometry[name].get("nrb").weight
        else:
            return False

    def define_topology(self):
        """
        Reads the position of the patch and relates the faces between patches
        # BOUNDARIES - faces of the patch in contact
        # 0: down
        # 1: right
        # 2: up
        # 3: left
        Returns:
        """
        for patch_name_current in self.geometry.keys():
            position_current = self.geometry[patch_name_current].get("position")
            for patch_name in self.geometry.keys():
                position = self.geometry[patch_name].get("position")
                if position_current != position:
                    if position[0] == position_current[0]: # They are in the same row
                        if position[1] == position_current[1]+1: # Is right
                            self.geometry[patch_name_current]["patch_faces"][1] = patch_name
                        elif position[1] == position_current[1]-1: # Is left
                            self.geometry[patch_name_current]["patch_faces"][3] = patch_name

                    if position[1] == position_current[1]: # They are in the same column
                        if position[0] == position_current[0]+1: # Is up
                            self.geometry[patch_name_current]["patch_faces"][2] = patch_name
                        elif position[0] == position_current[0]-1: # Is down
                            self.geometry[patch_name_current]["patch_faces"][0] = patch_name
        self.assert_continuity()
        return True

    def assign_boundary_condition(self, name_patch: str, id_boundary: str, face: int):
        """
        Takes a patch and set an id for where is the boundary condition
        Args:
            name_patch: name of the patch
            id_boundary: string to save the boundary
            face: 0: down
                  1: right
                  2: up
                  3: left
        Returns:

        """
        if name_patch in self.geometry.keys():
            bc = self.geometry[name_patch].get("BC")
            if bc is None:
                self.geometry[name_patch]["BC"] = {face: id_boundary}
            else:
                bc[face] = id_boundary
                self.geometry[name_patch]["BC"] = bc
            return True
        return False

    def fill_topological_info(self):
        from pygeoiga.analysis.MultiPatch import patch_topology
        self.geometry, gDoF = patch_topology(self.geometry)
        return gDoF

    def assert_continuity(self):
        """
        Between patches need to have continuity.
        This is achieved by repeating the control points at the connecting patches
        Returns:

        """
        def get_P(B):
            n, m = B.shape[0], B.shape[1]
            P = []
            for y in range(m):
                for x in range(n):
                    val = B[x, y, :].tolist()
                    P.append(val)
            return np.asarray(P)

        for patch_name in self.geometry.keys():
            B = self.geometry[patch_name].get("B")
            n, m = B.shape[0], B.shape[1]
            dof = get_P(B)
            sides = self.geometry[patch_name].get("patch_faces").keys()
            for face in sides:
                if face == 0:  # loking down
                    # position_repeated_cp.append(dof[:n])
                    name_contact = self.geometry[patch_name]["patch_faces"].get(face)
                    B_compare = self.geometry[name_contact].get("B")
                    n_comp, m_comp = B_compare.shape[0], B_compare.shape[1]
                    dof_compare = get_P(B_compare)
                    assert np.all(dof_compare[-n_comp:] == dof[:n]), "Not continuos at %s and %s patches"%(patch_name,name_contact)
                if face == 1:  # loking right side
                    # position_repeated_cp.append(dof[n - 1::m])
                    name_contact = self.geometry[patch_name]["patch_faces"].get(face)
                    B_compare = self.geometry[name_contact].get("B")
                    n_comp, m_comp = B_compare.shape[0], B_compare.shape[1]
                    dof_compare = get_P(B_compare)
                    assert np.all(dof_compare[::n_comp] == dof[n - 1::n]), "Not continuos at %s and %s patches" % (patch_name, name_contact)
                if face == 2:  # loking up
                    # position_repeated_cp.append(dof[-n:])
                    name_contact = self.geometry[patch_name]["patch_faces"].get(face)
                    B_compare = self.geometry[name_contact].get("B")
                    n_comp, m_comp = B_compare.shape[0], B_compare.shape[1]
                    dof_compare = get_P(B_compare)
                    assert np.all(dof_compare[:n_comp] == dof[-n:]), "Not continuos at %s and %s patches" % (patch_name, name_contact)
                if face == 3:  # loking left side
                    # position_repeated_cp.append(dof[::m])
                    name_contact = self.geometry[patch_name]["patch_faces"].get(face)
                    B_compare = self.geometry[name_contact].get("B")
                    n_comp, m_comp = B_compare.shape[0], B_compare.shape[1]
                    dof_compare = get_P(B_compare)
                    assert np.all(dof_compare[n_comp - 1::n_comp] == dof[::n]), "Not continuos at %s and %s patches" % (patch_name, name_contact)
