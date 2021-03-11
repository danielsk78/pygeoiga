import traceback
import numpy as np
from pygeoiga.plot.nrbplotting_mpl import p_cpoints, p_knots, p_surface, create_figure, p_curve


class NURB(object):
    """
    create a nurb object
    """
    def __init__(self, cpoints, knots: list, weight=None, resolution=30, engine="python"):
        """
        m=n+p+1
        m means numbers of elements inside the knot vector
        n means numbers of elements inside the control points (just for that case not total control points)
        p means degree
        Args:
            cpoints:
            knots:
            weight:
            degree:
        """
        self.resolution = resolution
        self.knots = knots

        _shape = np.asarray(cpoints.shape)
        shape = _shape.copy()
        shape[-1] = _shape[-1]+1 #To include the weights in the last term
        self.B = np.ones((shape))

        if weight is not None:
            self.B[..., -1] = weight[..., -1]

        cpoints= np.asarray(cpoints)
        self.B[..., :-1] = cpoints

        self.model = None
        self.triangles = None
        self.surf_triangles = None
        self.engine = engine
        self._NURB = None
        self.create_model()

    @property
    def degree(self):
        return np.asarray([len(np.where(np.asarray(self.knots[x]) == 0.)[0]) - 1 for x in range(len(self.knots))])

    @property
    def shape(self):
        return self.B.shape

    @property
    def cpoints(self):
        return self.B[..., :-1]

    @property
    def weight(self):
        return self.B[..., -1, np.newaxis]

    @property
    def dim(self):
        return self.cpoints.shape[-1]

    @property
    def transpose(self):
        """
        Transpose u, and v parameters
        Returns:

        """
        self.knots = [self.knots[1], self.knots[0]]
        to_shape = self.shape[:self.dim]
        new_B = np.ones((to_shape[1], to_shape[0], self.dim+1))

        for i in range(self.dim):
            new_B[...,i] = self.cpoints[...,i].T
        new_B[...,-1] = self.weight[...,0].T
        self.B = new_B

    def create_model(self, engine=None):
        if engine is None:
            engine = self.engine

        if engine == "c++":
            try:
                import gempyExplicit
            except Exception:
                traceback.print_exc()
            if self.dim == 1:
                self.model = gempyExplicit.NURBS_Curve(self.degree[0],
                                                       self.knots[0],
                                                       self.cpoints,
                                                       self.weight,
                                                       self.resolution,
                                                       engine="auto")
                return self.dim
            elif self.dim == 2:
                self.model, self.triangles = gempyExplicit.NURBS_Surface(self.degree[0], self.degree[1],
                                                         self.knots[0],self.knots[1],
                                                         self.cpoints, self.weight,
                                                         self.resolution, self.resolution, engine="auto")
                return self.dim

            elif self.dim == 3:
                self.model, self.triangles, self.surf_triangles = gempyExplicit.NURBS_Volume(self.degree[0],
                                                                                             self.degree[1],
                                                                                             self.degree[2],
                                                                                             self.knots[0],
                                                                                             self.knots[1],
                                                                                             self.knots[2],
                                                                                             self.cpoints,
                                                                                             self.weight,
                                                                                             self.resolution,
                                                                                             self.resolution,
                                                                                             self.resolution,
                                                                                             engine="auto")
                return self.dim
            else:
                print(self.dim)
                return TypeError

        elif engine == "python":
            try:
                from pygeoiga.engine import NURB_construction
            except:
                traceback.print_exc()

            self.model = NURB_construction(list(self.knots),
                                           self.cpoints,
                                           self.resolution,
                                           self.weight)

            return self.dim

        elif engine == "igakit":
            try:
                from igakit.nurbs import NURBS
            except Exception:
                traceback.print_exc()
            self._NURB = NURBS(knots=list(self.knots), control=self.B)#, weights=self.weight)

        else:
            print(self.engine, "Not available")
            return TypeError


    def copy(self):
        """
        Return a exact copy of the actual class
        Returns:
        """
        nrb = NURB.__new__(type(self))
        nrb.resolution = self.resolution
        nrb.shape = self.shape
        nrb.cpoints = self.cpoints
        nrb.knots = self.knots
        nrb.dim = self.dim
        nrb.B = self.B
        nrb.weight = self.weight
        nrb.model = self.model
        nrb.triangles = self.triangles
        nrb.surf_triangles=self.surf_triangles
        nrb.engine = self.engine
        nrb._NURB = self._NURB

        return nrb

    def knot_insert(self, knot_ins: list, direction:int = 0, leave=True):
        """
        Refinement by knot insertion
        Args:
            knot_ins: list of knots to insert
            direction: parametric direction to insert them
            leave: Show or hide progress bar after completion

        Returns:

        """
        from pygeoiga.nurb.refinement import knot_insertion
        self.B, self.knots = knot_insertion(self.B, self.degree, self.knots, knot_ins, direction, leave=leave)

    def degree_elevate(self, times: int =1, direction: int = 0):
        """
        Refinement by degree elevation
        Args:
            times:

        Returns:

        """
        from pygeoiga.nurb.refinement import degree_elevation
        self.B, self.knots = degree_elevation(self.B, self.knots, times=times, direction=direction)

    def get_basis_function(self, direction:int =0, resolution = None):
        """
        Obtain the basis functions and derivatives of the knot vector at desired direction
        Args:
            direction: parametric direction of knot vector
            resolution: amount of points to form the basis function
        Returns:
        """

        from pygeoiga.engine.NURB_engine import basis_function_array_nurbs
        N, dN = basis_function_array_nurbs(self.knots[direction],
                                           self.degree[direction],
                                           resolution if resolution is not None else self.resolution,
                                           self.weight)
        return N, dN


    def plot(self, ax=None, kwargs_surface={}, kwargs_knots={}, kwargs_cpoints={}):
        if ax is None:
            fig, ax = create_figure("2d", figsize=(10, 20))

        ax = self.plot_surface(ax, **kwargs_surface)
        ax = self.plot_knots(ax, **kwargs_knots)
        ax = self.plot_cpoints(ax, **kwargs_cpoints)
        return ax

    def plot_knots(self, ax=None, **kwargs_knots):
        if ax is None:
            fig, ax = create_figure("2d", figsize=(10, 20))

        ax = p_knots(self.knots,
                     self.B,
                     weight=self.weight,
                     ax=ax,
                     **kwargs_knots)
        return ax

    def plot_cpoints(self, ax=None, **kwargs_cpoints):
        if ax is None:
            fig, ax = create_figure("2d", figsize=(10, 20))
        ax = p_cpoints(self.B,
                       ax=ax,
                       **kwargs_cpoints)
        return ax

    def plot_surface(self, ax=None, **kwargs_surface):
        if ax is None:
            fig, ax = create_figure("2d", figsize=(10, 20))
        ax = p_surface(self.knots,
                       self.B,
                       weight=self.weight,
                       ax=ax,
                       **kwargs_surface)
        return ax

    def plot_curve(self, ax = None, **kwargs_curve):
        if ax is None:
            fig, ax = create_figure("2d", figsize=(10, 20))
        ax = p_curve(self.knots,
                     self.B,
                     weight=self.weight,
                     ax=ax,
                     **kwargs_curve)
        return ax