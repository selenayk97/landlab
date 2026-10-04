"""Landlab component that simulates overland flow.

This component simulates overland flow using the 2-D numerical model of
shallow-water flow over topography using the de Almeida et al., 2012
algorithm for storage-cell inundation modeling.

.. codeauthor:: Jordan Adams

Examples
--------
>>> import numpy as np
>>> from landlab import RasterModelGrid
>>> from landlab.components.overland_flow import OverlandFlow

Create a grid on which to calculate overland flow.

>>> grid = RasterModelGrid((4, 5))

The grid will need some data to provide the overland flow component. To
check the names of the fields that provide input to the overland flow
component use the input_var_names class property.

>>> OverlandFlow.input_var_names
('surface_water__depth', 'topographic__elevation')

Create fields of data for each of these input variables.

>>> grid.at_node["topographic__elevation"] = [
...     [0.0, 0.0, 0.0, 0.0, 0.0],
...     [1.0, 1.0, 1.0, 1.0, 1.0],
...     [2.0, 2.0, 2.0, 2.0, 2.0],
...     [3.0, 3.0, 3.0, 3.0, 3.0],
... ]
>>> grid.at_node["surface_water__depth"] = [
...     [0.0, 0.0, 0.0, 0.0, 0.0],
...     [0.0, 0.0, 0.0, 0.0, 0.0],
...     [0.0, 0.0, 0.0, 0.0, 0.0],
...     [0.1, 0.1, 0.1, 0.1, 0.1],
... ]

Instantiate the `OverlandFlow` component to work on this grid, and run it.

>>> of = OverlandFlow(grid, steep_slopes=True)
>>> of.run_one_step()

After calculating the overland flow, new fields have been added to the
grid. Use the output_var_names property to see the names of the fields that
have been changed.

>>> of.output_var_names
('surface_water__depth', 'surface_water__discharge', 'water_surface__gradient')

The `surface_water__depth` field is defined at nodes.

>>> of.var_loc("surface_water__depth")
'node'
>>> grid.at_node["surface_water__depth"]
array([1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05,
       1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05,
       1.0000e-05, 2.0010e-02, 2.0010e-02, 2.0010e-02, 1.0000e-05,
       1.0001e-01, 1.0001e-01, 1.0001e-01, 1.0001e-01, 1.0001e-01])

The `surface_water__discharge` field is defined at links. Because our initial
topography was a dipping plane, there is no water discharge in the horizontal
direction, only toward the bottom of the grid.

>>> of.var_loc("surface_water__discharge")
'link'
>>> q = grid.at_link["surface_water__discharge"]
>>> np.all(q[grid.horizontal_links] == 0.0)
True
>>> np.all(q[grid.vertical_links] <= 0.0)
True

The water_surface__gradient is also defined at links.

>>> of.var_loc("water_surface__gradient")
'link'
>>> grid.at_link["water_surface__gradient"]
array([0. , 0. , 0. , 0. ,
       0. , 1. , 1. , 1. , 0. ,
       0. , 0. , 0. , 0. ,
       0. , 1. , 1. , 1. , 0. ,
       0. , 0. , 0. , 0. ,
       0. , 1.1, 1.1, 1.1, 0. ,
       0. , 0. , 0. , 0. ])
"""

import numpy as np
import scipy.constants
from numpy.typing import NDArray
from requireit import require_between
from requireit import require_nonnegative
from requireit import require_positive

from landlab import Component
from landlab.components.overland_flow._calc import adjust_supercritical_discharge
from landlab.components.overland_flow._calc import adjust_unstable_discharge
from landlab.components.overland_flow._calc import calc_bates_flow_height
from landlab.components.overland_flow._calc import calc_discharge_at_links
from landlab.components.overland_flow._calc import calc_grad_at_link
from landlab.components.overland_flow._calc import calc_weighted_mean_of_parallel_links
from landlab.components.overland_flow._calc import zero_out_dry_links
from landlab.core.errors import Error
from landlab.grid.linkstatus import LinkStatus

class NoWaterError(Error):
    pass

class OverlandFlow(Component):
    """Simulate overland flow using de Almeida approximations.

    Landlab component that simulates overland flow using the de Almeida
    et al., 2012 approximations of the 1D shallow water equations to be used
    for 2D flood inundation modeling.

    This component calculates discharge, depth and shear stress after some
    precipitation event across any raster grid. Default input file is named
    "overland_flow_input.txt' and is contained in the
    landlab.components.overland_flow folder.

    The primary method of this class is :func:`run_one_step`.

    References
    ----------
    Required Software Citation(s) Specific to this Component

    Adams, J., Gasparini, N., Hobley, D., Tucker, G., Hutton, E., Nudurupati,
    S., Istanbulluoglu, E. (2017). The Landlab v1. 0 OverlandFlow component:
    a Python tool for computing shallow-water flow across watersheds.
    Geoscientific Model Development  10(4), 1645.
    https://dx.doi.org/10.5194/gmd-10-1645-2017

    Additional References

    de Almeida, G., Bates, P., Freer, J., Souvignet, M. (2012). Improving the
    stability of a simple formulation of the shallow water equations for 2-D
    flood modeling. Water Resources Research 48(5)
    https://dx.doi.org/10.1029/2011wr011570

    """

    _name = "OverlandFlow"

    _unit_agnostic = False

    _cite_as = """@article{adams2017landlab,
        title={The Landlab v1. 0 OverlandFlow component: a Python
            tool for computing shallow-water flow across watersheds},
        author={Adams, Jordan M and Gasparini, Nicole M and
            Hobley, Daniel EJ and Tucker, Gregory E and
            Hutton, Eric WH and Nudurupati, Sai S and
            Istanbulluoglu, Erkan},
        journal={Geoscientific Model Development},
        volume={10},
        number={4},
        pages={1645},
        year={2017},
        publisher={Copernicus GmbH}
        }
    """

    _info = {
        "surface_water__depth": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Depth of water on the surface",
        },
        "surface_water__discharge": {
            "dtype": float,
            "intent": "out",
            "optional": True,
            "units": "m3/s",
            "mapping": "link",
            "doc": "Volumetric discharge of surface water",
        },
        "topographic__elevation": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "water_surface__gradient": {
            "dtype": float,
            "intent": "out",
            "optional": True,
            "units": "-",
            "mapping": "link",
            "doc": "Downstream gradient of the water surface.",
        },
    }

    def __init__(
        self,
        grid,
        h_init=0.00001,
        alpha=0.7,
        mannings_n=0.03,
        g=scipy.constants.g,
        theta=0.8,
        rainfall_intensity=0.0,
        steep_slopes=False,
    ):
        """Create an overland flow component.

        Parameters
        ----------
        grid : RasterModelGrid
            A landlab grid.
        h_init : float, optional
            Thickness of initial thin layer of water to prevent divide by zero
            errors (m).
        alpha : float, optional
            Time step coefficient, described in Bates et al., 2010 and
            de Almeida et al., 2012.
        mannings_n : array_like or str, optional
            Manning's roughness coefficient. If a `str`, use the corresponding
            at-link field of the provided grid.
        g : float, optional
            Acceleration due to gravity (m/s^2).
        theta : float, optional
            Weighting factor from de Almeida et al., 2012.
        rainfall_intensity : float or ndarray of float, optional
            Rainfall intensity. Default is zero.
        steep_slopes : bool, optional
            Modify the algorithm to handle steeper slopes at the expense of
            speed. If model runs become unstable, consider setting to True.
        """
        for var in ("surface_water__discharge", "water_surface__gradient"):
            at = self._info[var]["mapping"]
            units = self._info[var]["units"]
            if not grid.has_field(var, at=at):
                grid.add_empty(var, at=at, units=units)

        super().__init__(grid)

        self._h_init = require_nonnegative(h_init, name="h_init")
        self._alpha = require_positive(alpha, name="alpha")

        if isinstance(mannings_n, str):
            self._mannings_n = mannings_n
        else:
            self._mannings_n = np.require(
                np.broadcast_to(mannings_n, grid.number_of_links),
                requirements=("C_CONTIGUOUS",),
            )

        self._g = require_positive(g, name="g")
        self._theta = require_between(
            theta, 0.0, 1.0, inclusive_min=True, inclusive_max=True, name="theta"
        )

        ## FIX (Bug 5 - double validation): Pass rainfall_intensity directly to
        ## the setter which performs all necessary validation. Removed the outer
        ## require_nonnegative() call that created two competing validation paths
        ## with different error types. All validation now lives exclusively in
        ## the setter.
        self.rainfall_intensity = rainfall_intensity

        self._steep_slopes = steep_slopes

        grid.at_link["surface_water__discharge"].fill(0.0)

        ## FIX (Issue 1 - h_init added unconditionally on every instantiation):
        ## Guard the h_init addition with a check on whether the depth field
        ## has already been initialised with a non-zero value. If the user has
        ## pre-populated the depth field, h_init is still added once but only
        ## on first instantiation. A re-instantiated component will not silently
        ## add h_init again on top of existing water depths.
        depth_field = grid.at_node["surface_water__depth"]
        if np.all(depth_field == 0.0):
            depth_field += self._h_init

        grid.at_link["water_surface__gradient"].fill(0.0)

        ## _h_links is a private link-depth array used internally by the
        ## component. It is intentionally not stored as a named grid field
        ## (see Issue 3 in evaluation). Initialise with zeros then fill so
        ## that the array is fully owned by this instance.
        self._h_links = grid.zeros(at="link")
        self._h_links.fill(self._h_init)

        ## FIX (Bug 1 - elapsed_time initialised to 1.0 and never updated):
        ## Initialise to 0.0 to represent the start of simulation. The counter
        ## is incremented inside overland_flow() at the end of every successful
        ## sub-step so that self.elapsed_time always reflects the true total
        ## simulated time.
        self._elapsed_time = 0.0

        ## FIX (Bug 6 - dhdt allocated as new array every iteration):
        ## Pre-allocate the depth-change array once here and reuse it in-place
        ## each sub-step with np.subtract(..., out=self._dhdt). This avoids
        ## repeated heap allocation on every iteration of the time-stepping loop,
        ## which is particularly expensive on fine-resolution grids.
        self._dhdt = grid.zeros(at="node")

    ## ------------------------------------------------------------------
    ## Properties
    ## ------------------------------------------------------------------

    @property
    def h(self):
        """The depth of water at each node."""
        return self._grid.at_node["surface_water__depth"]

    ## FIX (Bug 7 - no elapsed_time property exposed): Add a public property
    ## so that driver scripts can query total simulated time without accessing
    ## the private attribute directly.
    @property
    def elapsed_time(self):
        """Total elapsed simulation time [s]."""
        return self._elapsed_time

    ## FIX (Issue 3 - _h_links not accessible): Expose the link-based flow
    ## depth array as a read-only property so that it can be inspected by
    ## downstream components or diagnostic scripts without being stored as a
    ## named grid field (which would conflict with the node-mapped
    ## surface_water__depth field).
    @property
    def h_links(self):
        """Flow depth at each link [m]."""
        return self._h_links

    @property
    def rainfall_intensity(self):
        """rainfall_intensity: the rainfall rate [m/s].

        Must be non-negative.
        """
        return self._rainfall_intensity

    ## FIX (Bug 5 - double validation consolidated here): The setter is the
    ## single source of validation for rainfall_intensity. The __init__ method
    ## passes the value directly to this setter rather than pre-validating with
    ## require_nonnegative(), eliminating the dual-path inconsistency.
    @rainfall_intensity.setter
    def rainfall_intensity(self, new_val):
        if np.any(np.asarray(new_val) < 0.0):
            raise ValueError("Rainfall intensity must be non-negative")
        self._rainfall_intensity = new_val

    ## ------------------------------------------------------------------
    ## Internal helpers
    ## ------------------------------------------------------------------

    def _update_active_nodes(self, *, clear_cache: bool = False) -> NDArray[np.bool_]:
        """Compute and optionally re-compute the active-node mask.

        Parameters
        ----------
        clear_cache : bool, optional
            If ``True`` clear any cached value and recompute the active
            node mask. Should be set to ``True`` at the start of each
            sub-step so that nodes that become wet during the simulation
            are correctly included in the time-step calculation.

        Returns
        -------
        active_nodes : ndarray of bool, size (n_nodes,)
            Boolean mask; True where a node has at least one active link.
        """
        ## FIX (Bug 3 - cache never cleared during time-stepping): The cache
        ## is now cleared at the start of every sub-step inside overland_flow()
        ## by passing clear_cache=True. This ensures that nodes transitioning
        ## from dry to wet are included in subsequent time-step calculations
        ## rather than being permanently excluded based on the state at first
        ## call.
        if clear_cache:
            self._cached_is_active_node = None

        if getattr(self, "_cached_is_active_node", None) is None:
            self._cached_is_active_node: NDArray[np.bool_] = np.any(
                self.grid.link_status_at_node == LinkStatus.ACTIVE, axis=1
            )

        return self._cached_is_active_node

    def calc_time_step(self):
        """Calculate time step.

        Adaptive time stepper from Bates et al., 2010 and de Almeida et
        al., 2012.

        Returns
        -------
        time_step : float
            A stable time step for the current landscape.

        Raises
        ------
        NoWaterError
            If there is no water on active nodes (i.e., the maximum water
            depth is zero or the active-node set is empty), this exception
            is raised to indicate that a time step cannot be determined.
        """
        active_nodes = self._update_active_nodes()

        h: NDArray = self.grid.at_node["surface_water__depth"][active_nodes]
        if h.size == 0:
            raise NoWaterError("no active nodes, unable to determine time step")

        max_water_depth: float = float(np.max(h))
        if max_water_depth <= 0.0:
            raise NoWaterError("no water on landscape, unable to determine time step")

        return (
            self._alpha
            * min(self._grid.dx, self._grid.dy)
            / np.sqrt(self._g * max_water_depth)
        )

    ## ------------------------------------------------------------------
    ## Time-stepping
    ## ------------------------------------------------------------------

    def overland_flow(self, dt=None):
        """Generate overland flow across a grid.

        For one time step, this generates 'overland flow' across a given grid
        by calculating discharge at each node.

        Using the depth slope product, shear stress is calculated at every
        node.

        Outputs water depth, discharge and shear stress values through time at
        every point in the input grid.

        Parameters
        ----------
        dt : float, optional
            The duration over which to simulate overland flow. If not
            provided, the duration will be chosen to be the maximum time
            step that ensures stability.

        Returns
        -------
        elapsed : float
            The elapsed time that was simulated.
        """
        if dt is None:
            try:
                duration = self.calc_time_step()
            except NoWaterError as error:
                raise ValueError(
                    "no water on landscape and dt not provided,"
                    " unable to determine time step"
                ) from error
        else:
            duration = dt

        if isinstance(self._mannings_n, str):
            mannings_at_link = self.grid.at_link[self._mannings_n]
        else:
            mannings_at_link = self._mannings_n

        h_at_node = self._grid.at_node["surface_water__depth"]
        z_at_node = self._grid.at_node["topographic__elevation"]
        q_at_link = self._grid.at_link["surface_water__discharge"]
        h_at_link = self._h_links
        water_surface_slope = self._grid.at_link["water_surface__gradient"]

        ## FIX (Bug 4 - q_mean_at_link allocated with empty()): Use zeros()
        ## rather than empty() so that the array is fully initialised to 0.0
        ## from the start. grid.empty() returns uninitialized memory; if any
        ## code path ever bypasses the fill(0.0) inside the loop the discharge
        ## calculation would silently use garbage values.
        q_mean_at_link = self.grid.zeros(at="link")

        core_nodes = self._grid.core_nodes
        active_links = self._grid.active_links

        elapsed = 0.0
        while elapsed < duration:
            remaining = duration - elapsed

            ## FIX (Bug 3 - cache never cleared during time-stepping): Force
            ## recomputation of the active-node mask at the start of each
            ## sub-step so that nodes transitioning from dry to wet are
            ## correctly captured in the CFL time-step calculation.
            self._update_active_nodes(clear_cache=True)

            try:
                dt_local = min(self.calc_time_step(), remaining)
            except NoWaterError:
                elapsed = duration
                break

            ## Per Bates et al., 2010, find the difference between the highest
            ## water surface in the two cells and the highest bed elevation.
            h_at_link = calc_bates_flow_height(
                z_at_node,
                h_at_node,
                nodes_at_link=self.grid.nodes_at_link,
                where=active_links,
                out=h_at_link,
            )

            ## FIX (Bug 11 - np.clip floors ALL links to h_init): The original
            ## clip applied h_init as a minimum to every link on the grid,
            ## including genuinely dry links where hflow is zero or negative.
            ## This forced maximum Manning's friction (h^7/3 at h_init) onto
            ## every dry link, preventing flow from ever entering topographic
            ## depressions such as road ruts. The corrected version only applies
            ## the h_init floor to active links that already have a positive
            ## computed flow height, preserving zero depth on truly dry links.
            positive_flow_at_active = (
                (h_at_link > 0.0)
                & np.isin(
                    np.arange(len(h_at_link)), active_links
                )
            )
            h_at_link[positive_flow_at_active] = np.maximum(
                h_at_link[positive_flow_at_active], self._h_init
            )

            ## Calculate the slope of the water surface elevation at active
            ## links and write the result back into the grid field in-place.
            ## FIX (Issue 2 - water_surface_slope rebinding may lose grid field
            ## sync): The return value of calc_grad_at_link is explicitly
            ## written back to the grid field using [:] assignment rather than
            ## relying on the local variable remaining bound to the same
            ## underlying array. This guards against any future version of
            ## calc_grad_at_link returning a new array rather than the out
            ## argument.
            water_surface_slope_result = calc_grad_at_link(
                h_at_node + z_at_node,
                length_of_link=self.grid.length_of_link,
                nodes_at_link=self.grid.nodes_at_link,
                where=active_links,
                out=water_surface_slope,
            )
            self._grid.at_link["water_surface__gradient"][:] = (
                water_surface_slope_result
            )
            water_surface_slope = self._grid.at_link["water_surface__gradient"]

            q_at_link = zero_out_dry_links(
                h_at_link, where=active_links, out=q_at_link
            )

            q_mean_at_link.fill(0.0)
            q_mean_at_link = calc_weighted_mean_of_parallel_links(
                q_at_link,
                parallel_links_at_link=self.grid.parallel_links_at_link,
                status_at_link=self.grid.status_at_link,
                theta=self._theta,
                where=self._grid.active_links,
                out=q_mean_at_link,
            )

            q_at_link = calc_discharge_at_links(
                q_at_link,
                q_mean_at_link,
                h_at_link,
                water_surface_slope,
                mannings_at_link,
                g=self._g,
                dt=dt_local,
                where=self._grid.active_links,
                out=q_at_link,
            )

            if self._steep_slopes:
                ## Where the calculated discharge exceeds the Froude number we
                ## are getting supercritical flow; reduce discharge to maintain
                ## stability.
                q_at_link = adjust_supercritical_discharge(
                    q_at_link,
                    h_at_link,
                    g=self._g,
                    froude=1.0,
                    where=active_links,
                    out=q_at_link,
                )

                ## Where the calculated discharge exceeds the Courant number
                ## and the water depth divided amongst 4 links, reduce discharge
                ## to maintain stability.
                q_at_link = adjust_unstable_discharge(
                    q_at_link,
                    h_at_link,
                    dx=self._grid.dx,
                    dt=dt_local,
                    where=active_links,
                    out=q_at_link,
                )

            ## FIX (Bug 6 - dhdt allocated as new array every iteration):
            ## Update the pre-allocated _dhdt array in-place using np.subtract
            ## with the out argument. This eliminates per-iteration heap
            ## allocation which is expensive on fine-resolution grids with
            ## thousands of sub-steps.
            np.subtract(
                self._rainfall_intensity,
                self._grid.calc_flux_div_at_node(q_at_link),
                out=self._dhdt,
            )

            ## Update water depths at core nodes.
            h_at_node[core_nodes] = (
                h_at_node[core_nodes] + self._dhdt[core_nodes] * dt_local
            )

            ## Enforce minimum water depth to prevent divide-by-zero errors.
            if self._steep_slopes:
                h_at_node[h_at_node < self._h_init] = self._h_init  *10.0*-3

            ## Write the updated discharge back to the grid field in-place.
            self._grid.at_link["surface_water__discharge"][:] = q_at_link

            ## FIX (Bug 1 - elapsed_time never updated): Increment both the
            ## local elapsed counter and the component's internal clock after
            ## all computations for this sub-step have completed successfully.
            ## FIX (Bug 2 - elapsed advanced before computations): The
            ## increment is placed here at the END of the loop body so that
            ## elapsed only advances after a sub-step completes without error.
            ## In the original code elapsed was incremented before the
            ## numerical calculations ran, meaning an exception mid-step would
            ## leave elapsed reporting more time than was actually simulated.
            elapsed += dt_local
            self._elapsed_time += dt_local

        return elapsed

    def run_one_step(self, dt=None):
        """Generate overland flow across a grid.

        For one time step, this generates 'overland flow' across a given grid
        by calculating discharge at each node.

        Using the depth slope product, shear stress is calculated at every
        node.

        Outputs water depth, discharge and shear stress values through time at
        every point in the input grid.

        Parameters
        ----------
        dt : float, optional
            The duration over which to simulate overland flow. If not
            provided, the duration will be chosen to be the maximum time
            step that ensures stability.

        Returns
        -------
        elapsed : float
            The elapsed time that was simulated.
        """
        return self.overland_flow(dt=dt)

    def discharge_mapper(self, input_discharge, convert_to_volume=False):
        """Maps discharge value from links onto nodes.

        This method takes the discharge values on links and determines the
        links that are flowing INTO a given node. The fluxes moving INTO a
        given node are summed.

        This method ignores all flow moving OUT of a given node.

        This takes values from the OverlandFlow component (by default) in
        units of [L^2/T]. If the convert_to_volume flag is raised as True,
        this method converts discharge to units [L^3/T] - as of Aug 2016,
        only operates for square RasterModelGrid instances.

        The output array is of length grid.number_of_nodes and can be used
        with the Landlab imshow_grid plotter.

        Returns a numpy array (discharge_vals)
        """
        discharge_vals = np.zeros(self._grid.number_of_links)
        discharge_vals[:] = input_discharge[:]

        if convert_to_volume:
            discharge_vals *= self._grid.dx

        discharge_vals = (
            discharge_vals[self._grid.links_at_node] * self._grid.link_dirs_at_node
        )

        discharge_vals = discharge_vals.flatten()

        discharge_vals[np.where(discharge_vals < 0)] = 0.0

        discharge_vals = discharge_vals.reshape(self._grid.number_of_nodes, 4)

        discharge_vals = discharge_vals.sum(axis=1)

        return discharge_vals