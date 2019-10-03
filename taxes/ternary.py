from collections import OrderedDict

import numpy as np

from matplotlib import cbook
from matplotlib import docstring
from matplotlib.axes import Axes
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.axis as maxis
from .spines import Spine
from .transforms import (
    TernaryTransform, VerticalTernaryTransform,
    BarycentricTransform, TernaryScaleTransform)
from .axis.taxis import TAxis
from .axis.raxis import RAxis
from .axis.laxis import LAxis


def xy2brl(x, y, s=1.0):
    x = np.asarray(x)
    y = np.asarray(y)
    s = np.asarray(s)
    b = s * (x - y / np.sqrt(3.0))
    r = s * (y / np.sqrt(3.0) * 2.0)
    l = s * (1.0 - x - y / np.sqrt(3.0))
    return b, r, l


def _determine_anchor(angle0, angle1):
    """Determine the tick-label alignments from the spine and the tick angles.

    Parameters
    ----------
    angle0 : float
        Spine angle in radian.
    angle1 : float
        Tick angle in radian.

    Returns
    -------
    ha : str
        Horizontal alignment.
    va : str
        Vertical alignment.
    """
    if angle0 < 0.0:
        a0 = angle0 + 180.0
    else:
        a0 = angle0

    if a0 < 30.0:
        if angle1 < a0:
            return 'center', 'top'
        else:
            return 'center', 'bottom'
    elif 30.0 <= a0 < 150.0:
        if angle1 < a0 - 180.0:
            return 'right', 'center_baseline'
        elif a0 - 180.0 <= angle1 < 30.0:
            return 'left', 'center_baseline'
        elif 30.0 <= angle1 < a0:
            return 'left', 'baseline'
        elif a0 <= angle1 < 150.0:
            return 'right', 'baseline'
        else:
            return 'right', 'center_baseline'
    elif 150.0 <= a0:
        if angle1 < a0 - 180.0 or a0 <= angle1:
            return 'center', 'top'
        else:
            return 'center', 'bottom'


class TernaryAxesBase(Axes):
    def __init__(self, *args, ternary_scale=1.0, points=None, **kwargs):
        if points is None:
            # By default, regular upward triangle is created.
            # The bottom and the top of the triangle have 0.0 and 1.0,
            # respectively, as the *y* coordinate in the original `Axes`
            # coordinates.
            # The horizontal center of the triangle has 0.5 as the *x*
            # coordinate in the original `Axes` coordinates.
            # The other coordinates are given to make the regular triangle.
            corners = (
                (0.5, 1.0),
                (0.5 - 1.0 / np.sqrt(3.0), 0.0),
                (0.5 + 1.0 / np.sqrt(3.0), 0.0),
            )
        else:
            corners = points

        self.corners = np.asarray(corners)

        self.ternary_scale = ternary_scale
        super().__init__(*args, **kwargs)
        self.set_aspect('equal', adjustable='box', anchor='C')
        self.set_ternary_lim(
            0.0, ternary_scale, 0.0, ternary_scale, 0.0, ternary_scale)

    @property
    def clockwise(self):
        corners = self.transAxes.transform(self.corners)
        d0 = corners[1] - corners[0]
        d1 = corners[2] - corners[1]
        d = d0[0] * d1[1] - d1[0] * d0[1]
        return d < 0.0

    def set_figure(self, fig):
        self.viewTLim = mtransforms.Bbox.unit()
        self.viewLLim = mtransforms.Bbox.unit()
        self.viewRLim = mtransforms.Bbox.unit()
        super().set_figure(fig)

    def _get_axis_list(self):
        return (self.taxis, self.laxis, self.raxis)

    def _init_axis(self):
        self.xaxis = maxis.XAxis(self)
        self.yaxis = maxis.YAxis(self)

        self.taxis = TAxis(self)
        self.laxis = LAxis(self)
        self.raxis = RAxis(self)

        self.spines['bottom'].register_axis(self.taxis)
        self.spines['right' ].register_axis(self.laxis)
        self.spines['left'  ].register_axis(self.raxis)

        self._update_transScale()

    def _set_lim_and_transforms(self):
        super()._set_lim_and_transforms()
        transTernaryScale = TernaryScaleTransform(self.ternary_scale)
        transTLimits = mtransforms.BboxTransformFrom(
            mtransforms.TransformedBbox(self.viewTLim, self.transScale))
        transLLimits = mtransforms.BboxTransformFrom(
            mtransforms.TransformedBbox(self.viewLLim, self.transScale))
        transRLimits = mtransforms.BboxTransformFrom(
            mtransforms.TransformedBbox(self.viewRLim, self.transScale))

        taxis_transform = TernaryTransform(self.corners, 0)
        laxis_transform = TernaryTransform(self.corners, 1)
        raxis_transform = TernaryTransform(self.corners, 2)

        self._taxis_transform = transTLimits + taxis_transform + self.transAxes
        self._laxis_transform = transLLimits + laxis_transform + self.transAxes
        self._raxis_transform = transRLimits + raxis_transform + self.transAxes

        # For axis labels
        self._vertical_taxis_transform = VerticalTernaryTransform(self.transAxes, self.corners, 0)
        self._vertical_laxis_transform = VerticalTernaryTransform(self.transAxes, self.corners, 1)
        self._vertical_raxis_transform = VerticalTernaryTransform(self.transAxes, self.corners, 2)

        # For data

        # This should be called only once at the first time to define the
        # transformations between (b, r, l) and (x, y)
        corners_xy = self.transLimits.transform(self.corners)
        self._brl2xy_transform = transTernaryScale + BarycentricTransform(corners_xy)

        # Transform from the barycentric coordinates to the original
        # Axes coordinates
        self._ternary_axes_transform = self._brl2xy_transform + self.transLimits

    def get_taxis_transform(self, which='grid'):
        return self._taxis_transform

    def get_laxis_transform(self, which='grid'):
        return self._laxis_transform

    def get_raxis_transform(self, which='grid'):
        return self._raxis_transform

    def _get_axis_text_transform(self, pad_points, trans, which):
        if which == 'tick1':
            ps0 = trans.transform([[0.0, 0.0], [1.0, 0.0]])
            ps1 = trans.transform([[0.0, 0.0], [0.0, 1.0]])
        else:
            ps0 = trans.transform([[0.0, 1.0], [1.0, 1.0]])
            ps1 = trans.transform([[0.0, 1.0], [0.0, 0.0]])
        d0 = ps0[0] - ps0[1]
        d1 = ps1[0] - ps1[1]
        angle0 = np.rad2deg(np.arctan2(d0[1], d0[0]))
        angle1 = np.rad2deg(np.arctan2(d1[1], d1[0]))
        ha, va = _determine_anchor(angle0, angle1)
        x, y = d1 / np.linalg.norm(d1) * pad_points / 72.0
        return (trans +
                mtransforms.ScaledTranslation(x, y,
                                              self.figure.dpi_scale_trans),
                va, ha)

    def get_taxis_text1_transform(self, pad_points):
        trans = self.get_taxis_transform(which='tick1')
        return self._get_axis_text_transform(pad_points, trans, 'tick1')

    def get_taxis_text2_transform(self, pad_points):
        trans = self.get_taxis_transform(which='tick2')
        return self._get_axis_text_transform(pad_points, trans, 'tick2')

    def get_laxis_text1_transform(self, pad_points):
        trans = self.get_laxis_transform(which='tick1')
        return self._get_axis_text_transform(pad_points, trans, 'tick1')

    def get_laxis_text2_transform(self, pad_points):
        trans = self.get_laxis_transform(which='tick2')
        return self._get_axis_text_transform(pad_points, trans, 'tick2')

    def get_raxis_text1_transform(self, pad_points):
        trans = self.get_raxis_transform(which='tick1')
        return self._get_axis_text_transform(pad_points, trans, 'tick1')

    def get_raxis_text2_transform(self, pad_points):
        trans = self.get_raxis_transform(which='tick2')
        return self._get_axis_text_transform(pad_points, trans, 'tick2')

    def _gen_axes_patch(self):
        return mpatches.Polygon(self.corners)

    def _gen_axes_spines(self, locations=None, offset=0.0, units='inches'):
        # Use `Spine` in `taxes`
        spines = OrderedDict((side, Spine.linear_spine(self, side))
                             for side in ['left', 'right', 'bottom', 'top'])
        spines['top'].set_visible(False)  # Not to make an unexpected dot
        return spines

    def get_taxis(self):
        """Return the TAxis instance"""
        return self.taxis

    def get_laxis(self):
        """Return the LAxis instance"""
        return self.laxis

    def get_raxis(self):
        """Return the RAxis instance"""
        return self.raxis

    def cla(self):
        self.set_tlim(0.0, self.ternary_scale)
        self.set_llim(0.0, self.ternary_scale)
        self.set_rlim(0.0, self.ternary_scale)
        super().cla()

    @docstring.dedent_interpd
    def grid(self, b=None, which='major', axis='both', **kwargs):
        """
        Configure the grid lines.

        Parameters
        ----------
        b : bool or None, optional
            Whether to show the grid lines. If any *kwargs* are supplied,
            it is assumed you want the grid on and *b* will be set to True.

            If *b* is *None* and there are no *kwargs*, this toggles the
            visibility of the lines.

        which : {'major', 'minor', 'both'}, optional
            The grid lines to apply the changes on.

        axis : {'both', 't', 'l', 'r'}, optional
            The axis to apply the changes on.

        **kwargs : `.Line2D` properties
            Define the line properties of the grid, e.g.::

                grid(color='r', linestyle='-', linewidth=2)

            Valid *kwargs* are

        %(_Line2D_docstr)s

        Notes
        -----
        The axis is drawn as a unit, so the effective zorder for drawing the
        grid is determined by the zorder of each axis, not by the zorder of the
        `.Line2D` objects comprising the grid.  Therefore, to set grid zorder,
        use `.set_axisbelow` or, for more control, call the
        `~matplotlib.axis.Axis.set_zorder` method of each axis.
        """
        if len(kwargs):
            b = True
        cbook._check_in_list(['t', 'l', 'r', 'both'], axis=axis)
        if axis in ['t', 'both']:
            self.taxis.grid(b, which=which, **kwargs)
        if axis in ['l', 'both']:
            self.laxis.grid(b, which=which, **kwargs)
        if axis in ['r', 'both']:
            self.raxis.grid(b, which=which, **kwargs)

    def tick_params(self, axis='both', **kwargs):
        cbook._check_in_list(['t', 'l', 'r', 'both'], axis=axis)
        if axis in ['t', 'both']:
            bkw = dict(kwargs)
            bkw.pop('left', None)
            bkw.pop('right', None)
            bkw.pop('labelleft', None)
            bkw.pop('labelright', None)
            self.taxis.set_tick_params(**bkw)
        if axis in ['l', 'both']:
            rkw = dict(kwargs)
            rkw.pop('left', None)
            rkw.pop('right', None)
            rkw.pop('labelleft', None)
            rkw.pop('labelright', None)
            self.laxis.set_tick_params(**rkw)
        if axis in ['r', 'both']:
            lkw = dict(kwargs)
            lkw.pop('left', None)
            lkw.pop('right', None)
            lkw.pop('labelleft', None)
            lkw.pop('labelright', None)
            self.raxis.set_tick_params(**lkw)

    def _create_bbox_from_ternary_lim(self):
        tmin, tmax = self.get_tlim()
        lmin, lmax = self.get_llim()
        rmin, rmax = self.get_rlim()
        points = [[tmax, lmin, rmin], [tmin, lmax, rmin], [tmin, lmin, rmax]]
        points = self._brl2xy_transform.transform(points)
        bbox = mtransforms.Bbox.unit()
        bbox.update_from_data_xy(points, ignore=True)
        return bbox

    def set_ternary_lim(self, tmin, tmax, lmin, lmax, rmin, rmax, *args, **kwargs):
        """

        Notes
        -----
        xmin, xmax : holizontal limits of the triangle
        ymin, ymax : bottom and top of the triangle
        """
        t = tmax + lmin + rmin
        l = tmin + lmax + rmin
        r = tmin + lmin + rmax
        s = self.ternary_scale
        tol = 1e-12
        if (abs(t - s) > tol) or (abs(l - s) > tol) or (abs(r - s) > tol):
            raise ValueError(t, l, r, s)

        boxin = self._create_bbox_from_ternary_lim()

        self.set_tlim(tmin, tmax)
        self.set_llim(lmin, lmax)
        self.set_rlim(rmin, rmax)

        boxout = self._create_bbox_from_ternary_lim()

        trans = mtransforms.BboxTransform(boxin, boxout)

        xmin, xmax = self.get_xlim()
        ymin, ymax = self.get_ylim()
        points = [[xmin, ymin], [xmax, ymax]]
        ((xmin, ymin), (xmax, ymax)) = trans.transform(points)

        self.set_xlim(xmin, xmax)
        self.set_ylim(ymin, ymax)

    def set_ternary_min(self, tmin, lmin, rmin, *args, **kwargs):
        s = self.ternary_scale
        tmax = s - lmin - rmin
        lmax = s - rmin - tmin
        rmax = s - tmin - lmin
        self.set_ternary_lim(tmin, tmax, lmin, lmax, rmin, rmax, *args, **kwargs)

    def set_ternary_max(self, tmax, lmax, rmax, *args, **kwargs):
        s = self.ternary_scale
        tmin = (s + tmax - lmax - rmax) * 0.5
        lmin = (s + lmax - rmax - tmax) * 0.5
        rmin = (s + rmax - tmax - lmax) * 0.5
        self.set_ternary_lim(tmin, tmax, lmin, lmax, rmin, rmax, *args, **kwargs)

    def get_tlim(self):
        return tuple(self.viewTLim.intervalx)

    def get_llim(self):
        return tuple(self.viewLLim.intervalx)

    def get_rlim(self):
        return tuple(self.viewRLim.intervalx)

    def set_tlim(self, tmin, tmax):
        self.viewTLim.intervalx = (tmin, tmax)
        self.stale = True
        return tmin, tmax

    def set_llim(self, lmin, lmax):
        self.viewLLim.intervalx = (lmin, lmax)
        self.stale = True
        return lmin, lmax

    def set_rlim(self, rmin, rmax):
        self.viewRLim.intervalx = (rmin, rmax)
        self.stale = True
        return rmin, rmax

    # Interactive manipulation

    def can_zoom(self):
        """
        Return *True* if this axes supports the zoom box button functionality.

        Ternary axes do not support zoom boxes.
        """
        return False

    def _set_view(self, view):
        super()._set_view(view)
        self._set_ternary_lim_from_xlim_and_ylim()

    # def _set_view_from_bbox(self, *args, **kwargs):
    #     super()._set_view_from_bbox(*args, **kwargs)
    #     self._set_ternary_lim_from_xlim_and_ylim()

    def drag_pan(self, *args, **kwargs):
        super().drag_pan(*args, **kwargs)
        self._set_ternary_lim_from_xlim_and_ylim()

    def _set_ternary_lim_from_xlim_and_ylim(self):
        """Set ternary lim from xlim and ylim in the interactive mode.

        This is called from
        - _set_view (`Home`, `Forward`, `Backward`)
        - _set_view_from_bbox (`Zoom-to-rectangle`)
        - drag_pan (`Pan/Zoom`)
        (https://matplotlib.org/users/navigation_toolbar.html)
        """
        # points = self._brl2xy_transform.inverted().transform(self.corners)
        points = self._ternary_axes_transform.inverted().transform(self.corners)

        tmax = points[0, 0]
        tmin = points[1, 0]
        lmax = points[1, 1]
        lmin = points[2, 1]
        rmax = points[2, 2]
        rmin = points[0, 2]

        self.set_tlim(tmin, tmax)
        self.set_llim(lmin, lmax)
        self.set_rlim(rmin, rmax)

    def opposite_ticks(self, b=None):
        if b:
            if self.taxis.get_label_position() != 'corner':
                self.taxis.set_label_position('top')
            if self.laxis.get_label_position() != 'corner':
                self.laxis.set_label_position('top')
            if self.raxis.get_label_position() != 'corner':
                self.raxis.set_label_position('top')
            self.taxis.set_ticks_position('top')
            self.laxis.set_ticks_position('top')
            self.raxis.set_ticks_position('top')
        else:
            if self.taxis.get_label_position() != 'corner':
                self.taxis.set_label_position('bottom')
            if self.laxis.get_label_position() != 'corner':
                self.laxis.set_label_position('bottom')
            if self.raxis.get_label_position() != 'corner':
                self.raxis.set_label_position('bottom')
            self.taxis.set_ticks_position('bottom')
            self.laxis.set_ticks_position('bottom')
            self.raxis.set_ticks_position('bottom')


class TernaryAxes(TernaryAxesBase):
    """
    A ternary graph projection, where the input dimensions are *t*, *l*, *r*.
    The plot starts from the bottom and goes anti-clockwise.
    """
    name = 'ternary'

    def get_tlabel(self):
        """
        Get the tlabel text string.
        """
        label = self.taxis.get_label()
        return label.get_text()

    def set_tlabel(self, tlabel, fontdict=None, labelpad=None, **kwargs):
        if labelpad is not None:
            self.taxis.labelpad = labelpad
        return self.taxis.set_label_text(tlabel, fontdict, **kwargs)

    def get_llabel(self):
        """
        Get the llabel text string.
        """
        label = self.laxis.get_label()
        return label.get_text()

    def set_llabel(self, llabel, fontdict=None, labelpad=None, **kwargs):
        if labelpad is not None:
            self.laxis.labelpad = labelpad
        return self.laxis.set_label_text(llabel, fontdict, **kwargs)

    def get_rlabel(self):
        """
        Get the rlabel text string.
        """
        label = self.raxis.get_label()
        return label.get_text()

    def set_rlabel(self, rlabel, fontdict=None, labelpad=None, **kwargs):
        if labelpad is not None:
            self.raxis.labelpad = labelpad
        return self.raxis.set_label_text(rlabel, fontdict, **kwargs)

    def text(self, t, l, r, s, *args, **kwargs):
        tlr = np.column_stack((t, l, r))
        x, y = self._brl2xy_transform.transform(tlr).T
        return super().text(x, y, s, *args, **kwargs)

    def text_xy(self, x, y, s, *args, **kwargs):
        return super().text(x, y, s, *args, **kwargs)

    def axtline(self, x=0, ymin=0, ymax=1, **kwargs):
        """
        Add a equi-t line across the axes.

        Parameters
        ----------
        x : scalar, optional, default: 0
            x position in data coordinates of the equi-t line.

        ymin : scalar, optional, default: 0
            Should be between 0 and 1, 0 being one end of the plot, 1 the
            other of the plot.

        ymax : scalar, optional, default: 1
            Should be between 0 and 1, 0 being one end of the plot, 1 the
            other of the plot.

        Returns
        -------
        line : :class:`~matplotlib.lines.Line2D`

        Other Parameters
        ----------------
        **kwargs
            Valid kwargs are :class:`~matplotlib.lines.Line2D` properties,
            with the exception of 'transform':

        %(_Line2D_docstr)s

        See also
        --------
        axtspan : Add a equi-t span across the axis.
        """
        if "transform" in kwargs:
            raise ValueError(
                "'transform' is not allowed as a kwarg;"
                + "axtline generates its own transform.")
        trans = self.get_taxis_transform(which='grid')
        l = mlines.Line2D([x, x], [ymin, ymax], transform=trans, **kwargs)
        self.add_line(l)
        return l

    def axlline(self, x=0, ymin=0, ymax=1, **kwargs):
        """
        Add a equi-l line across the axes.

        Parameters
        ----------
        x : scalar, optional, default: 0
            x position in data coordinates of the equi-l line.

        ymin : scalar, optional, default: 0
            Should be between 0 and 1, 0 being one end of the plot, 1 the
            other of the plot.

        ymax : scalar, optional, default: 1
            Should be between 0 and 1, 0 being one end of the plot, 1 the
            other of the plot.

        Returns
        -------
        line : :class:`~matplotlib.lines.Line2D`

        Other Parameters
        ----------------
        **kwargs
            Valid kwargs are :class:`~matplotlib.lines.Line2D` properties,
            with the exception of 'transform':

        %(_Line2D_docstr)s

        See also
        --------
        axlspan : Add a equi-l span across the axis.
        """
        if "transform" in kwargs:
            raise ValueError(
                "'transform' is not allowed as a kwarg;"
                + "axlline generates its own transform.")
        trans = self.get_laxis_transform(which='grid')
        l = mlines.Line2D([x, x], [ymin, ymax], transform=trans, **kwargs)
        self.add_line(l)
        return l

    def axrline(self, x=0, ymin=0, ymax=1, **kwargs):
        """
        Add a equi-r line across the axes.

        Parameters
        ----------
        x : scalar, optional, default: 0
            x position in data coordinates of the equi-r line.

        ymin : scalar, optional, default: 0
            Should be between 0 and 1, 0 being one end of the plot, 1 the
            other of the plot.

        ymax : scalar, optional, default: 1
            Should be between 0 and 1, 0 being one end of the plot, 1 the
            other of the plot.

        Returns
        -------
        line : :class:`~matplotlib.lines.Line2D`

        Other Parameters
        ----------------
        **kwargs
            Valid kwargs are :class:`~matplotlib.lines.Line2D` properties,
            with the exception of 'transform':

        %(_Line2D_docstr)s

        See also
        --------
        axrspan : Add a equi-r span across the axis.
        """
        if "transform" in kwargs:
            raise ValueError(
                "'transform' is not allowed as a kwarg;"
                + "axrline generates its own transform.")
        trans = self.get_raxis_transform(which='grid')
        l = mlines.Line2D([x, x], [ymin, ymax], transform=trans, **kwargs)
        self.add_line(l)
        return l

    def axtspan(self, xmin, xmax, ymin=0, ymax=1, **kwargs):
        """
        Add a span for the bottom coordinate.

        Parameters
        ----------
        xmin : float
               Lower limit of the top span in data units.
        xmax : float
               Upper limit of the top span in data units.
        ymin : float, optional, default: 0
               Lower limit of the span from end to end in relative
               (0-1) units.
        ymax : float, optional, default: 1
               Upper limit of the span from end to end in relative
               (0-1) units.

        Returns
        -------
        Polygon : `~matplotlib.patches.Polygon`

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.patches.Polygon` properties.

        %(Polygon)s

        See Also
        --------
        axlspan : Add a span for the left coordinate.
        axrspan : Add a span for the right coordinate.
        """
        trans = self.get_taxis_transform(which='grid')
        verts = (xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)
        p = mpatches.Polygon(verts, **kwargs)
        p.set_transform(trans)
        self.add_patch(p)
        return p

    def axlspan(self, xmin, xmax, ymin=0, ymax=1, **kwargs):
        """
        Add a span for the left coordinate.

        Parameters
        ----------
        xmin : float
               Lower limit of the left span in data units.
        xmax : float
               Upper limit of the left span in data units.
        ymin : float, optional, default: 0
               Lower limit of the span from end to end in relative
               (0-1) units.
        ymax : float, optional, default: 1
               Upper limit of the span from end to end in relative
               (0-1) units.

        Returns
        -------
        Polygon : `~matplotlib.patches.Polygon`

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.patches.Polygon` properties.

        %(Polygon)s

        See Also
        --------
        axbspan : Add a span for the bottom coordinate.
        axrspan : Add a span for the right coordinate.
        """
        trans = self.get_laxis_transform(which='grid')
        verts = (xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)
        p = mpatches.Polygon(verts, **kwargs)
        p.set_transform(trans)
        self.add_patch(p)
        return p

    def axrspan(self, xmin, xmax, ymin=0, ymax=1, **kwargs):
        """
        Add a span for the right coordinate.

        Parameters
        ----------
        xmin : float
               Lower limit of the right span in data units.
        xmax : float
               Upper limit of the right span in data units.
        ymin : float, optional, default: 0
               Lower limit of the span from end to end in relative
               (0-1) units.
        ymax : float, optional, default: 1
               Upper limit of the span from end to end in relative
               (0-1) units.

        Returns
        -------
        Polygon : `~matplotlib.patches.Polygon`

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.patches.Polygon` properties.

        %(Polygon)s

        See Also
        --------
        axbspan : Add a span for the bottom coordinate.
        axlspan : Add a span for the left coordinate.
        """
        trans = self.get_raxis_transform(which='grid')
        verts = (xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)
        p = mpatches.Polygon(verts, **kwargs)
        p.set_transform(trans)
        self.add_patch(p)
        return p

    def plot(self, t, l, r, *args, **kwargs):
        tlr = np.column_stack((t, l, r))
        x, y = self._brl2xy_transform.transform(tlr).T
        return super().plot(x, y, *args, **kwargs)

    def scatter(self, t, l, r, *args, **kwargs):
        tlr = np.column_stack((t, l, r))
        x, y = self._brl2xy_transform.transform(tlr).T
        return super().scatter(x, y, *args, **kwargs)

    def hexbin(self, t, l, r, *args, **kwargs):
        tlr = np.column_stack((t, l, r))
        x, y = self._brl2xy_transform.transform(tlr).T
        return super().hexbin(x, y, *args, **kwargs)

    def quiver(self, t, l, r, dt, dl, dr, *args, **kwargs):
        tlr = np.column_stack((t, l, r))
        x, y = self._brl2xy_transform.transform(tlr).T
        tlr = np.column_stack((t + dt, l + dl, r + dr))
        u, v = self._brl2xy_transform.transform(tlr).T
        u -= x
        v -= y
        return super().quiver(x, y, u, v, *args, **kwargs)

    def fill(self, t, l, r, *args, **kwargs):
        tlr = np.column_stack((t, l, r))
        x, y = self._brl2xy_transform.transform(tlr).T
        return super().fill(x, y, *args, **kwargs)

    def tricontour(self, t, l, r, *args, **kwargs):
        tlr = np.column_stack((t, l, r))
        x, y = self._brl2xy_transform.transform(tlr).T
        return super().tricontour(x, y, *args, **kwargs)

    def tricontourf(self, t, l, r, *args, **kwargs):
        tlr = np.column_stack((t, l, r))
        x, y = self._brl2xy_transform.transform(tlr).T
        return super().tricontourf(x, y, *args, **kwargs)

    def tripcolor(self, t, l, r, *args, **kwargs):
        tlr = np.column_stack((t, l, r))
        x, y = self._brl2xy_transform.transform(tlr).T
        return super().tripcolor(x, y, *args, **kwargs)

    def triplot(self, t, l, r, *args, **kwargs):
        tlr = np.column_stack((t, l, r))
        x, y = self._brl2xy_transform.transform(tlr).T
        tplot = self.plot
        self.plot = super().plot
        tmp = super().triplot(x, y, *args, **kwargs)
        self.plot = tplot
        return tmp
