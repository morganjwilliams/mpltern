import numpy as np

import matplotlib.transforms as mtransforms
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpltern.ternary.transforms import (
        TernaryTransform, TernaryPerpendicularTransform,
        BarycentricTransform, TernaryScaleTransform)
from mpltern.ternary._axes import _create_corners
from mpltern.ternary.ternary_parser import _get_xy


class TernaryAxes3D(Axes3D):
    name = 'ternary3d'

    def __init__(self, *args, ternary_scale=1.0, corners=None, rotation=None,
                 **kwargs):
        kwargs.setdefault('auto_add_to_figure', False)
        self.ternary_scale = ternary_scale

        # Triangle corners in the original data coordinates
        self.corners_data = _create_corners(corners, rotation)
        sx = np.sqrt(3.0) * 0.5  # Scale for x
        xmin = -1.0 / np.sqrt(3.0)
        v = xmin * sx
        trans = mtransforms.Affine2D().from_values(sx, 0.0, 0.0, 1.0, -v, 0.0)
        # Triangle corners in the original ``Axes`` coordinates
        self.corners_axes = trans.transform(self.corners_data)

        self.viewTLim = mtransforms.Bbox.unit()
        self.viewLLim = mtransforms.Bbox.unit()
        self.viewRLim = mtransforms.Bbox.unit()

        super().__init__(*args, **kwargs)

        # As of Matplotlib 3.5.0, Axes3D does not support `set_aspect` other
        # than `auto` and therefore we need (1) to set the box aspect and the
        # data limits consistently.
        self.set_box_aspect((8.0 / np.sqrt(3.0), 4.0, 3.0))
        self.set_ternary_lim(
            0.0, ternary_scale, 0.0, ternary_scale, 0.0, ternary_scale)

    def _set_lim_and_transforms(self):
        super()._set_lim_and_transforms()
        transTernaryScale = TernaryScaleTransform(self.ternary_scale)
        transTLimits = mtransforms.BboxTransformFrom(
            mtransforms.TransformedBbox(self.viewTLim, self.transScale))
        transLLimits = mtransforms.BboxTransformFrom(
            mtransforms.TransformedBbox(self.viewLLim, self.transScale))
        transRLimits = mtransforms.BboxTransformFrom(
            mtransforms.TransformedBbox(self.viewRLim, self.transScale))

        corners_axes = self.corners_axes

        taxis_transform = TernaryTransform(corners_axes, 0)
        laxis_transform = TernaryTransform(corners_axes, 1)
        raxis_transform = TernaryTransform(corners_axes, 2)

        self._taxis_transform = transTLimits + taxis_transform + self.transAxes
        self._laxis_transform = transLLimits + laxis_transform + self.transAxes
        self._raxis_transform = transRLimits + raxis_transform + self.transAxes

        # For axis labels
        t_l_t = TernaryPerpendicularTransform(self.transAxes, corners_axes, 0)
        l_l_t = TernaryPerpendicularTransform(self.transAxes, corners_axes, 1)
        r_l_t = TernaryPerpendicularTransform(self.transAxes, corners_axes, 2)
        self._taxis_label_transform = t_l_t
        self._laxis_label_transform = l_l_t
        self._raxis_label_transform = r_l_t

        # From ternary coordinates to the original data coordinates
        self.transProjection = (transTernaryScale
                                + BarycentricTransform(self.corners_data))

        # From ternary coordinates to the original Axes coordinates
        self._ternary_axes_transform = self.transProjection + self.transLimits

        # From barycentric coordinates to the original Axes coordinates
        self.transAxesProjection = BarycentricTransform(self.corners_axes)

        # From barycentric coordinates to display coordinates
        self.transTernaryAxes = self.transAxesProjection + self.transAxes

    def cla(self):
        self.set_tlim(0.0, self.ternary_scale)
        self.set_llim(0.0, self.ternary_scale)
        self.set_rlim(0.0, self.ternary_scale)
        super().cla()
        xmin = -1.0 / np.sqrt(3.0)
        xmax = +1.0 / np.sqrt(3.0)
        self.set_xlim(xmin, xmax)
        self.set_ylim(0.0, 1.0)

    def _create_bbox_from_ternary_lim(self):
        tmin, tmax = self.get_tlim()
        lmin, lmax = self.get_llim()
        rmin, rmax = self.get_rlim()
        points = [[tmax, lmin, rmin], [tmin, lmax, rmin], [tmin, lmin, rmax]]
        points = self.transProjection.transform(points)
        bbox = mtransforms.Bbox.unit()
        bbox.update_from_data_xy(points, ignore=True)
        return bbox

    def set_ternary_lim(self, tmin, tmax, lmin, lmax, rmin, rmax):
        """
        Set the ternary-axes view limits.

        Parameters
        ----------
        tmin, tmax : float
            The lower and the upper bounds for the `t` axis.

        lmin, lmax : float
            The lower and the upper bounds for the `l` axis.

        rmin, rmax : float
            The lower and the upper bounds for the `r` axis.

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

        xmin, xmax = self.get_xlim3d()
        ymin, ymax = self.get_ylim3d()
        points = [[xmin, ymin], [xmax, ymax]]
        ((xmin, ymin), (xmax, ymax)) = trans.transform(points)

        self.set_xlim3d(xmin, xmax)
        self.set_ylim3d(ymin, ymax)

    def set_ternary_min(self, tmin, lmin, rmin):
        s = self.ternary_scale
        tmax = s - lmin - rmin
        lmax = s - rmin - tmin
        rmax = s - tmin - lmin
        self.set_ternary_lim(tmin, tmax, lmin, lmax, rmin, rmax)

    def set_ternary_max(self, tmax, lmax, rmax):
        s = self.ternary_scale
        tmin = (s + tmax - lmax - rmax) * 0.5
        lmin = (s + lmax - rmax - tmax) * 0.5
        rmin = (s + rmax - tmax - lmax) * 0.5
        self.set_ternary_lim(tmin, tmax, lmin, lmax, rmin, rmax)

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

    def plot(self, *args, **kwargs):
        trans = kwargs.pop('transform', None)
        this, args = args[:3], args[3:]
        x, y, kwargs['transform'] = _get_xy(self, this, trans)
        args = (x, y, *args)
        return super().plot(*args, **kwargs)
