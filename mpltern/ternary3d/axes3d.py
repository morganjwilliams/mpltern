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

    def plot(self, *args, **kwargs):
        trans = kwargs.pop('transform', None)
        this, args = args[:3], args[3:]
        x, y, kwargs['transform'] = _get_xy(self, this, trans)
        args = (x, y, *args)
        return super().plot(*args, **kwargs)
