from mpl_toolkits.mplot3d.axes3d import Axes3D


class TernaryAxes3D(Axes3D):
    name = 'ternary3d'

    def __init__(self, *args, ternary_scale=1.0, **kwargs):
        kwargs.setdefault('auto_add_to_figure', False)
        self.ternary_scale = ternary_scale
        super().__init__(*args, **kwargs)
