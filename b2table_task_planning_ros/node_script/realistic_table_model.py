import time
from typing import ClassVar, List, Union

import numpy as np
from plainmp.psdf import UnionSDF
from rpbench.articulated.world.utils import BoxSkeleton
from rpbench.utils import SceneWrapper
from skrobot.coordinates import CascadedCoords
from skrobot.viewers import PyrenderViewer, TrimeshSceneViewer


class JskTable(CascadedCoords):
    table_primitives: List[BoxSkeleton]
    TABLE_HEIGHT: ClassVar[float] = 0.7
    TABLE_DEPTH: ClassVar[float] = 0.8
    TABLE_WIDTH: ClassVar[float] = 1.24

    def __init__(self):
        super().__init__()
        plate_d = 0.05
        size = [self.TABLE_DEPTH, self.TABLE_WIDTH, plate_d]
        height = self.TABLE_HEIGHT
        plate = BoxSkeleton(size, pos=[0, 0, height - 0.5 * plate_d])

        leg_width = 0.11
        leg_depth = 0.05
        leg1 = BoxSkeleton(
            [leg_width, leg_depth, height],
            pos=[0.5 * size[0] - 0.5 * leg_width, 0.5 * size[1] - 0.5 * leg_width, height * 0.5],
        )
        leg1.rotate(np.pi * 0.25, "z")
        leg1.translate([-0.05, -0.0, 0])

        leg2 = BoxSkeleton(
            [leg_width, leg_depth, height],
            pos=[0.5 * size[0] - 0.5 * leg_width, -0.5 * size[1] + 0.5 * leg_width, height * 0.5],
        )
        leg2.rotate(-np.pi * 0.25, "z")
        leg2.translate([-0.05, -0.0, 0])

        leg3 = BoxSkeleton(
            [leg_width, leg_depth, height],
            pos=[-0.5 * size[0] + 0.5 * leg_width, -0.5 * size[1] + 0.5 * leg_width, height * 0.5],
        )
        leg3.rotate(+np.pi * 0.25, "z")
        leg3.translate([+0.05, -0.0, 0])

        leg4 = BoxSkeleton(
            [leg_width, leg_depth, height],
            pos=[-0.5 * size[0] + 0.5 * leg_width, 0.5 * size[1] - 0.5 * leg_width, height * 0.5],
        )
        leg4.rotate(-np.pi * 0.25, "z")
        leg4.translate([+0.05, -0.0, 0])

        primitive_list = [
            plate,
            leg1,
            leg2,
            leg3,
            leg4,
        ]
        self.table_primitives = primitive_list

    def visualize(self, viewer: Union[TrimeshSceneViewer, SceneWrapper]) -> None:
        for prim in self.table_primitives:
            viewer.add(prim.to_visualizable((200, 200, 200, 255)))

    def create_sdf(self) -> UnionSDF:
        return UnionSDF([p.to_plainmp_sdf() for p in self.table_primitives])

    def translate(self, *args, **kwargs):
        raise NotImplementedError("This method is deleted")

    def rotate(self, *args, **kwargs):
        raise NotImplementedError("This method is deleted")


if __name__ == "__main__":
    v = PyrenderViewer()
    table = JskTable()
    sdf = table.create_sdf()
    table.visualize(v)
    v.show()
    time.sleep(100)
