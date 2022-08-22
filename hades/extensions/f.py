import astar
import numpy as np

from hades.generation.primitives import Point

astar.calculate_astar_path(np.array([[1, 2, 3], [4, 5, 6]]), Point(5, 6), Point(7, 1))
