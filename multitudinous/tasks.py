from enum import Enum

class Task(Enum):
    """
    The task to perform with the model:
    - GRID: voxel grid prediction
    - CONES: cone prediction
    """
    GRID = "grid"
    CONES = "cones"
