from dataclasses import dataclass
import numpy as np
from datetime import datetime
from coopstructs.vectors import Vector2
from typing import Optional

@dataclass
class PeriodObj:
    grid: np.ndarray
    start_datetime: Optional[datetime] = None
    end_datetime: Optional[datetime] = None

@dataclass
class GridDefinition:
    shape: Vector2
    size: Vector2
    origin: Vector2