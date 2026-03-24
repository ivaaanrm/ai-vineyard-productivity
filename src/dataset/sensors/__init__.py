from typing import Callable, Dict
from .base import *  # noqa: F403
from .sentinel1 import *  # noqa: F403
from .sentinel2 import *  # noqa: F403
from .sentinel3 import *  # noqa: F403

from .sentinel1 import S1_CALCULATORS
from .sentinel1.indices import preprocess_sar
from .sentinel2 import S2_CALCULATORS
from .sentinel3 import S3_CALCULATORS

ALL_CALCULATORS = S2_CALCULATORS + S1_CALCULATORS + S3_CALCULATORS

# Sensor name → preprocessing function applied before index computation.
# Add an entry here when a new sensor requires calibration or filtering.
SENSOR_PREPROCESSORS: Dict[str, Callable] = {
    "SENTINEL-1": preprocess_sar,
}
