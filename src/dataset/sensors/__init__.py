from typing import Callable, Dict
from .base import *  # noqa: F403
from .sentinel1 import *  # noqa: F403
from .sentinel2 import *  # noqa: F403
from .sentinel3 import *  # noqa: F403
from .era5 import *  # noqa: F403
from .modis import *  # noqa: F403

from .sentinel1 import S1_CALCULATORS
from .sentinel1.indices import preprocess_s1, postprocess_s1_to_db
from .sentinel2 import S2_CALCULATORS, preprocess_s2
from .sentinel3 import S3_CALCULATORS
from .era5 import ERA5_CALCULATORS
from .era5.indices import preprocess_era5
from .modis import MODIS_CALCULATORS

ALL_CALCULATORS = S2_CALCULATORS + S1_CALCULATORS + S3_CALCULATORS + ERA5_CALCULATORS + MODIS_CALCULATORS

SENSOR_PREPROCESSORS: Dict[str, Callable] = {
    "SENTINEL-2": preprocess_s2,   # DN (0–10 000) → reflectance (0–1)
    # "SENTINEL-1": preprocess_s1,   # speckle filter on linear sigma0
    "ERA5":       preprocess_era5, # unit conversion
}

# Sensor name → postprocessing function applied after index computation.
SENSOR_POSTPROCESSORS: Dict[str, Callable] = {
    "SENTINEL-1": postprocess_s1_to_db,
}
