from .base import *  # noqa: F403
from .sentinel1 import *  # noqa: F403
from .sentinel2 import *  # noqa: F403
from .sentinel3 import *  # noqa: F403

from .sentinel1 import S1_CALCULATORS
from .sentinel2 import S2_CALCULATORS
from .sentinel3 import S3_CALCULATORS

ALL_CALCULATORS = S2_CALCULATORS + S1_CALCULATORS + S3_CALCULATORS
