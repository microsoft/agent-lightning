import sys

import numpy

from shapely.geos import geos_version_string

# Show some diagnostic information; handy for CI
print("Python version: " + sys.version.replace("\n", " "))
print("GEOS version: " + geos_version_string)
print("Numpy version: " + numpy.version.version)
