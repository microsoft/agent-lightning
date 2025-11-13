"""
Provides a wrapper for the shapely.lib C API for use in Cython.
Internally, the shapely C extension uses a PyCapsule to provide run-time access
to function pointers within the C API.

To use these functions, you must first call the following function in each Cython module:
`import_shapely_c_api()`

This uses a macro to dynamically load the functions from pointers in the PyCapsule.
Each C function in shapely.lib exposed in the C API must be specially-wrapped to enable
this capability.

Segfaults will occur if the C API is not imported properly.
"""

cimport numpy as np
from cpython.ref cimport PyObject

from shapely._geos cimport GEOSContextHandle_t, GEOSCoordSequence, GEOSGeometry


cdef extern from "c_api.h":
    # shapely.lib C API loader; returns -1 on error
    # MUST be called before calling other C API functions
    int import_shapely_c_api() except -1

    # C functions provided by the shapely.lib C API
    # Note: GeometryObjects are always managed as Python objects
    # in Cython to avoid memory leaks, not PyObject* (even though
    # they are declared that way in the header file).
    object PyGEOS_CreateGeometry(GEOSGeometry *ptr, GEOSContextHandle_t ctx)
    char PyGEOS_GetGEOSGeometry(PyObject *obj, GEOSGeometry **out) nogil
    GEOSCoordSequence* PyGEOS_CoordSeq_FromBuffer(GEOSContextHandle_t ctx, const double* buf,
                                                 unsigned int size, unsigned int dims,
                                                 char ring_closure) nogil
