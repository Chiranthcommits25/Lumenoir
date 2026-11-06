"""Training-time tooling for LUMENOIR.

Nothing here is imported by the FastAPI application. The package holds the
offline dataset preparation pipeline and thin evaluation entry points, so the
deployed service stays free of numpy/opencv/torch imports.
"""
