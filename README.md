# Ray Tracing in one weekend - Python

Running through the guide at https://raytracing.github.io/books/RayTracingInOneWeekend.html 
to create a raytracer.

# Docs

To build the docs in windows run:

    sphinx-build.exe . _build

in `docs`. This also runs the apidoc command on build with a convenience function in conf.py.


# Tests

To run the tests, run:

    tox

in the root directory.

To generate a coverage report run:

    tox -e cov

in the root directory.
