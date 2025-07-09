from setuptools import setup, Extension
import sys

# TO DO: customize template

# Define the C++ extension module
ext_modules = [
    Extension(
        'nocsim',  # Name of the module
        sources=[
            'src/simulation_bindings.cpp',  # Source file for pybind11 bindings
            'src/routers/router.cpp',        # Router implementation
            'src/arbiters/arbiter.cpp',      # Arbiter implementation
            'src/allocators/allocator.cpp',  # Allocator implementation
            'src/power/power.cpp',           # Power management implementation
            'src/networks/network.cpp'        # Network implementation
        ],
        include_dirs=['include'],  # Include directory for headers
        language='c++',  # Specify C++ language
    )
]

# Setup function for the package
setup(
    name='my-python-cpp-project',  # Package name
    version='0.1.0',               # Package version
    description='A Python project with C++ extensions using pybind11',  # Description
    ext_modules=ext_modules,       # C++ extensions
    zip_safe=False,                 # Not safe to install as a .egg file
)