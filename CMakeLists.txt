# Run  with:
# mkdir build
# cd build
# cmake -DPYTHON_EXECUTABLE=$(python3 -c "import sys; print(sys.executable)") ..


cmake_minimum_required(VERSION 3.5.0)
set(CMAKE_CXX_STANDARD 17)
project(restart VERSION 0.1.0 LANGUAGES CXX)

# set the warning compiler flags
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")

# set debug mode
set(CMAKE_BUILD_TYPE Debug)

# set the generator to Unix Makefiles
set(CMAKE_GENERATOR "Unix Makefiles")

# check the system, if its apple and arm64, then set the architecture to arm64,
# also activate debug mode
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architecture" FORCE)
endif()

# function to prepend the current source directory to a list of files
# this is used to make the files in the subdirectories visible to the main executable
function(PREPEND invar)
    set(listVar "")
    foreach(f ${${invar}})
        list(APPEND listVar ${CMAKE_CURRENT_SOURCE_DIR}/${f})
    endforeach()
    set(${invar} ${listVar} PARENT_SCOPE)
endfunction()


# Set output directories for binaries and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build/lib")


# Find pybind11
find_package(Python EXACT 3.10 REQUIRED COMPONENTS Interpreter Development)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)



# Add subdirectories for each module
add_subdirectory(src/restart/src)
add_subdirectory(src/restart/src/routers)
add_subdirectory(src/restart/src/arbiters)
add_subdirectory(src/restart/src/allocators)
add_subdirectory(src/restart/src/power)
add_subdirectory(src/restart/src/networks)


# Find and include the nlohmann JSON library
# if not present, include the header in present in the include dir
find_path(nlohmann_json_INCLUDE_DIR nlohmann/json.hpp)
if(NOT nlohmann_json_INCLUDE_DIR)
    message(WARNING "nlohmann JSON library not found, using the header provided in the include directory")
    set(nlohmann_json_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/restart/include)
else()
    message(STATUS "nlohmann JSON library found at ${nlohmann_json_INCLUDE_DIR}")
endif()


add_library(_restart STATIC ${common_SRC} ${routers_SRC} ${arbiters_SRC} ${allocators_SRC} ${power_SRC} ${network_SRC}  ${common_INC} ${routers_INC} ${arbiters_INC} ${allocators_INC} ${power_INC} ${network_INC})
target_include_directories(_restart PUBLIC src/restart/src src/restart/src/routers src/restart/src/arbiters src/restart/src/allocators src/restart/src/power src/restart/src/networks ${nlohmann_json_INCLUDE_DIR} ${pybind11_INCLUDE_DIRS})

# Define the main executable
add_executable(run_restart src/restart/src/main.cpp)
# Define the pybind11 module
pybind11_add_module(nocsim src/restart/src/simulation_bindings.cpp)
# Link libraries
target_link_libraries(run_restart PUBLIC _restart pybind11::module Python::Python m)
target_link_libraries(nocsim PUBLIC _restart pybind11::module Python::Python m)
