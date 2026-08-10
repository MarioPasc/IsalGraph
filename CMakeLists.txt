cmake_minimum_required(VERSION 3.18)
project(isalgraph_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -----------------------------------------------------------------------
# ISA target selection.
#
# Default: x86-64-v3 (AVX2 + FMA + POPCNT).  This is the lowest common
# denominator across the Picasso node classes: 178 of 333 nodes lack
# AVX-512, so -march=native would produce a binary that dies with SIGILL on
# a random fraction of SLURM tasks -- a failure mode that reads like flaky
# hardware rather than a build fault.  One flag set -> one build hash across
# the whole campaign.
#
# Local override: -DISALGRAPH_NATIVE_MARCH=ON swaps x86-64-v3 for native.
# Use this on a workstation for profiling; never submit such a build to SLURM.
# -----------------------------------------------------------------------
option(ISALGRAPH_NATIVE_MARCH "Replace -march=x86-64-v3 with -march=native" OFF)

# -----------------------------------------------------------------------
# Sanitizer support (development only -- never combine with Release/LTO)
# -----------------------------------------------------------------------
option(ISALGRAPH_ENABLE_SANITIZERS "Enable Address + Undefined sanitizers" OFF)

# -----------------------------------------------------------------------
# Python + nanobind
# NOTE: find_package must come before nanobind_add_module.
# -----------------------------------------------------------------------
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# -----------------------------------------------------------------------
# Extension module: isalgraph.core._native
#
# Sources live in src/isalgraph/core/native/, alongside the Python package.
# The directory is named `native`, NOT `_native`: the built extension is
# `isalgraph.core._native`, and a sibling directory of that exact name would
# be a candidate implicit namespace package.  CPython's FileFinder tries
# extension loaders before the namespace fallback so the .so would still win,
# but the margin is one import-machinery detail and the names are free to
# differ -- so they do.
#
# The C++ sources are excluded from the wheel via tool.scikit-build
# wheel.exclude in pyproject.toml; only the compiled .so is shipped.
# -----------------------------------------------------------------------
nanobind_add_module(_native
    NB_STATIC
    src/isalgraph/core/native/src/bindings.cpp
    src/isalgraph/core/native/src/probe.cpp
    src/isalgraph/core/native/src/cdll.cpp
    src/isalgraph/core/native/src/sparse_graph.cpp
    src/isalgraph/core/native/src/pairs.cpp
    src/isalgraph/core/native/src/string_to_graph.cpp
    src/isalgraph/core/native/src/graph_to_string.cpp
    src/isalgraph/core/native/src/canonical.cpp
    src/isalgraph/core/native/src/levenshtein.cpp
)

target_include_directories(_native PRIVATE src/isalgraph/core/native/include)

# -----------------------------------------------------------------------
# Per-target compile options -- applied only to _native, NOT to
# nanobind-static or any other CMake target.  Using global
# add_compile_options() would propagate our flags into nanobind's own static
# library and trigger a GCC LTO/ODR redefinition bug in nb_types.h.
# -----------------------------------------------------------------------
target_compile_options(_native PRIVATE -Wall -Wextra -Wpedantic)

if(ISALGRAPH_ENABLE_SANITIZERS)
    target_compile_options(_native PRIVATE
        -fsanitize=address,undefined -fno-omit-frame-pointer -g -O1)
    target_link_options(_native PRIVATE
        -fsanitize=address,undefined)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(ISALGRAPH_NATIVE_MARCH)
        target_compile_options(_native PRIVATE
            -O3 -march=native -DNDEBUG -fno-plt -funroll-loops)
    else()
        target_compile_options(_native PRIVATE
            -O3 -march=x86-64-v3 -DNDEBUG -fno-plt -funroll-loops)
    endif()

    # LTO / IPO: applied to the _native target only (not to nanobind-static).
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _IPO_OK OUTPUT _IPO_ERR)
    if(_IPO_OK)
        set_property(TARGET _native PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endif()

# -----------------------------------------------------------------------
# Static libstdc++ / libgcc on Linux.
#
# Picasso's login-node default is gcc 7.5.0 (too old for C++17).  Linking
# statically removes the "module load gcc/..." requirement from every SLURM
# task at run time.  The build itself still needs a C++17 compiler
# (module load gcc/13.2.0 on Picasso at build time).
# -----------------------------------------------------------------------
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(_native PRIVATE -static-libstdc++ -static-libgcc)
endif()

# -----------------------------------------------------------------------
# Install inside the isalgraph.core package so `import isalgraph.core._native`
# resolves correctly in both editable and regular installs.
# -----------------------------------------------------------------------
install(TARGETS _native LIBRARY DESTINATION isalgraph/core)
