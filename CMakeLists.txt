cmake_minimum_required(VERSION 3.12)
project(latticeDNAOrigami VERSION 1.0 LANGUAGES CXX)

# Build type
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
        STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

add_executable(latticeDNAOrigami
    apps/main.cpp
    src/ideal_random_walk.cpp
    src/constant_temp_simulation.cpp
    src/random_gens.cpp
#    src/us_simulation.cpp
    src/nearest_neighbour.cpp
    src/orientation_movetype.cpp
    src/met_movetypes.cpp
    src/annealing_simulation.cpp
    src/order_params.cpp
    src/domain.cpp
    src/utility.cpp
    src/hash.cpp
    src/top_constraint_points.cpp
    #    src/enumerate.cpp
    src/files.cpp
    src/cb_movetypes.cpp
    src/movetypes.cpp
    src/rg_movetypes.cpp
    src/simulation.cpp
    src/bias_functions.cpp
    src/parser.cpp
    src/origami_system.cpp
    src/origami_potential.cpp
    src/ptmc_simulation.cpp
)
target_include_directories(latticeDNAOrigami PUBLIC include/latticeDNAOrigami)
target_compile_features(latticeDNAOrigami PUBLIC cxx_std_17)
set_target_properties(latticeDNAOrigami PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_CLANG_TIDY "clang-tidy;-checks=boost-*,cppcoreguidelines-*,clang-analyzer-*,misc-*,modernize-*,mpi-*,performance-*,readability-*,clang-diagnostic-*,-clang-diagnostic-unused-parameter,-clang-analyzer-core.NonNullParamChecker,-clang-analyzer-deadcode.DeadStores,-clang-diagnostic-c++11-narrowing,-readability-named-parameter,-clang-diagnostic-shorten-64-to-32,-clang-diagnostic-sign-conversion,-clang-analyzer-optin.cplusplus.VirtualCall;-warnings-as-errors=*"
)

# MPI
find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
target_link_libraries(latticeDNAOrigami PUBLIC MPI::MPI_CXX)

# Boost
find_package(Boost 1.50 REQUIRED COMPONENTS
    program_options
    mpi
    serialization
    system
    filesystem
)
message(STATUS "Boost version: ${Boost_VERSION}")
target_link_libraries(latticeDNAOrigami PUBLIC
    Boost::filesystem
    Boost::program_options
    Boost::mpi
    Boost::serialization
    Boost::system
    Boost::filesystem
)

# Interprocedular optimization
#include(CheckIPOSupported)
#check_ipo_supported(RESULT result)
#if(result)
#  message(STATUS "Using interprocedural optimization.")
#  set_target_properties(latticeDNAOrigami PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
#endif()

# Brute force method is required to add interprocedural optimization because CMake does not support it for clang
# There is a bug in the llvm code that prevents this from being used in combination with clang-tidy
#target_compile_options(-flto=thin)

# CCache (increased compiliation speed)
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

install(TARGETS latticeDNAOrigami
    RUNTIME DESTINATION bin
)

# jsoncp
include(FetchContent)
FetchContent_Declare(jsoncpp
    GIT_REPOSITORY https://github.com/open-source-parsers/jsoncpp
)
FetchContent_GetProperties(jsoncpp)
if(NOT jsoncpp_POPULATED)
    FetchContent_Populate(jsoncpp)
    add_subdirectory(${jsoncpp_SOURCE_DIR} ${jsoncpp_BINARY_DIR})
endif()
target_link_libraries(latticeDNAOrigami PUBLIC jsoncpp)

# Compiler warnings
#target_compile_options(latticeDNAOrigami PUBLIC -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wunused -Woverloaded-virtual -Wpedantic -Wconversion -Wsign-conversion -Wnull-dereference -Wdouble-promotion -Werror)
target_compile_options(latticeDNAOrigami PUBLIC -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Woverloaded-virtual -Wpedantic -Wconversion -Wsign-conversion -Wnull-dereference -Wdouble-promotion)

set_target_properties(latticeDNAOrigami PROPERTIES LINK_FLAGS "-fuse-ld=lld")
