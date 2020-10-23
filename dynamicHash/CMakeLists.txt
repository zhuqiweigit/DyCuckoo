cmake_minimum_required(VERSION 2.8)
project(dynamic)
find_package(CUDA QUIET REQUIRED)
if (NOT CUDA_FOUND)
    message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

include_directories(/usr/local/cuda/samples/common/inc)
set(CMAKE_CXX_STANDARD 11)

set(Dy
        data/data_layout.cuh
        include/dynamic_helpers.cuh include/hash_functions.cuh include/qualifiers.cuh
        core/dynamic_hash.cuh core/static_cuckoo.cuh
        thirdParty/cnmem.h thirdParty/cnmem.cpp
        tools/gputimer.h
        test/dynamic_test.cu
        )
set(St
        data/data_layout.cuh
        include/dynamic_helpers.cuh include/hash_functions.cuh include/qualifiers.cuh
        core/dynamic_hash.cuh core/static_cuckoo.cuh
        thirdParty/cnmem.h thirdParty/cnmem.cpp
        tools/gputimer.h
        test/static_test.cu
        )
cuda_add_executable(dynamic_test ${Dy})
cuda_add_executable(static_test ${St})

