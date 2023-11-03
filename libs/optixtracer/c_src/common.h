// Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

// Helper functions to do broadcast guarded fetches
#if defined(__CUDACC__)
    template<class T, typename U, typename... Args>
    static __device__ inline float3 fetch3(const T &tensor, U idx, Args... args) {
    return tensor.size(0) == 1 ? fetch3(tensor[0], args...) : fetch3(tensor[idx], args...);
    }
    template<class T> static __device__ inline float3 fetch3(const T &tensor) {
    return tensor.size(0) == 1 ? make_float3(tensor[0], tensor[0], tensor[0]) : make_float3(tensor[0], tensor[1], tensor[2]);
    }

    template<class T, typename U, typename... Args>
    static __device__ inline float2 fetch2(const T &tensor, U idx, Args... args) {
    return tensor.size(0) == 1 ? fetch2(tensor[0], args...) : fetch2(tensor[idx], args...);
    }
    template<class T> static __device__ inline float2 fetch2(const T &tensor) {
    return tensor.size(0) == 1 ? make_float2(tensor[0], tensor[0]) : make_float2(tensor[0], tensor[1]);
    }

    #include "math_utils.h"
    #include "bsdf.h"
#endif

//------------------------------------------------------------------------------
// CUDA error-checking macros
//------------------------------------------------------------------------------

#define CUDA_CHECK( call )                                                     \
    do                                                                         \
    {                                                                          \
        cudaError_t error = call;                                              \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "CUDA call (" << #call << " ) failed with error: '"          \
               << cudaGetErrorString( error )                                  \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
        }                                                                      \
    } while( 0 )


#define OPTIX_CHECK( call )                                                    \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"         \
               << __LINE__ << ")\n";                                           \
        }                                                                      \
    } while( 0 )

#define OPTIX_CHECK_LOG( call )                                                \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        const size_t sizeof_log_returned = sizeof_log;                         \
        sizeof_log = sizeof( log ); /* reset sizeof_log for future calls */    \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"         \
               << __LINE__ << ")\nLog:\n" << log                               \
               << ( sizeof_log_returned > sizeof( log ) ? "<TRUNCATED>" : "" ) \
               << "\n";                                                        \
        }                                                                      \
    } while( 0 )

#define NVRTC_CHECK_ERROR( func )                                                                                           \
    do                                                                                                                      \
    {                                                                                                                       \
        nvrtcResult code = func;                                                                                            \
        if( code != NVRTC_SUCCESS )                                                                                         \
            throw std::runtime_error( "ERROR: " __FILE__ "(): " + std::string( nvrtcGetErrorString( code ) ) );             \
    } while( 0 )
