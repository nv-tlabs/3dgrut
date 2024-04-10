// Copyright(c) 2020, NVIDIA CORPORATION.All rights reserved.
//
// NVIDIA CORPORATIONand its licensors retain all intellectual property
// and proprietary rights inand to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

//------------------------------------------------------------------------
// Shared C++ code prototypes.

//------------------------------------------------------------------------------
//
// CUDA error-checking
//
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

#define CUDA_CHECK_LAST()                                                      \
    CUDA_CHECK(cudaGetLastError())

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
            throw std::runtime_error( "ERROR: " __FILE__ "(): " + std::string( nvrtcGetErrorString( code ) ) ); \
    } while( 0 )
