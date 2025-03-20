// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifdef _MSC_VER
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#include <3dgrt/tensorBuffering.h>
#include <3dgrt/cuoptixMacros.h>
#include <3dgrt/optixTracer.h>
#include <3dgrt/particlePrimitives.h>
#include <3dgrt/pipelineParameters.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <algorithm>
#include <fstream>
#include <optix.h>
#include <optix_function_table_definition.h>

//------------------------------------------------------------------------------
//
//------------------------------------------------------------------------------
namespace {

void contextLogCB(unsigned int level, const char* tag, const char* message, void* /*cbdata */) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

bool readSourceFile(std::string& str, const std::string& filename) {
    // Try to open file
    std::ifstream file(filename.c_str(), std::ios::binary);
    if (file.good()) {
        // Found usable source file
        std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
        str.assign(buffer.begin(), buffer.end());
        return true;
    }
    return false;
}

void getCuStringFromFile(std::string& cu, const char* filename) {
    // Try to get source code from file
    if (readSourceFile(cu, filename)) {
        return;
    }

    // Wasn't able to find or open the requested file
    throw std::runtime_error("Couldn't open source file " + std::string(filename));
}

void getPtxFromCuString(std::string& ptx,
                        const char* include_dir,
                        const char* optix_include_dir,
                        const char* cuda_include_dir,
                        const char* cu_source,
                        const char* name,
                        const std::vector<std::string>& defines,
                        const char** log_string,
                        const std::vector<std::string>& extra_includes = {}) {
    // Create program
    nvrtcProgram prog = 0;
    NVRTC_CHECK_ERROR(nvrtcCreateProgram(&prog, cu_source, name, 0, NULL, NULL));

    // Gather NVRTC options
    std::vector<const char*> options;

    std::string sample_dir;
    sample_dir = std::string("-I") + include_dir;
    options.push_back(sample_dir.c_str());

    // Collect include dirs
    std::vector<std::string> include_dirs;

    include_dirs.push_back(std::string("-I") + optix_include_dir);
    include_dirs.push_back(std::string("-I") + cuda_include_dir);

    for (const std::string& extra_path: extra_includes) {
        include_dirs.push_back(std::string("-I") + extra_path);
    }


    for (const std::string& dir : include_dirs) {
        options.push_back(dir.c_str());
    }

    for (const std::string& def : defines) {
        options.push_back(def.c_str());
    }

    // Collect NVRTC options
    const char* compiler_options[] = {CUDA_NVRTC_OPTIONS};
    std::copy(std::begin(compiler_options), std::end(compiler_options), std::back_inserter(options));

    // JIT compile CU to PTX
    const nvrtcResult compileRes = nvrtcCompileProgram(prog, (int)options.size(), options.data());

    // Retrieve log output
    std::string g_nvrtcLog;
    size_t log_size = 0;
    NVRTC_CHECK_ERROR(nvrtcGetProgramLogSize(prog, &log_size));
    g_nvrtcLog.resize(log_size);
    if (log_size > 1) {
        NVRTC_CHECK_ERROR(nvrtcGetProgramLog(prog, &g_nvrtcLog[0]));
        if (log_string)
            *log_string = g_nvrtcLog.c_str();
    }
    if (compileRes != NVRTC_SUCCESS)
        throw std::runtime_error("NVRTC Compilation failed.\n" + g_nvrtcLog);

    // Retrieve PTX code
    size_t ptx_size = 0;
    NVRTC_CHECK_ERROR(nvrtcGetPTXSize(prog, &ptx_size));
    ptx.resize(ptx_size);
    NVRTC_CHECK_ERROR(nvrtcGetPTX(prog, &ptx[0]));

    // Cleanup
    NVRTC_CHECK_ERROR(nvrtcDestroyProgram(&prog));
}

const char* getInputData(const char* filename,
                         const char* include_dir,
                         const char* optix_include_dir,
                         const char* cuda_include_dir,
                         const char* name,
                         size_t& dataSize,
                         const std::vector<std::string>& defines,
                         const char** log,
                         const std::vector<std::string>& extra_includes = {}) {
    if (log)
        *log = NULL;

    std::string *ptx, cu;
    ptx = new std::string();

    getCuStringFromFile(cu, filename);
    getPtxFromCuString(*ptx, include_dir, optix_include_dir, cuda_include_dir, cu.c_str(), name, defines, log,
                       extra_includes);

    dataSize = ptx->size();
    return ptx->c_str();
}

inline MOGPrimitiveTypes primitiveTypeFromStr(const std::string& primitiveTypeStr) {
    if (primitiveTypeStr == "icosahedron") {
        return MOGPrimitiveTypes::MOGTracingIcosaHedron;

    } else if (primitiveTypeStr == "octahedron") {
        return MOGPrimitiveTypes::MOGTracingOctraHedron;

    } else if (primitiveTypeStr == "tetrahedron") {
        return MOGPrimitiveTypes::MOGTracingTetraHedron;

    } else if (primitiveTypeStr == "diamond") {
        return MOGPrimitiveTypes::MOGTracingDiamond;

    } else if (primitiveTypeStr == "sphere") {
        return MOGPrimitiveTypes::MOGTracingSphere;

    } else if (primitiveTypeStr == "trihexa") {
        return MOGPrimitiveTypes::MOGTracingTriHexa;

    } else if (primitiveTypeStr == "trisurfel") {
        return MOGPrimitiveTypes::MOGTracingTriSurfel;
    } else if (primitiveTypeStr == "custom") {
        return MOGPrimitiveTypes::MOGTracingCustom;
    } else {
        return MOGPrimitiveTypes::MOGTracingInstances;
    }
}

} // namespace

//------------------------------------------------------------------------------
// OptixTracer
//------------------------------------------------------------------------------

 std::vector<std::string> OptixTracer::generateDefines(
    float particleKernelDegree,
    bool particleKernelDensityClamping,
    int particleRadianceSphDegree,
    bool enableNormals,
    bool enableHitCounts
) {
    std::vector<std::string> defines;
    if (_state) {
        defines.emplace_back("-DPARTICLE_KERNEL_DEGREE=" + std::to_string(static_cast<int32_t>(particleKernelDegree)));
        if (enableNormals) {
            defines.emplace_back("-DENABLE_NORMALS");
        }
        if (enableHitCounts) {
            defines.emplace_back("-DENABLE_HIT_COUNTS");
        }
        defines.emplace_back("-DSPH_MAX_NUM_COEFFS=" + std::to_string((_state->particleRadianceSphDegree + 1) * (_state->particleRadianceSphDegree + 1)));
        defines.emplace_back("-DPARTICLE_PRIMITIVE_TYPE=" + std::to_string(_state->gPrimType));
        defines.emplace_back("-DPARTICLE_PRIMITIVE_CLAMPED=" + std::to_string(particleKernelDensityClamping ? 1 : 0));
    }
    return defines;
}

OptixTracer::OptixTracer(
    const std::string& path,
    const std::string& cuda_path,
    const std::string& pipeline,
    const std::string& backwardPipeline,
    const std::string& primitive,
    float particleKernelDegree,
    float particleKernelMinResponse,
    bool particleKernelDensityClamping,
    int particleRadianceSphDegree,
    bool enableNormals,
    bool enableHitCounts) {

    _state = new State();
    memset(_state, 0, sizeof(State));

    char log[2048]; // For error reporting from OptiX creation functions

    // create OptiX context
    _state->context = nullptr;
    {
        // Initialize the OptiX API, loading all API entry points
        OPTIX_CHECK(optixInit());

        // Specify context options
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction       = &contextLogCB;
        options.logCallbackLevel          = 3;

        // Associate a CUDA context (and therefore a specific GPU) with this
        // device context
        CUcontext cuCtx = 0; // zero means take the current context
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &_state->context));
    }

    _state->particleRadianceSphDegree     = particleRadianceSphDegree;
    _state->particleKernelDegree          = particleKernelDegree;
    _state->particleKernelMinResponse     = particleKernelMinResponse;
    _state->particleKernelDensityClamping = particleKernelDensityClamping;
    _state->gNum                          = 0;
    _state->gPrimType                     = primitiveTypeFromStr(primitive);
    _state->gPrimNumVert                  = 0;
    _state->gPrimNumTri                   = 0;
    _state->gPrimNumVert                  = 0;
    _state->gPrimNumTri                   = 0;

    std::vector<std::string> defines = generateDefines(particleKernelDegree, particleKernelDensityClamping,
                                                        particleRadianceSphDegree, enableNormals, enableHitCounts);

    const uint32_t sharedFlags =
        (_state->gPrimType == MOGTracingSphere ? PipelineFlag_SpherePrim : ((_state->gPrimType == MOGTracingCustom) || (_state->gPrimType == MOGTracingInstances) ? PipelineFlag_HasIS : 0));

    //
    // Create pipelines
    //
    _state->moduleTracingFwd   = nullptr;
    _state->pipelineTracingFwd = nullptr;
    _state->sbtTracingFwd      = {};
    createPipeline(_state->context, path, path, cuda_path, defines, pipeline + "Optix", sharedFlags | PipelineFlag_HasRG | PipelineFlag_HasAH,
                   &_state->moduleTracingFwd, &_state->pipelineTracingFwd, _state->sbtTracingFwd, 32);

    _state->moduleTracingBwd   = nullptr;
    _state->pipelineTracingBwd = nullptr;
    _state->sbtTracingBwd      = {};
    createPipeline(_state->context, path, path, cuda_path, defines, backwardPipeline + "Optix", sharedFlags | PipelineFlag_HasRG | PipelineFlag_HasAH,
                   &_state->moduleTracingBwd, &_state->pipelineTracingBwd, _state->sbtTracingBwd, 32);
}

OptixTracer::~OptixTracer(void) {
    OPTIX_CHECK(optixPipelineDestroy(_state->pipelineTracingFwd));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->sbtTracingFwd.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->sbtTracingFwd.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->sbtTracingFwd.hitgroupRecordBase)));
    OPTIX_CHECK(optixModuleDestroy(_state->moduleTracingFwd));

    OPTIX_CHECK(optixPipelineDestroy(_state->pipelineTracingBwd));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->sbtTracingBwd.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->sbtTracingBwd.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->sbtTracingBwd.hitgroupRecordBase)));
    OPTIX_CHECK(optixModuleDestroy(_state->moduleTracingBwd));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->gPrimVrt)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->gPrimTri)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->gPrimAABB)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->iasBuffer)));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->optixAabbPtr)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->paramsDevice)));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->gasBuffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state->gasBufferTmp)));
    OPTIX_CHECK(optixDeviceContextDestroy(_state->context));

    delete _state;
}


void OptixTracer::createPipeline(const OptixDeviceContext context,
                                 const std::string& path,
                                 const std::string& dependencies_path,
                                 const std::string& cuda_path,
                                 const std::vector<std::string>& defines,
                                 const std::string& kernel_name,
                                 uint32_t flags,
                                 OptixModule* module,
                                 OptixPipeline* pipeline,
                                 OptixShaderBindingTable& sbt,
                                 uint32_t numPayloadValues,
                                 const std::vector<std::string>& extra_includes) {
    char log[2048];

    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixModule builtinIsModule                          = nullptr;
    {
        OptixModuleCompileOptions module_compile_options = {};
        module_compile_options.maxRegisterCount          = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        module_compile_options.optLevel                  = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
        module_compile_options.debugLevel                = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

        pipeline_compile_options.usesMotionBlur                   = false;
        pipeline_compile_options.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_compile_options.numPayloadValues                 = numPayloadValues;
        pipeline_compile_options.numAttributeValues               = 0;
        pipeline_compile_options.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        pipeline_compile_options.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        if (flags & PipelineFlag_HasIS) {
            pipeline_compile_options.usesPrimitiveTypeFlags |= OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
        } else if (flags & PipelineFlag_SpherePrim) {
            pipeline_compile_options.usesPrimitiveTypeFlags |= OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
        }

        size_t inputSize              = 0;
        std::string shaderFile        = path + "/src/kernels/cuda/" + kernel_name + ".cu";
        std::string includeDir        = path + "/include";
        std::string optix_include_dir = dependencies_path + "/dependencies/optix-dev/include";
        std::string cuda_include_dir  = cuda_path + "/include";

        const char* input = getInputData(shaderFile.c_str(), includeDir.c_str(), optix_include_dir.c_str(),
                                         cuda_include_dir.c_str(), kernel_name.c_str(), inputSize, defines,
                                         (const char**)&log, extra_includes);
        size_t sizeof_log = sizeof(log);

        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
            context, &module_compile_options, &pipeline_compile_options, input, inputSize, log, &sizeof_log, module));

        if (!(flags & PipelineFlag_HasIS) && (flags & PipelineFlag_SpherePrim)) {
            OptixBuiltinISOptions isOptions = {OPTIX_PRIMITIVE_TYPE_SPHERE, 0, OPTIX_BUILD_FLAG_PREFER_FAST_TRACE, 0};
            OPTIX_CHECK_LOG(optixBuiltinISModuleGet(context, &module_compile_options, &pipeline_compile_options, &isOptions, &builtinIsModule));
        }
    }

    //
    // Create program groups
    //
    OptixProgramGroup raygen_prog_group   = nullptr;
    OptixProgramGroup miss_prog_group     = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;

    {
        OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

        OptixProgramGroupDesc raygen_prog_group_desc = {}; //
        raygen_prog_group_desc.kind                  = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        if (flags & PipelineFlag_HasRG) {
            raygen_prog_group_desc.raygen.module            = *module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
        }
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &raygen_prog_group_desc,
                                                1, // num program groups
                                                &program_group_options, log, &sizeof_log, &raygen_prog_group));

        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind                  = OPTIX_PROGRAM_GROUP_KIND_MISS;
        if (flags & PipelineFlag_HasMS) {
            miss_prog_group_desc.miss.module            = *module;
            miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
        }
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &miss_prog_group_desc,
                                                1, // num program groups
                                                &program_group_options, log, &sizeof_log, &miss_prog_group));

        OptixProgramGroupDesc hitgroup_prog_group_desc = {};
        hitgroup_prog_group_desc.kind                  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        if (flags & PipelineFlag_HasCH) {
            hitgroup_prog_group_desc.hitgroup.moduleCH            = *module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        }
        if (flags & PipelineFlag_HasIS) {
            hitgroup_prog_group_desc.hitgroup.moduleIS            = *module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__is";
        } else if (flags & PipelineFlag_SpherePrim) {
            hitgroup_prog_group_desc.hitgroup.moduleIS            = builtinIsModule;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = nullptr;
        }
        if (flags & PipelineFlag_HasAH) {
            hitgroup_prog_group_desc.hitgroup.moduleAH            = *module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
        }
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &hitgroup_prog_group_desc,
                                                1, // num program groups
                                                &program_group_options, log, &sizeof_log, &hitgroup_prog_group));
    }

    //
    // Link pipeline
    //
    {
        const uint32_t max_trace_depth = 1;

        std::vector<OptixProgramGroup> program_groups;
        program_groups.push_back(raygen_prog_group);
        program_groups.push_back(miss_prog_group);
        program_groups.push_back(hitgroup_prog_group);

        // OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth            = max_trace_depth;
        pipeline_link_options.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;
        size_t sizeof_log                              = sizeof(log);
        OPTIX_CHECK_LOG(optixPipelineCreate(context, &pipeline_compile_options, &pipeline_link_options,
                                            program_groups.data(), static_cast<unsigned int>(program_groups.size()),
                                            log, &sizeof_log, pipeline));

        OptixStackSizes stack_sizes = {};
        for (auto& prog_group : program_groups) {
            OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
        }

        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;
        OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                                               0, // maxCCDepth
                                               0, // maxDCDEpth
                                               &direct_callable_stack_size_from_traversal,
                                               &direct_callable_stack_size_from_state, &continuation_stack_size));
        OPTIX_CHECK(optixPipelineSetStackSize(*pipeline, direct_callable_stack_size_from_traversal,
                                              direct_callable_stack_size_from_state, continuation_stack_size,
                                              1 // maxTraversableDepth
                                              ));
    }

    //
    // Set up shader binding table
    //
    {
        CUdeviceptr raygen_record;
        const size_t raygen_record_size = sizeof(RayGenSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
        RayGenSbtRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
        CUDA_CHECK(
            cudaMemcpy(reinterpret_cast<void*>(raygen_record), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice));

        CUdeviceptr miss_record;
        size_t miss_record_size = sizeof(MissSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
        MissSbtRecord ms_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(miss_record), &ms_sbt, miss_record_size, cudaMemcpyHostToDevice));

        CUdeviceptr hitgroup_record;
        size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
        HitGroupSbtRecord hg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(hitgroup_record), &hg_sbt, hitgroup_record_size, cudaMemcpyHostToDevice));

        sbt.raygenRecord                = raygen_record;
        sbt.missRecordBase              = miss_record;
        sbt.missRecordStrideInBytes     = sizeof(MissSbtRecord);
        sbt.missRecordCount             = 1;
        sbt.hitgroupRecordBase          = hitgroup_record;
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        sbt.hitgroupRecordCount         = 1;
    }
}

OptixTraversableHandle OptixTracer::createParticleInstanceAS(cudaStream_t cudaStream) {
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    if (!_state->optixAabbPtr) {
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&_state->optixAabbPtr), sizeof(OptixAabb), cudaStream));
    }
    OptixAabb hostOptixAabb{-1.f, -1.f, -1.f, 1.f, 1.f, 1.f};
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(_state->optixAabbPtr), &hostOptixAabb, sizeof(OptixAabb), cudaMemcpyHostToDevice, cudaStream));

    OptixBuildInput prim_input                    = {};
    uint32_t prim_input_flags                     = 0;
    prim_input_flags                              = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
    prim_input.type                               = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    prim_input.customPrimitiveArray.numPrimitives = 1;
    prim_input.customPrimitiveArray.aabbBuffers   = &_state->optixAabbPtr;
    prim_input.customPrimitiveArray.strideInBytes = 0;
    prim_input.customPrimitiveArray.flags         = &prim_input_flags;
    prim_input.customPrimitiveArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(_state->context, &accel_options, &prim_input,
                                             1, // Number of build inputs
                                             &gas_buffer_sizes));
    if (_state->gasBufferTmpSz < gas_buffer_sizes.tempSizeInBytes) {
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(_state->gasBufferTmp), cudaStream));
        CUDA_CHECK(cudaMallocAsync(
            reinterpret_cast<void**>(&_state->gasBufferTmp), gas_buffer_sizes.tempSizeInBytes, cudaStream));
        _state->gasBufferTmpSz = gas_buffer_sizes.tempSizeInBytes;
    }
    if ((_state->iasBufferSz < gas_buffer_sizes.outputSizeInBytes)) {
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(_state->iasBuffer), cudaStream));
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&_state->iasBuffer),
                                   gas_buffer_sizes.outputSizeInBytes, cudaStream));
        _state->iasBufferSz = gas_buffer_sizes.outputSizeInBytes;
    }

    OptixTraversableHandle iasHandle;
    OPTIX_CHECK(optixAccelBuild(_state->context,
                                cudaStream, // CUDA stream
                                &accel_options, &prim_input,
                                1, // num build inputs
                                _state->gasBufferTmp, gas_buffer_sizes.tempSizeInBytes, _state->iasBuffer,
                                gas_buffer_sizes.outputSizeInBytes, &iasHandle,
                                nullptr, // emitted property list
                                0        // num emitted properties
                                ));
    return iasHandle;
}

void OptixTracer::reallocateBuffer(CUdeviceptr* bufferPtr, size_t& size, size_t newSize, cudaStream_t cudaStream) {
    if (newSize > size) {
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(*bufferPtr), cudaStream));
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(bufferPtr), newSize, cudaStream));
        size = newSize;
    }
}

void OptixTracer::reallocatePrimGeomBuffer(cudaStream_t cudaStream) {
    if (_state) {
        reallocateBuffer(&_state->gPrimVrt, _state->gPrimVrtSz, sizeof(float3) * _state->gPrimNumVert * _state->gNum, cudaStream);
        reallocateBuffer(&_state->gPrimTri, _state->gPrimTriSz, sizeof(int3) * _state->gPrimNumTri * _state->gNum, cudaStream);
    }
}

void OptixTracer::reallocateParamsDevice(size_t sz, cudaStream_t cudaStream) {
    if (_state) {
        reallocateBuffer(&_state->paramsDevice, _state->paramsDeviceSz, sz, cudaStream);
    }
}

void OptixTracer::buildBVH(torch::Tensor mogPos,
                           torch::Tensor mogRot,
                           torch::Tensor mogScl,
                           torch::Tensor mogDns,
                           unsigned int rebuild,
                           bool allow_update) {

    const uint32_t gNum = mogPos.size(0);

    const uint32_t primitiveOpts = _state->particleKernelDensityClamping ? MOGRenderOpts::MOGRenderAdaptiveKernelClamping : MOGRenderOpts::MOGRenderNone;

    const torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    if (!rebuild && (_state->gNum != gNum)) {
        std::cerr << "ERROR:: cannot refit GAS with a different number of gaussian" << std::endl;
        rebuild = 1;
    }
    _state->gNum = gNum;

    cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream();

    // Create enclosing geometry primitives from 3d gaussians
    if (_state->gPrimType == MOGTracingCustom) {
        if (_state->gPrimAABBSz < sizeof(OptixAabb) * gNum) {
            CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(_state->gPrimAABB), cudaStream));
            CUDA_CHECK(cudaMallocAsync(
                reinterpret_cast<void**>(&_state->gPrimAABB), sizeof(OptixAabb) * gNum, cudaStream));
            _state->gPrimAABBSz = sizeof(OptixAabb) * gNum;
        }

        if (!_state->optixAabbPtr) {
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&_state->optixAabbPtr), sizeof(OptixAabb), cudaStream));
        }

        OptixAabb hostOptixAabb{1e30f, 1e30f, 1e30f, -1e30f, -1e30f, -1e30f};
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void*>(_state->optixAabbPtr), &hostOptixAabb, sizeof(OptixAabb), cudaMemcpyHostToDevice, cudaStream));

        _state->gPrimNumVert = 0;
        _state->gPrimNumTri  = 0;

        computeGaussianEnclosingAABB(gNum,
                                     getPtr<float3>(mogPos),
                                     getPtr<float4>(mogRot),
                                     getPtr<float3>(mogScl), getPtr<float>(mogDns),
                                     _state->particleKernelMinResponse, primitiveOpts,
                                     _state->particleKernelDegree,
                                     reinterpret_cast<OptixAabb*>(_state->gPrimAABB),
                                     reinterpret_cast<OptixAabb*>(_state->optixAabbPtr), cudaStream);

        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(&_state->gasAABB),
                                   reinterpret_cast<void*>(_state->optixAabbPtr), sizeof(OptixAabb), cudaMemcpyDeviceToHost,
                                   cudaStream));
    } else if (_state->gPrimType == MOGTracingInstances) {

        if (_state->gPrimAABBSz < sizeof(OptixInstance) * gNum) {
            CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(_state->gPrimAABB), cudaStream));
            CUDA_CHECK(cudaMallocAsync(
                reinterpret_cast<void**>(&_state->gPrimAABB), sizeof(OptixInstance) * gNum, cudaStream));
            _state->gPrimAABBSz = sizeof(OptixInstance) * gNum;
        }

        OptixTraversableHandle ias = createParticleInstanceAS(cudaStream);

        if (!_state->optixAabbPtr) {
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&_state->optixAabbPtr), sizeof(OptixAabb), cudaStream));
        }

        OptixAabb hostOptixAabb{1e30f, 1e30f, 1e30f, -1e30f, -1e30f, -1e30f};
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void*>(_state->optixAabbPtr), &hostOptixAabb, sizeof(OptixAabb), cudaMemcpyHostToDevice, cudaStream));

        _state->gPrimNumVert = 0;
        _state->gPrimNumTri  = 0;

        computeGaussianEnclosingInstances(gNum,
                                          getPtr<float3>(mogPos),
                                          getPtr<float4>(mogRot),
                                          getPtr<float3>(mogScl),
                                          getPtr<float>(mogDns),
                                          _state->particleKernelMinResponse,
                                          primitiveOpts,
                                          _state->particleKernelDegree,
                                          ias,
                                          reinterpret_cast<OptixInstance*>(_state->gPrimAABB),
                                          reinterpret_cast<OptixAabb*>(_state->optixAabbPtr), cudaStream);

        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(&_state->gasAABB),
                                   reinterpret_cast<void*>(_state->optixAabbPtr), sizeof(OptixAabb), cudaMemcpyDeviceToHost,
                                   cudaStream));
    } else {
        if (!_state->optixAabbPtr) {
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&_state->optixAabbPtr), sizeof(OptixAabb), cudaStream));
        }

        OptixAabb hostOptixAabb{1e30f, 1e30f, 1e30f, -1e30f, -1e30f, -1e30f};
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void*>(_state->optixAabbPtr), &hostOptixAabb, sizeof(OptixAabb), cudaMemcpyHostToDevice, cudaStream));

        if (_state->gPrimType == MOGTracingIcosaHedron) {
            _state->gPrimNumVert = 12;
            _state->gPrimNumTri  = 20;
            reallocatePrimGeomBuffer(cudaStream);

            computeGaussianEnclosingIcosaHedron(gNum,
                                                getPtr<float3>(mogPos),
                                                getPtr<float4>(mogRot),
                                                getPtr<float3>(mogScl),
                                                getPtr<float>(mogDns),
                                                _state->particleKernelMinResponse,
                                                primitiveOpts,
                                                _state->particleKernelDegree,
                                                reinterpret_cast<float3*>(_state->gPrimVrt),
                                                reinterpret_cast<int3*>(_state->gPrimTri),
                                                reinterpret_cast<OptixAabb*>(_state->optixAabbPtr), cudaStream);
        } else if (_state->gPrimType == MOGTracingOctraHedron) {
            _state->gPrimNumVert = 6;
            _state->gPrimNumTri  = 8;
            reallocatePrimGeomBuffer(cudaStream);

            computeGaussianEnclosingOctaHedron(gNum,
                                               getPtr<float3>(mogPos),
                                               getPtr<float4>(mogRot),
                                               getPtr<float3>(mogScl),
                                               getPtr<float>(mogDns),
                                               _state->particleKernelMinResponse,
                                               primitiveOpts,
                                               _state->particleKernelDegree,
                                               reinterpret_cast<float3*>(_state->gPrimVrt),
                                               reinterpret_cast<int3*>(_state->gPrimTri),
                                               reinterpret_cast<OptixAabb*>(_state->optixAabbPtr),
                                               cudaStream);
            CUDA_CHECK_LAST();
        } else if (_state->gPrimType == MOGTracingTriHexa) {
            _state->gPrimNumVert = 6;
            _state->gPrimNumTri  = 6;
            reallocatePrimGeomBuffer(cudaStream);

            computeGaussianEnclosingTriHexa(gNum,
                                            getPtr<float3>(mogPos),
                                            getPtr<float4>(mogRot),
                                            getPtr<float3>(mogScl),
                                            getPtr<float>(mogDns),
                                            _state->particleKernelMinResponse,
                                            primitiveOpts,
                                            _state->particleKernelDegree,
                                            reinterpret_cast<float3*>(_state->gPrimVrt),
                                            reinterpret_cast<int3*>(_state->gPrimTri),
                                            reinterpret_cast<OptixAabb*>(_state->optixAabbPtr),
                                            cudaStream);
            CUDA_CHECK_LAST();
        } else if (_state->gPrimType == MOGTracingTriSurfel) {
            _state->gPrimNumVert = 4;
            _state->gPrimNumTri  = 2;
            reallocatePrimGeomBuffer(cudaStream);
            reallocateBuffer(&_state->gPipelineParticleData, _state->gPipelineParticleDataSz, sizeof(float4) * gNum, cudaStream);

            computeGaussianEnclosingTriSurfel(gNum,
                                              getPtr<float3>(mogPos),
                                              getPtr<float4>(mogRot),
                                              getPtr<float3>(mogScl),
                                              getPtr<float>(mogDns),
                                              _state->particleKernelMinResponse,
                                              primitiveOpts,
                                              _state->particleKernelDegree,
                                              reinterpret_cast<float3*>(_state->gPrimVrt),
                                              reinterpret_cast<int3*>(_state->gPrimTri),
                                              reinterpret_cast<OptixAabb*>(_state->optixAabbPtr),
                                              reinterpret_cast<float4*>(_state->gPipelineParticleData),
                                              cudaStream);
            CUDA_CHECK_LAST();
        } else if (_state->gPrimType == MOGTracingTetraHedron) {
            _state->gPrimNumVert = 4;
            _state->gPrimNumTri  = 4;
            reallocatePrimGeomBuffer(cudaStream);

            computeGaussianEnclosingTetraHedron(gNum,
                                                getPtr<float3>(mogPos),
                                                getPtr<float4>(mogRot),
                                                getPtr<float3>(mogScl),
                                                getPtr<float>(mogDns),
                                                _state->particleKernelMinResponse,
                                                primitiveOpts,
                                                _state->particleKernelDegree,
                                                reinterpret_cast<float3*>(_state->gPrimVrt),
                                                reinterpret_cast<int3*>(_state->gPrimTri),
                                                reinterpret_cast<OptixAabb*>(_state->optixAabbPtr), cudaStream);
        } else if (_state->gPrimType == MOGTracingSphere) {
            _state->gPrimNumVert = 0;
            _state->gPrimNumTri  = 1; // number of primtive per gaussians
            reallocatePrimGeomBuffer(cudaStream);

            computeGaussianEnclosingSphere(gNum,
                                           getPtr<float3>(mogPos),
                                           getPtr<float4>(mogRot),
                                           getPtr<float3>(mogScl),
                                           getPtr<float>(mogDns),
                                           _state->particleKernelMinResponse,
                                           primitiveOpts,
                                           _state->particleKernelDegree,
                                           reinterpret_cast<float3*>(_state->gPrimVrt),
                                           reinterpret_cast<float*>(_state->gPrimTri),
                                           reinterpret_cast<OptixAabb*>(_state->optixAabbPtr), cudaStream);
        } else {
            _state->gPrimNumVert = 5;
            _state->gPrimNumTri  = 6;
            reallocatePrimGeomBuffer(cudaStream);

            computeGaussianEnclosingDiamond(gNum,
                                            getPtr<float3>(mogPos),
                                            getPtr<float4>(mogRot),
                                            getPtr<float3>(mogScl),
                                            getPtr<float>(mogDns),
                                            _state->particleKernelMinResponse,
                                            primitiveOpts,
                                            _state->particleKernelDegree,
                                            reinterpret_cast<float3*>(_state->gPrimVrt),
                                            reinterpret_cast<int3*>(_state->gPrimTri),
                                            reinterpret_cast<OptixAabb*>(_state->optixAabbPtr), cudaStream);
        }
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(&_state->gasAABB),
                                   reinterpret_cast<void*>(_state->optixAabbPtr), sizeof(OptixAabb), cudaMemcpyDeviceToHost,
                                   cudaStream));
    }

    {
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags             = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        if (allow_update) {
            accel_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
        }
        accel_options.operation = rebuild ? OPTIX_BUILD_OPERATION_BUILD : OPTIX_BUILD_OPERATION_UPDATE;

        OptixBuildInput prim_input = {};
        uint32_t prim_input_flags  = 0;

        if (_state->gPrimType == MOGTracingCustom) {
            prim_input_flags                              = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
            prim_input.type                               = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
            prim_input.customPrimitiveArray.numPrimitives = gNum;
            prim_input.customPrimitiveArray.aabbBuffers   = &_state->gPrimAABB;
            prim_input.customPrimitiveArray.strideInBytes = 0;
            prim_input.customPrimitiveArray.flags         = &prim_input_flags;
            prim_input.customPrimitiveArray.numSbtRecords = 1;
        } else if (_state->gPrimType == MOGTracingInstances) {
            prim_input_flags                      = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
            prim_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            prim_input.instanceArray.numInstances = gNum;
            prim_input.instanceArray.instances    = _state->gPrimAABB;
        } else if (_state->gPrimType == MOGTracingSphere) {
            prim_input_flags                           = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
            prim_input.type                            = OPTIX_BUILD_INPUT_TYPE_SPHERES;
            prim_input.sphereArray.vertexBuffers       = &_state->gPrimVrt;
            prim_input.sphereArray.vertexStrideInBytes = 0;
            prim_input.sphereArray.numVertices         = gNum;
            prim_input.sphereArray.radiusBuffers       = &_state->gPrimTri;
            prim_input.sphereArray.radiusStrideInBytes = 0;
            prim_input.sphereArray.singleRadius        = 0;
            prim_input.sphereArray.flags               = &prim_input_flags;
            prim_input.sphereArray.numSbtRecords       = 1;
        } else {
            // Our build input is a simple list of non-indexed triangle vertices
            prim_input_flags                          = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
            prim_input.type                           = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            prim_input.triangleArray.vertexFormat     = OPTIX_VERTEX_FORMAT_FLOAT3;
            prim_input.triangleArray.numVertices      = _state->gPrimNumVert * gNum;
            prim_input.triangleArray.vertexBuffers    = &_state->gPrimVrt;
            prim_input.triangleArray.indexFormat      = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            prim_input.triangleArray.numIndexTriplets = _state->gPrimNumTri * gNum;
            prim_input.triangleArray.indexBuffer      = _state->gPrimTri;
            prim_input.triangleArray.flags            = &prim_input_flags;
            prim_input.triangleArray.numSbtRecords    = 1;
        }

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(_state->context, &accel_options, &prim_input,
                                                 1, // Number of build inputs
                                                 &gas_buffer_sizes));
        if (_state->gasBufferTmpSz < gas_buffer_sizes.tempSizeInBytes) {
            CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(_state->gasBufferTmp), cudaStream));
            CUDA_CHECK(cudaMallocAsync(
                reinterpret_cast<void**>(&_state->gasBufferTmp), gas_buffer_sizes.tempSizeInBytes, cudaStream));
            _state->gasBufferTmpSz = gas_buffer_sizes.tempSizeInBytes;
        }

        if (rebuild && (_state->gasBufferSz < gas_buffer_sizes.outputSizeInBytes)) {
            CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(_state->gasBuffer), cudaStream));
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&_state->gasBuffer),
                                       gas_buffer_sizes.outputSizeInBytes, cudaStream));
            _state->gasBufferSz = gas_buffer_sizes.outputSizeInBytes;
        }

        OPTIX_CHECK(optixAccelBuild(_state->context,
                                    cudaStream, // CUDA stream
                                    &accel_options, &prim_input,
                                    1, // num build inputs
                                    _state->gasBufferTmp, gas_buffer_sizes.tempSizeInBytes, _state->gasBuffer,
                                    gas_buffer_sizes.outputSizeInBytes, &_state->gasHandle,
                                    nullptr, // emitted property list
                                    0        // num emitted properties
                                    ));
    }

    CUDA_CHECK_LAST();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
OptixTracer::trace(uint32_t frameNumber,
                   torch::Tensor rayToWorld,
                   torch::Tensor rayOri,
                   torch::Tensor rayDir,
                   torch::Tensor particleDensity,
                   torch::Tensor particleRadiance,
                   uint32_t renderOpts,
                   int sphDegree,
                   float minTransmittance) {

    const torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor rayRad            = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 3}, opts);
    torch::Tensor rayDns            = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, opts);
    torch::Tensor rayHit            = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 2}, opts);
    torch::Tensor rayNrm            = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 3}, opts);
    torch::Tensor rayHitsCount      = torch::zeros({rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, opts);

    PipelineParameters paramsHost;
    paramsHost.handle = _state->gasHandle;
    paramsHost.aabb   = _state->gasAABB;

    paramsHost.frameBounds.x = rayOri.size(2) - 1;
    paramsHost.frameBounds.y = rayOri.size(1) - 1;
    paramsHost.frameNumber   = frameNumber;
    paramsHost.gPrimNumTri   = _state->gPrimNumTri;

    paramsHost.minTransmittance       = minTransmittance;
    paramsHost.hitMinGaussianResponse = _state->particleKernelMinResponse;
    paramsHost.alphaMinThreshold      = 1.0f / 255.0f;
    paramsHost.sphDegree              = sphDegree;

    std::memcpy(&paramsHost.rayToWorld[0].x, rayToWorld.cpu().data_ptr<float>(), 3 * sizeof(float4));
    paramsHost.rayOrigin    = packed_accessor32<float, 4>(rayOri);
    paramsHost.rayDirection = packed_accessor32<float, 4>(rayDir);

    paramsHost.particleDensity      = getPtr<const ParticleDensity>(particleDensity);
    paramsHost.particleRadiance     = getPtr<const float>(particleRadiance);
    paramsHost.particleExtendedData = reinterpret_cast<const void*>(_state->gPipelineParticleData);

    paramsHost.rayRadiance    = packed_accessor32<float, 4>(rayRad);
    paramsHost.rayDensity     = packed_accessor32<float, 4>(rayDns);
    paramsHost.rayHitDistance = packed_accessor32<float, 4>(rayHit);
    paramsHost.rayNormal      = packed_accessor32<float, 4>(rayNrm);
    paramsHost.rayHitsCount   = packed_accessor32<float, 4>(rayHitsCount);

    cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream();
    reallocateParamsDevice(sizeof(paramsHost), cudaStream);

    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(_state->paramsDevice), &paramsHost, sizeof(paramsHost), cudaMemcpyHostToDevice, cudaStream));

    OPTIX_CHECK(optixLaunch(_state->pipelineTracingFwd, cudaStream, _state->paramsDevice,
                            sizeof(PipelineParameters), &_state->sbtTracingFwd, rayRad.size(2),
                            rayRad.size(1), rayRad.size(0)));

    CUDA_CHECK_LAST();

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(rayRad, rayDns, rayHit, rayNrm, rayHitsCount);
}

std::tuple<torch::Tensor, torch::Tensor>
OptixTracer::traceBwd(uint32_t frameNumber,
                      torch::Tensor rayToWorld,
                      torch::Tensor rayOri,
                      torch::Tensor rayDir,
                      torch::Tensor rayRad,
                      torch::Tensor rayDns,
                      torch::Tensor rayHit,
                      torch::Tensor rayNrm,
                      torch::Tensor particleDensity,
                      torch::Tensor particleRadiance,
                      torch::Tensor rayRadGrd,
                      torch::Tensor rayDnsGrd,
                      torch::Tensor rayHitGrd,
                      torch::Tensor rayNrmGrd,
                      uint32_t renderOpts,
                      int sphDegree,
                      float minTransmittance) {

    const torch::TensorOptions opts    = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor particleDensityGrad  = torch::zeros({particleDensity.size(0), particleDensity.size(1)}, opts);
    torch::Tensor particleRadianceGrad = torch::zeros({particleRadiance.size(0), particleRadiance.size(1)}, opts);

    PipelineBackwardParameters paramsHost;
    paramsHost.handle = _state->gasHandle;
    paramsHost.aabb   = _state->gasAABB;

    paramsHost.frameBounds.x = rayOri.size(2) - 1;
    paramsHost.frameBounds.y = rayOri.size(1) - 1;
    paramsHost.frameNumber   = frameNumber;
    paramsHost.gPrimNumTri   = _state->gPrimNumTri;

    paramsHost.minTransmittance       = minTransmittance;
    paramsHost.hitMinGaussianResponse = _state->particleKernelMinResponse;
    paramsHost.alphaMinThreshold      = 1.0f / 255.0f;
    paramsHost.sphDegree              = sphDegree;

    std::memcpy(&paramsHost.rayToWorld[0].x, rayToWorld.cpu().data_ptr<float>(), 3 * sizeof(float4));
    paramsHost.rayOrigin    = packed_accessor32<float, 4>(rayOri);
    paramsHost.rayDirection = packed_accessor32<float, 4>(rayDir);

    paramsHost.particleDensity      = getPtr<const ParticleDensity>(particleDensity);
    paramsHost.particleRadiance     = getPtr<const float>(particleRadiance);
    paramsHost.particleExtendedData = reinterpret_cast<const void*>(_state->gPipelineParticleData);

    paramsHost.rayRadiance    = packed_accessor32<float, 4>(rayRad);
    paramsHost.rayDensity     = packed_accessor32<float, 4>(rayDns);
    paramsHost.rayHitDistance = packed_accessor32<float, 4>(rayHit);
    paramsHost.rayNormal      = packed_accessor32<float, 4>(rayNrm);

    paramsHost.particleDensityGrad  = getPtr<ParticleDensity>(particleDensityGrad);
    paramsHost.particleRadianceGrad = getPtr<float>(particleRadianceGrad);

    paramsHost.rayRadianceGrad    = packed_accessor32<float, 4>(rayRadGrd);
    paramsHost.rayDensityGrad     = packed_accessor32<float, 4>(rayDnsGrd);
    paramsHost.rayHitDistanceGrad = packed_accessor32<float, 4>(rayHitGrd);
    paramsHost.rayNormalGrad      = packed_accessor32<float, 4>(rayNrmGrd);

    cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream();

    reallocateParamsDevice(sizeof(paramsHost), cudaStream);
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(_state->paramsDevice), &paramsHost, sizeof(paramsHost), cudaMemcpyHostToDevice, cudaStream));

    OPTIX_CHECK(optixLaunch(_state->pipelineTracingBwd, cudaStream, _state->paramsDevice,
                            sizeof(PipelineBackwardParameters), &_state->sbtTracingBwd,
                            rayRad.size(2), rayRad.size(1), rayRad.size(0)));

    return std::tuple<torch::Tensor, torch::Tensor>(particleDensityGrad, particleRadianceGrad);
}
