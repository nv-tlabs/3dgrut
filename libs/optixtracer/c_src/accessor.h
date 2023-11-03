// Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

// Stripped down version from pytorch. Made to work with optix kernels where it's 
// hard to include dependencies
// https://github.com/pytorch/pytorch/blob/dc169d53aa266560750ea25ee0cf31c7e614550d/aten/src/ATen/core/TensorAccessor.h

/////////////////////////////////////////////////////////////////////////////
// From PyTorch:

// Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
// Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
// Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
// Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
// Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
// Copyright (c) 2011-2013 NYU                      (Clement Farabet)
// Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
// Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
// Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

// From Caffe2:

// Copyright (c) 2016-present, Facebook Inc. All rights reserved.

// All contributions by Facebook:
// Copyright (c) 2016 Facebook Inc.

// All contributions by Google:
// Copyright (c) 2015 Google Inc.
// All rights reserved.

// All contributions by Yangqing Jia:
// Copyright (c) 2015 Yangqing Jia
// All rights reserved.

// All contributions by Kakao Brain:
// Copyright 2019-2020 Kakao Brain

// All contributions by Cruise LLC:
// Copyright (c) 2022 Cruise LLC.
// All rights reserved.

// All contributions from Caffe:
// Copyright(c) 2013, 2014, 2015, the respective contributors
// All rights reserved.

// All other contributions:
// Copyright(c) 2015, 2016 the respective contributors
// All rights reserved.

// Caffe2 uses a copyright model similar to Caffe: each contributor holds
// copyright over their contributions to Caffe2. The project versioning records
// all such contribution and copyright details. If a contributor wants to further
// mark their specific copyright on a particular contribution, they should
// indicate their copyright solely in the commit message of the change when it is
// committed.

// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.

// 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
//    and IDIAP Research Institute nor the names of its contributors may be
//    used to endorse or promote products derived from this software without
//    specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////

#pragma once

#if defined(__OPTIX__)
    typedef int int32_t;
    typedef long long int64_t;
#else
    #include <stdint.h>
#endif

#ifdef __CUDACC__
    #ifdef __CUDA_ARCH__
        #define C10_DEVICE __device__
        #define C10_HOST_DEVICE __device__
    #else
        #define C10_DEVICE __device__
        #define C10_HOST __host__
        #define C10_HOST_DEVICE __host__ __device__
    #endif
#else
    #include <algorithm>
    #define C10_HOST_DEVICE
    #define C10_HOST
#endif

// The PtrTraits argument to the TensorAccessor/GenericPackedTensorAccessor
// is used to enable the __restrict__ keyword/modifier for the data
// passed to cuda.
template <typename T>
struct DefaultPtrTraits {
    typedef T* PtrType;
};

#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename T>
struct RestrictPtrTraits {
    typedef T* __restrict__ PtrType;
};
#endif

// TensorAccessorBase and TensorAccessor are used for both CPU and CUDA tensors.
// For CUDA tensors it is used in device code (only). This means that we restrict ourselves
// to functions and types available there (e.g. IntArrayRef isn't).

// The PtrTraits argument is only relevant to cuda to support `__restrict__` pointers.
template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
class TensorAccessorBase {
public:
    typedef typename PtrTraits<T>::PtrType PtrType;

    C10_HOST_DEVICE TensorAccessorBase(
        PtrType data_,
        const index_t* sizes_,
        const index_t* strides_)
        : data_(data_), sizes_(sizes_), strides_(strides_) {}
    C10_HOST_DEVICE index_t stride(index_t i) const {
        return strides_[i];
    }
    C10_HOST_DEVICE index_t size(index_t i) const {
        return sizes_[i];
    }
    C10_HOST_DEVICE PtrType data() {
        return data_;
    }
    C10_HOST_DEVICE const PtrType data() const {
        return data_;
    }
protected:
    PtrType data_;
    const index_t* sizes_;
    const index_t* strides_;
};

// The `TensorAccessor` is typically instantiated for CPU `Tensor`s using
// `Tensor.accessor<T, N>()`.
// For CUDA `Tensor`s, `GenericPackedTensorAccessor` is used on the host and only
// indexing on the device uses `TensorAccessor`s.
template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
class TensorAccessor : public TensorAccessorBase<T,N,PtrTraits,index_t> {
public:
    typedef typename PtrTraits<T>::PtrType PtrType;

    C10_HOST_DEVICE TensorAccessor(
        PtrType data_,
        const index_t* sizes_,
        const index_t* strides_)
        : TensorAccessorBase<T, N, PtrTraits, index_t>(data_,sizes_,strides_) {}

    C10_HOST_DEVICE TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t i) {
        return TensorAccessor<T,N-1,PtrTraits,index_t>(this->data_ + this->strides_[0]*i,this->sizes_+1,this->strides_+1);
    }

    C10_HOST_DEVICE const TensorAccessor<T, N-1, PtrTraits, index_t> operator[](index_t i) const {
        return TensorAccessor<T,N-1,PtrTraits,index_t>(this->data_ + this->strides_[0]*i,this->sizes_+1,this->strides_+1);
    }
};

template<typename T, template <typename U> class PtrTraits, typename index_t>
class TensorAccessor<T,1,PtrTraits,index_t> : public TensorAccessorBase<T,1,PtrTraits,index_t> {
public:
    typedef typename PtrTraits<T>::PtrType PtrType;

    C10_HOST_DEVICE TensorAccessor(
        PtrType data_,
        const index_t* sizes_,
        const index_t* strides_)
        : TensorAccessorBase<T, 1, PtrTraits, index_t>(data_,sizes_,strides_) {}
    C10_HOST_DEVICE T & operator[](index_t i) {
        // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
        return this->data_[this->strides_[0]*i];
    }
    C10_HOST_DEVICE const T & operator[](index_t i) const {
        return this->data_[this->strides_[0]*i];
    }
};

// GenericPackedTensorAccessorBase and GenericPackedTensorAccessor are used on for CUDA `Tensor`s on the host
// and as
// In contrast to `TensorAccessor`s, they copy the strides and sizes on instantiation (on the host)
// in order to transfer them on the device when calling kernels.
// On the device, indexing of multidimensional tensors gives to `TensorAccessor`s.
// Use RestrictPtrTraits as PtrTraits if you want the tensor's data pointer to be marked as __restrict__.
// Instantiation from data, sizes, strides is only needed on the host and std::copy isn't available
// on the device, so those functions are host only.
template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
class GenericPackedTensorAccessorBase {
public:
    typedef typename PtrTraits<T>::PtrType PtrType;

#if !defined(__CUDACC__)
    C10_HOST GenericPackedTensorAccessorBase() {}

    C10_HOST GenericPackedTensorAccessorBase(
        PtrType data_,
        const index_t* sizes_,
        const index_t* strides_)
        : data_(data_) {
        std::copy(sizes_, sizes_ + N, std::begin(this->sizes_));
        std::copy(strides_, strides_ + N, std::begin(this->strides_));
    }

    // if index_t is not int64_t, we want to have an int64_t constructor
    template <typename source_index_t, class = typename std::enable_if<std::is_same<source_index_t, int64_t>::value>::type>
    C10_HOST GenericPackedTensorAccessorBase(
        PtrType data_,
        const source_index_t* sizes_,
        const source_index_t* strides_)
        : data_(data_) {
        for (const auto i : c10::irange(N)) {
            this->sizes_[i] = sizes_[i];
            this->strides_[i] = strides_[i];
        }
    }
#endif
    C10_HOST_DEVICE index_t stride(index_t i) const {
        return strides_[i];
    }
    C10_HOST_DEVICE index_t size(index_t i) const {
        return sizes_[i];
    }
    C10_HOST_DEVICE PtrType data() {
        return data_;
    }
    C10_HOST_DEVICE const PtrType data() const {
        return data_;
    }
protected:
    PtrType data_;
    index_t sizes_[N];
    index_t strides_[N];
};

template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
class GenericPackedTensorAccessor : public GenericPackedTensorAccessorBase<T,N,PtrTraits,index_t> {
public:
    typedef typename PtrTraits<T>::PtrType PtrType;

#if !defined(__CUDACC__)
    C10_HOST GenericPackedTensorAccessor() : GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t>() {}

    C10_HOST GenericPackedTensorAccessor(
        PtrType data_,
        const index_t* sizes_,
        const index_t* strides_)
        : GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t>(data_, sizes_, strides_) {}

    // if index_t is not int64_t, we want to have an int64_t constructor
    template <typename source_index_t, class = typename std::enable_if<std::is_same<source_index_t, int64_t>::value>::type>
    C10_HOST GenericPackedTensorAccessor(
        PtrType data_,
        const source_index_t* sizes_,
        const source_index_t* strides_)
        : GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t>(data_, sizes_, strides_) {}
#else
    C10_DEVICE TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t i) {
        index_t* new_sizes = this->sizes_ + 1;
        index_t* new_strides = this->strides_ + 1;
        return TensorAccessor<T,N-1,PtrTraits,index_t>(this->data_ + this->strides_[0]*i, new_sizes, new_strides);
    }

    C10_DEVICE const TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t i) const {
        const index_t* new_sizes = this->sizes_ + 1;
        const index_t* new_strides = this->strides_ + 1;
        return TensorAccessor<T,N-1,PtrTraits,index_t>(this->data_ + this->strides_[0]*i, new_sizes, new_strides);
    }
#endif

};

template<typename T, template <typename U> class PtrTraits, typename index_t>
class GenericPackedTensorAccessor<T,1,PtrTraits,index_t> : public GenericPackedTensorAccessorBase<T,1,PtrTraits,index_t> {
public:
    typedef typename PtrTraits<T>::PtrType PtrType;

#if !defined(__CUDACC__)
    C10_HOST GenericPackedTensorAccessor() : GenericPackedTensorAccessorBase<T, 1, PtrTraits, index_t>() {}

    C10_HOST GenericPackedTensorAccessor(
        PtrType data_,
        const index_t* sizes_,
        const index_t* strides_)
        : GenericPackedTensorAccessorBase<T, 1, PtrTraits, index_t>(data_, sizes_, strides_) {}

    // if index_t is not int64_t, we want to have an int64_t constructor
    template <typename source_index_t, class = typename std::enable_if<std::is_same<source_index_t, int64_t>::value>::type>
    C10_HOST GenericPackedTensorAccessor(
        PtrType data_,
        const source_index_t* sizes_,
        const source_index_t* strides_)
        : GenericPackedTensorAccessorBase<T, 1, PtrTraits, index_t>(data_, sizes_, strides_) {}
#else
    C10_DEVICE T & operator[](index_t i) {
        return this->data_[this->strides_[0] * i];
    }
    C10_DEVICE const T& operator[](index_t i) const {
        return this->data_[this->strides_[0]*i];
    }
#endif
};

template <typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
using PackedTensorAccessor32 = GenericPackedTensorAccessor<T, N, PtrTraits, int32_t>;

template <typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
using PackedTensorAccessor64 = GenericPackedTensorAccessor<T, N, PtrTraits, int64_t>;