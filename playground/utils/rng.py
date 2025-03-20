# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from typing import Union, List


""" 
A random number generator, specifically tailored for low-discrepency sequences. 
Code Adapted from InstantNGP, originally from Burley [2019] https://www.jcgt.org/published/0009/04/01/paper.pdf

Module supports arbitrary uint32 buffer types, including torch tensors and numpy arrays
"""

UInt32Buffer = Union[torch.LongTensor, np.uint32]
UINT32_MASK = 0xFFFFFFFF    # torch doesn't natively support uint32 so we force the overflow with a mask


SOBOL_DIRECTIONS = [
        [
            0x80000000, 0x40000000, 0x20000000, 0x10000000,
            0x08000000, 0x04000000, 0x02000000, 0x01000000,
            0x00800000, 0x00400000, 0x00200000, 0x00100000,
            0x00080000, 0x00040000, 0x00020000, 0x00010000,
            0x00008000, 0x00004000, 0x00002000, 0x00001000,
            0x00000800, 0x00000400, 0x00000200, 0x00000100,
            0x00000080, 0x00000040, 0x00000020, 0x00000010,
            0x00000008, 0x00000004, 0x00000002, 0x00000001
        ],
        [
            0x80000000, 0xc0000000, 0xa0000000, 0xf0000000,
            0x88000000, 0xcc000000, 0xaa000000, 0xff000000,
            0x80800000, 0xc0c00000, 0xa0a00000, 0xf0f00000,
            0x88880000, 0xcccc0000, 0xaaaa0000, 0xffff0000,
            0x80008000, 0xc000c000, 0xa000a000, 0xf000f000,
            0x88008800, 0xcc00cc00, 0xaa00aa00, 0xff00ff00,
            0x80808080, 0xc0c0c0c0, 0xa0a0a0a0, 0xf0f0f0f0,
            0x88888888, 0xcccccccc, 0xaaaaaaaa, 0xffffffff
        ],
        [
            0x80000000, 0xc0000000, 0x60000000, 0x90000000,
            0xe8000000, 0x5c000000, 0x8e000000, 0xc5000000,
            0x68800000, 0x9cc00000, 0xee600000, 0x55900000,
            0x80680000, 0xc09c0000, 0x60ee0000, 0x90550000,
            0xe8808000, 0x5cc0c000, 0x8e606000, 0xc5909000,
            0x6868e800, 0x9c9c5c00, 0xeeee8e00, 0x5555c500,
            0x8000e880, 0xc0005cc0, 0x60008e60, 0x9000c590,
            0xe8006868, 0x5c009c9c, 0x8e00eeee, 0xc5005555
        ],
        [
            0x80000000, 0xc0000000, 0x20000000, 0x50000000,
            0xf8000000, 0x74000000, 0xa2000000, 0x93000000,
            0xd8800000, 0x25400000, 0x59e00000, 0xe6d00000,
            0x78080000, 0xb40c0000, 0x82020000, 0xc3050000,
            0x208f8000, 0x51474000, 0xfbea2000, 0x75d93000,
            0xa0858800, 0x914e5400, 0xdbe79e00, 0x25db6d00,
            0x58800080, 0xe54000c0, 0x79e00020, 0xb6d00050,
            0x800800f8, 0xc00c0074, 0x200200a2, 0x50050093,
        ],
        [
            0x80000000, 0x40000000, 0x20000000, 0xb0000000,
            0xf8000000, 0xdc000000, 0x7a000000, 0x9d000000,
            0x5a800000, 0x2fc00000, 0xa1600000, 0xf0b00000,
            0xda880000, 0x6fc40000, 0x81620000, 0x40bb0000,
            0x22878000, 0xb3c9c000, 0xfb65a000, 0xddb2d000,
            0x78022800, 0x9c0b3c00, 0x5a0fb600, 0x2d0ddb00,
            0xa2878080, 0xf3c9c040, 0xdb65a020, 0x6db2d0b0,
            0x800228f8, 0x400b3cdc, 0x200fb67a, 0xb00ddb9d
        ]
    ]


def reverse_bits(x: UInt32Buffer) -> UInt32Buffer:
    x = x & UINT32_MASK
    x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1))
    x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2))
    x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4))
    x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8))
    return (x >> 16) | (x << 16) & UINT32_MASK


def laine_karras_permutation(x: UInt32Buffer, seed: UInt32Buffer) -> UInt32Buffer:
    x = (x + seed) & UINT32_MASK
    x = x ^ (x * 0x6c50b47c) & UINT32_MASK
    x = x ^ (x * 0xb82f1e52) & UINT32_MASK
    x = x ^ (x * 0xc7afe638) & UINT32_MASK
    x = x ^ (x * 0x8d22f6e6) & UINT32_MASK
    return x


def nested_uniform_scramble_base2(x: UInt32Buffer, seed: UInt32Buffer) -> UInt32Buffer:
    x = reverse_bits(x)
    x = laine_karras_permutation(x, seed)
    x = reverse_bits(x)
    return x


def sobol(index: UInt32Buffer, dim: UInt32Buffer) -> UInt32Buffer:
    X = 0
    for bit in range(0, 32):
        mask = (index >> bit) & 1
        X ^= mask * SOBOL_DIRECTIONS[dim][bit]
    return X & UINT32_MASK


def sobol2d(index: UInt32Buffer) -> List[UInt32Buffer]:
    return [sobol(index, 0), sobol(index, 1)]


def hash_combine(seed: UInt32Buffer, v: int) -> UInt32Buffer:
    return seed ^ (v + (seed << 6) + (seed >> 2))


def shuffled_scrambled_sobol2d(index: UInt32Buffer, seed: UInt32Buffer) -> List[UInt32Buffer]:
    index = nested_uniform_scramble_base2(index, seed)
    X = sobol2d(index)
    X[0] = nested_uniform_scramble_base2(X[0], hash_combine(seed, 0)) & UINT32_MASK
    X[1] = nested_uniform_scramble_base2(X[1], hash_combine(seed, 1)) & UINT32_MASK
    return X


def ld_random_val_2d(index: UInt32Buffer, seed: UInt32Buffer) -> List[torch.FloatTensor]:
    S = float(1.0/ (1 << 32))
    x = shuffled_scrambled_sobol2d(index, seed)
    return [x[0] * S, x[1] * S] # Implicitly converted to float here


def rng_torch_low_discrepancy(index: torch.LongTensor, seed: torch.LongTensor):
    # torch doesn't natively support uint32 so we use long and force the overflow with a mask
    index = index.long()
    seed = seed.long()
    return ld_random_val_2d(index, seed)


def rng_numpy_low_discrepancy(index: np.array, seed: np.array):
    index = index.astype(np.uint32)
    seed = seed.astype(np.uint32)
    return ld_random_val_2d(index, seed)
