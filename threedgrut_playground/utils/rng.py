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

from typing import List, Union

import numpy as np
import torch

"""
A random number generator, specifically tailored for low-discrepency sequences.
Code Adapted from InstantNGP, originally from Burley [2019] https://www.jcgt.org/published/0009/04/01/paper.pdf

Module supports arbitrary uint32 buffer types, including torch tensors and numpy arrays
"""

UInt32Buffer = Union[torch.LongTensor, np.uint32]
UINT32_MASK = 0xFFFFFFFF  # torch doesn't natively support uint32 so we force the overflow with a mask

# fmt: off
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
        0x80000000, 0xC0000000, 0xA0000000, 0xF0000000,
        0x88000000, 0xCC000000, 0xAA000000, 0xFF000000,
        0x80800000, 0xC0C00000, 0xA0A00000, 0xF0F00000,
        0x88880000, 0xCCCC0000, 0xAAAA0000, 0xFFFF0000,
        0x80008000, 0xC000C000, 0xA000A000, 0xF000F000,
        0x88008800, 0xCC00CC00, 0xAA00AA00, 0xFF00FF00,
        0x80808080, 0xC0C0C0C0, 0xA0A0A0A0, 0xF0F0F0F0,
        0x88888888, 0xCCCCCCCC, 0xAAAAAAAA, 0xFFFFFFFF
    ],
    [
        0x80000000, 0xC0000000, 0x60000000, 0x90000000,
        0xE8000000, 0x5C000000, 0x8E000000, 0xC5000000,
        0x68800000, 0x9CC00000, 0xEE600000, 0x55900000,
        0x80680000, 0xC09C0000, 0x60EE0000, 0x90550000,
        0xE8808000, 0x5CC0C000, 0x8E606000, 0xC5909000,
        0x6868E800, 0x9C9C5C00, 0xEEEE8E00, 0x5555C500,
        0x8000E880, 0xC0005CC0, 0x60008E60, 0x9000C590,
        0xE8006868, 0x5C009C9C, 0x8E00EEEE, 0xC5005555
    ],
    [
        0x80000000, 0xC0000000, 0x20000000, 0x50000000,
        0xF8000000, 0x74000000, 0xA2000000, 0x93000000,
        0xD8800000, 0x25400000, 0x59E00000, 0xE6D00000,
        0x78080000, 0xB40C0000, 0x82020000, 0xC3050000,
        0x208F8000, 0x51474000, 0xFBEA2000, 0x75D93000,
        0xA0858800, 0x914E5400, 0xDBE79E00, 0x25DB6D00,
        0x58800080, 0xE54000C0, 0x79E00020, 0xB6D00050,
        0x800800F8, 0xC00C0074, 0x200200A2, 0x50050093,
    ],
    [
        0x80000000, 0x40000000, 0x20000000, 0xB0000000,
        0xF8000000, 0xDC000000, 0x7A000000, 0x9D000000,
        0x5A800000, 0x2FC00000, 0xA1600000, 0xF0B00000,
        0xDA880000, 0x6FC40000, 0x81620000, 0x40BB0000,
        0x22878000, 0xB3C9C000, 0xFB65A000, 0xDDB2D000,
        0x78022800, 0x9C0B3C00, 0x5A0FB600, 0x2D0DDB00,
        0xA2878080, 0xF3C9C040, 0xDB65A020, 0x6DB2D0B0,
        0x800228F8, 0x400B3CDC, 0x200FB67A, 0xB00DDB9D
    ]
]
# fmt: on


def reverse_bits(x: UInt32Buffer) -> UInt32Buffer:
    x = x & UINT32_MASK
    x = ((x & 0xAAAAAAAA) >> 1) | ((x & 0x55555555) << 1)
    x = ((x & 0xCCCCCCCC) >> 2) | ((x & 0x33333333) << 2)
    x = ((x & 0xF0F0F0F0) >> 4) | ((x & 0x0F0F0F0F) << 4)
    x = ((x & 0xFF00FF00) >> 8) | ((x & 0x00FF00FF) << 8)
    return (x >> 16) | (x << 16) & UINT32_MASK


def laine_karras_permutation(x: UInt32Buffer, seed: UInt32Buffer) -> UInt32Buffer:
    x = (x + seed) & UINT32_MASK
    x = x ^ (x * 0x6C50B47C) & UINT32_MASK
    x = x ^ (x * 0xB82F1E52) & UINT32_MASK
    x = x ^ (x * 0xC7AFE638) & UINT32_MASK
    x = x ^ (x * 0x8D22F6E6) & UINT32_MASK
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
    S = float(1.0 / (1 << 32))
    x = shuffled_scrambled_sobol2d(index, seed)
    return [x[0] * S, x[1] * S]  # Implicitly converted to float here


def rng_torch_low_discrepancy(index: torch.LongTensor, seed: torch.LongTensor):
    # torch doesn't natively support uint32 so we use long and force the overflow with a mask
    index = index.long()
    seed = seed.long()
    return ld_random_val_2d(index, seed)


def rng_numpy_low_discrepancy(index: np.array, seed: np.array):
    index = index.astype(np.uint32)
    seed = seed.astype(np.uint32)
    return ld_random_val_2d(index, seed)
