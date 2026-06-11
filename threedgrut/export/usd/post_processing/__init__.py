# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-processing (PPISP) USD export.

Groups the modules that author post-processing effects onto exported USD
assets: the SH-bake paths (``sh_bake``, ``sh_simple_bake``,
``view_interpolation``), the PPISP RenderProduct/controller writers
(``ppisp_writer``, ``ppisp_controller_writer``, ``ppisp_controller_weights``),
and the SPG CUDA sidecars (``ppisp_spg``).

See ``README.md`` in this directory for the full PPISP export documentation.
"""
