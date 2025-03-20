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

import functools
import logging
import cProfile
import time
from dataclasses import dataclass, field
from types import TracebackType
from typing import Final, Self, Type, TypeVar, cast, Optional, Callable

import torch

log: Final = logging.getLogger(__name__)


@dataclass(slots=True, kw_only=True)
class TimingOptions:
    """
    Options for ScopedTimer.
        Parameters:
            active (bool): Enables this timer
            print_enabled (bool): At context manager exit, print elapsed time to func_print_host
            print_details (bool): Collects additional profiling data using cProfile and calls ``print_stats()`` at context exit
            synchronize (bool): Synchronize the CPU thread with any outstanding CUDA work to return accurate GPU timings
            all_results (dict): A dictionary of lists to which the elapsed time will be appended using ``name`` as a key
            func_print_host (Callable): A callback function to print the activity report (``log.info()`` is used by default)
    """
    active: bool = True  # global variable that determines whether to enable timing.
    print_enabled: bool = False 
    print_details: bool = False
    synchronize: bool = False  # sync cuda

    all_results: dict[str, list[float]] = field(default_factory=dict)
    func_print_host: Callable = log.info


# global variable that can be imported in other scripts for usage
timing_options = TimingOptions()

C = TypeVar("C", bound=Callable)


# timer utils based on https://gitlab-master.nvidia.com/omniverse/warp/-/blob/main/warp/utils.py#L662
class ScopedTimer:
    indent: int = -1
    enabled: bool = True
    options: TimingOptions

    def __init__(self, name: Optional[str] = None, opts: TimingOptions = timing_options, enabled: bool = True) -> None:
        """Context manager object for a timer

        Parameters:
            name (str): Name of timer. Decorator default: func.__name__. Context manager: mandatory
            opts (TimingOptions): Options for timing
            enabled (bool): Local variable that determines whether to enable timing for self. Default: True
        """
        self.enabled = enabled
        self.options = opts
        self.name = name
        self.elapsed = 0.0
        self.extra_msg = ""  # Can be used to add to the message printed at manager exit

        self.cp = cProfile.Profile() if self.options.print_details else None

    @staticmethod
    def print_summary(opts: TimingOptions) -> None:
        if opts.active:
            opts.func_print_host("Timings Summary:")
            for key, value in opts.all_results.items():
                if len(value) > 1:
                    opts.func_print_host(f" - {key} {sum(value[2:]) / len(value[2:])} (ms)")
                else:  # len(value)<=1 (actually ==1)
                    opts.func_print_host(f" - {key} {value[0]} (ms)")

    def _print_local_summary(self) -> None:
        assert self.name is not None, "Timer name is required"

        self.options.all_results[self.name].append(self.elapsed)
        
        if self.options.print_enabled:
            indent = "  " * ScopedTimer.indent
            if self.extra_msg:
                self.options.func_print_host(f"{indent}{self.name} took {self.elapsed:.2f} ms {self.extra_msg}")
            else:
                self.options.func_print_host(f"{indent}{self.name} took {self.elapsed:.2f} ms")
        
        if self.cp is not None and self.options.print_details and self.options.print_enabled:
            self.cp.print_stats(sort="tottime")

    def __enter__(self) -> Self:
        """for using the with statement"""
        assert self.name is not None, "Timer name is required"

        if not (self.options.active and self.enabled):
            return self

        self.options.all_results.setdefault(self.name, [])

        if self.options.synchronize:
            torch.cuda.synchronize()

        ScopedTimer.indent += 1

        # start different timers
        self.start = time.perf_counter_ns()

        if self.cp is not None and self.options.print_details:
            self.cp.clear()
            self.cp.enable()

        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """for using the with statement"""
        assert self.name is not None, "Timer name is required"

        if not (self.options.active and self.enabled):
            return

        if self.options.synchronize:
            torch.cuda.synchronize()

        if self.cp is not None:
            self.cp.disable()

        self.elapsed = (time.perf_counter_ns() - self.start) / 1000000.0  # measure in milliseconds

        # post-processing
        self._print_local_summary()

        ScopedTimer.indent -= 1

    def __call__(self, func: C) -> C:
        """for using the the class as a decorator"""

        # Set the timer name to the function's name if no name was provided
        if self.name is None:
            self.name = func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # only in "wrapper", we have access to the updated value of timing_options/self.options from command line
            if not (self.options.active and self.enabled):
                return func(*args, **kwargs)

            with self:
                return func(*args, **kwargs)

        return cast(C, wrapper)  # we just forward the function call in the wrapper


class CudaTimer:
    def __init__(self, enabled=True):
        self.enabled = enabled
        if self.enabled:
            self._start = torch.cuda.Event(enable_timing=True)
            self._recording = False
            self._end = torch.cuda.Event(enable_timing=True)
        
    def start(self):
        if self.enabled:
            assert not self._recording, "CudaTimer has already started."
            self._start.record()
            self._recording = True

    def end(self):
        if self.enabled:
            assert self._recording, "CudaTimer has not started."
            self._end.record()
            self._recording = False

    def timing(self) -> float:
        if self.enabled:
            assert not self._recording, "CudaTimer has not ended."
            self._end.synchronize()
            return self._start.elapsed_time(self._end)
        return 0.0
