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

import warnings
from typing import Optional

from rich.console import Console
from rich.progress import BarColumn, Progress, ProgressColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.text import Text
from rich.table import Table

# Colors from: https://rich.readthedocs.io/en/stable/appendix/colors.html

# Disable future warnings from 3rd parties
warnings.simplefilter(action="ignore", category=FutureWarning)


class IterationSpeedColumn(ProgressColumn):
    """Renders human readable iteration speed."""

    def render(self, task) -> Text:
        """Show data iteration speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("?", style="bold")
        return Text(f"{speed:.2f}it/s", style="bold")


class RichLogger:
    console = Console()
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.fields[additional_info]}"),
        TextColumn(":: 🚗💨 [red]Speed:"),
        IterationSpeedColumn(),
        TextColumn(":: 🕒 [yellow]Elapsed:"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    progress_alive = False
    progress_tasks = dict()
    finished_tasks = dict()

    def info(self, msg):
        self.console.log(f"[INFO] {msg}", style="white")

    def warning(self, msg):
        self.console.log(f"[WARNING] {msg}", style="bright_yellow")

    def error(self, msg):
        self.console.log(f"[ERROR] {msg}", style="bright_red")

    def log_rule(self, text):
        self.console.rule(text)

    def log_table(self, title: str, record: dict):
        table = Table(title=title)
        for key in record.keys():
            table.add_column(key, justify="left", style="yellow3", no_wrap=True)
        table.add_row(*[f"{'{:.3f}'.format(v)}" if isinstance(v, float) else v for v in record.values()])
        self.console.print(table)

    def get_task(self, task_id):
        if task_id in self.progress._tasks:
            return self.progress._tasks[task_id]
        else:
            raise ValueError(f"TaskID '{task_id}' not found, perhaps the task has already finished.")

    def _concat_additional_progress_info(self, **metrics):
        additional_info = []
        for k, v in metrics.items():
            formatted_k = f"[bold cyan]{k}"
            formatted_v = f"[default]{'{:.3f}'.format(v)}" if isinstance(v, float) else f"[default]{v}"
            additional_info.append(f"{formatted_k}: {formatted_v}")
        return " ".join(additional_info)

    def start_progress(self, task_name, total_steps, color=None, **metrics):
        additional_info = self._concat_additional_progress_info(**metrics)
        task_id = self.progress.add_task(f"[{color}]{task_name}", total=total_steps, additional_info=additional_info)
        # start progress if it's the first task
        if len(self.progress_tasks) == 0:
            self.progress.start()
            self.progress_alive = True
        self.progress_tasks[task_name] = dict(task_id=task_id, additional_info="")

    def log_progress(self, task_name, advance, **metrics):
        additional_info = self._concat_additional_progress_info(**metrics)
        self.progress.update(
            self.progress_tasks[task_name]["task_id"], advance=advance, additional_info=additional_info
        )

    def end_progress(self, task_name):
        task_id = self.progress_tasks[task_name]["task_id"]
        # log task time
        self.finished_tasks[task_name] = dict(name=task_name, elapsed=self.get_task(task_id).elapsed)
        # remove task from progress
        self.progress.remove_task(task_id)
        # clean up
        del self.progress_tasks[task_name]
        # stop progress if it's the last task
        if len(self.progress_tasks) == 0:
            self.progress.stop()
            self.progress_alive = False

    def track(
        self,
        sequence,
        description: str = "Working...",
        total: Optional[float] = None,
        update_period: float = 0.1,
        color: str = None,
        transient: bool = False,
    ):
        if color is not None:
            desc_column = TextColumn(f"[{color}][progress.description]" + "{task.description}[default]")
        else:
            desc_column = TextColumn("[progress.description]{task.description}")
        _progress = Progress(
            desc_column,
            BarColumn(),
            TaskProgressColumn(),
            TextColumn(":: 🕒 [yellow]Elapsed"),
            TimeElapsedColumn(),
            console=self.console,
            transient=transient,
        )

        if self.progress_alive:
            self.progress.stop()

        with _progress:
            yield from _progress.track(sequence, total=total, description=description, update_period=update_period)

        if self.progress_alive:
            self.progress.start()


logger = RichLogger()
