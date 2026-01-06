# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

import os
import random
import threading
from abc import ABC
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig

from common.distributed import get_global_rank, get_world_size
from common.fs import copy, exists, listdir, mkdir, remove
from common.partition import partition_by_groups
from common.persistence.utils import get_local_path
from data.common.parquet_sampler import (
    IdentityParquetSampler,
    ParquetSampler,
    create_parquet_sampler,
)
from data.common.utils import filter_parquets, get_parquet_metadata


# Function to save a Parquet file and copy it to a target path
def save_and_copy(
    pa_table,
    local_path: str,
    target_path: str,
    row_group_size: int,
    executor: ThreadPoolExecutor,
    do_async: bool = False,
    futures: List[Tuple[threading.Thread, str]] = [],
):
    # Function to handle completion of the future
    def _make_on_complete(local_path):
        def _on_complete(future):
            target_path = future.result()
            remove(local_path)
            # del future
            print(f"Target path saved: {target_path}")

        return _on_complete

    # Function to write Parquet table and copy it
    def _fn(pa_table, local_path, target_path, row_group_size):
        pq.write_table(
            pa_table,
            local_path,
            row_group_size=row_group_size,
        )
        mkdir(os.path.dirname(target_path))
        copy(local_path, target_path)
        return target_path

    # Submit the task to the executor
    future = executor.submit(_fn, pa_table, local_path, target_path, row_group_size)
    future.add_done_callback(_make_on_complete(local_path))
    futures.append(future)

    # If not asynchronous, wait for all futures to complete
    if not do_async:
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Generated an exception: {exc}")
        executor.shutdown(wait=True)


@dataclass
class FileListOutput:
    existing_files: List[str]
    source_files: List[Any]
    target_files: List[str]


@dataclass
class PersistedParquet:
    path: str

    # Method to save the Parquet file
    def save(
        self,
        row_group_size: int,
        executor: ThreadPoolExecutor,
        pa_table: Optional[pa.Table] = None,
        data_dict: Optional[Dict[str, List[Union[str, bytes]]]] = None,
        is_last_file=False,
        futures: List[threading.Thread] = [],
    ):
        assert (pa_table is None) != (data_dict is None)
        local_path = get_local_path(self.path)
        if not pa_table:
            schema_dict = self.generate_schema_from_dict(data_dict)
            pa_table = pa.Table.from_pydict(data_dict, schema=schema_dict)
        save_and_copy(
            pa_table,
            local_path=local_path,
            target_path=self.path,
            row_group_size=row_group_size,
            executor=executor,
            do_async=not is_last_file,
            futures=futures,
        )

    # Method to generate schema from a dictionary
    def generate_schema_from_dict(
        self,
        data_dict: Dict[str, List[Union[str, bytes]]],
    ):
        schema_dict = {}
        for key, value in data_dict.items():
            if isinstance(value[0], str):
                schema_dict[key] = pa.string()
            elif isinstance(value[0], bytes):
                schema_dict[key] = pa.binary()
            else:
                raise ValueError(f"Unsupported data type for key '{key}': {type(value)}")
        return pa.schema(schema_dict)


# Base class for managing Parquet files
class ParquetManager(ABC):
    """
    Base class for the DumpingManager and RepackingManager.
    """

    def __init__(
        self,
        task: Optional[DictConfig] = None,
        target_dir: str = ".",
    ):
        self.task = task
        self.target_dir = target_dir.rstrip("/")
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.futures = []

    # Method to get list of Parquet files from source path
    def get_parquet_files(
        self,
        source_path: str,
        parquet_sampler: ParquetSampler = IdentityParquetSampler(),
        path_mode: str = "dir",
    ):

        # Helper function to flatten nested lists
        def _flatten(paths):
            if isinstance(paths, list):
                if any(isinstance(i, list) for i in paths):
                    return list(chain(*paths))
                else:
                    return paths
            else:
                return [paths]

        file_paths = _flatten(source_path)
        if path_mode == "dir":
            file_paths = map(listdir, file_paths)
            if isinstance(parquet_sampler.size, float):
                file_paths = map(filter_parquets, file_paths)
                file_paths = map(parquet_sampler, file_paths)
                file_paths = list(chain(*file_paths))
            else:
                file_paths = chain(*file_paths)
                file_paths = parquet_sampler(filter_parquets(file_paths))

        return file_paths

    # Method to save a Parquet file
    def save_parquet(
        self,
        *,
        file_name: str,
        row_group_size: int,
        pa_table: Optional[pa.Table] = None,
        data_dict: Optional[Dict[str, List[Union[str, bytes]]]] = None,
        override: bool = True,
        is_last_file: bool = False,
    ):

        persist = self._get_parquet(file_name)
        if override or not exists(persist.path):
            persist.save(
                pa_table=pa_table,
                data_dict=data_dict,
                executor=self.executor,
                row_group_size=row_group_size,
                is_last_file=is_last_file,
                futures=self.futures,
            )

    # Method to get a PersistedParquet object
    def _get_parquet(self, file_name: str) -> PersistedParquet:
        return PersistedParquet(file_name)


# Class to manage dumping of Parquet files
class DumpingManager(ParquetManager):
    """
    Dumping manager handles parquet saving and resuming.
    """

    def __init__(
        self,
        task: DictConfig,
        target_dir: str,
    ):
        super().__init__(task=task, target_dir=target_dir)

    # Method to generate saving path
    def generate_saving_path(self, file_path: str, rsplit: int):
        part_list = file_path.rsplit("/", rsplit)
        result_folder = "/".join(
            [self.target_dir] + [f"epoch_{self.task.epoch}"] + part_list[-rsplit:-1]
        )
        result_file = "/".join([result_folder, part_list[-1]])
        return result_folder, result_file

    # Method to configure task paths
    def configure_task_path(self, source_path: str, rsplit: int, path_mode: str = "dir"):

        file_paths = self.get_parquet_files(
            source_path=source_path,
            path_mode=path_mode,
        )

        # Shuffle file paths
        random.Random(0).shuffle(file_paths)

        # Partition the file paths based on task configuration
        full_source_files = partition_by_groups(file_paths, self.task.total_count)[self.task.index]
        full_source_files = partition_by_groups(full_source_files, get_world_size())[
            get_global_rank()
        ]

        if not full_source_files:
            return FileListOutput([], [], [])

        generate_saving_path = partial(self.generate_saving_path, rsplit=rsplit)
        full_paths = map(generate_saving_path, full_source_files)
        full_target_folders, full_target_files = map(list, zip(*full_paths))
        full_target_folders = set(full_target_folders)

        existing_file_paths = map(
            lambda folder: listdir(folder) if exists(folder) else [], full_target_folders
        )
        existing_file_paths = chain(*existing_file_paths)
        self.existing_files = list(
            filter(
                lambda path: path.endswith(".parquet") and path in full_target_files,
                existing_file_paths,
            )
        )

        filtered_pairs = list(
            filter(
                lambda pair: pair[1] not in self.existing_files,
                zip(full_source_files, full_target_files),
            )
        )
        if filtered_pairs:
            filtered_source_files, filtered_target_files = map(list, zip(*filtered_pairs))
        else:
            filtered_source_files, filtered_target_files = [], []

        # Skip existing file paths if specified
        skip_exists = self.task.skip_exists
        self.source_files = filtered_source_files if skip_exists else full_source_files
        self.target_files = filtered_target_files if skip_exists else full_target_files

        return FileListOutput(self.existing_files, self.source_files, self.target_files)


class RepackingManager(ParquetManager):
    """
    Repacking manager handles parquet spliting and saving.
    """

    def __init__(
        self,
        task: DictConfig,
        target_dir: str,
        repackaging: DictConfig,
    ):
        super().__init__(task=task, target_dir=target_dir)
        self.repackaging = repackaging

    # Configure the task paths for repacking
    def configure_task_path(
        self,
        source_path: str,
        parquet_sampler: Optional[DictConfig] = None,
        path_mode: str = "dir",
    ):

        parquet_sampler = create_parquet_sampler(config=parquet_sampler)
        file_paths = self.get_parquet_files(
            source_path=source_path,
            parquet_sampler=parquet_sampler,
            path_mode=path_mode,
        )

        random.Random(0).shuffle(file_paths)
        target_dir = self.target_dir
        size = abs(parquet_sampler.size)

        if self.task:
            # Partition the file paths based on task configuration
            file_paths = partition_by_groups(file_paths, self.task.total_count)[self.task.index]
            target_dir = os.path.join(target_dir, f"{self.task.total_count}_{self.task.index}")

            if size > 1:
                size = len(
                    partition_by_groups(range(size), self.task.total_count)[self.task.index]
                )

        # Get metadata for each Parquet file
        metadatas = get_parquet_metadata(file_paths, self.repackaging.num_processes)

        # Create a list of (file_path, row) tuples for each row in the files
        target_items = [
            (file_path, row)
            for file_path, metadata in zip(file_paths, metadatas)
            for row in range(metadata.num_rows)
        ]

        # Shuffle the target items
        random.Random(0).shuffle(target_items)

        if size > 1:
            target_items = target_items[:size]

        # Partition the items into groups for each target file
        items_per_file = partition_by_groups(target_items, self.repackaging.num_files)

        # Generate target file paths
        target_files = [
            os.path.join(target_dir, f"{str(i).zfill(5)}.parquet")
            for i in range(self.repackaging.num_files)
        ]

        existing_file_paths = listdir(target_dir) if exists(target_dir) else []
        self.existing_files = list(
            filter(
                lambda path: path.endswith(".parquet"),
                existing_file_paths,
            )
        )
        self.source_files = items_per_file
        self.target_files = target_files

        return FileListOutput(self.existing_files, self.source_files, self.target_files)
