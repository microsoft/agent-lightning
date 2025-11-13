# -*- coding: utf-8 -*-
# Copyright 2019-2024 The TensorFlow Authors. All Rights Reserved.
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Loader for event file data for an entire TensorBoard log directory."""


import collections
import os
import tensorflow as tf

from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.util import tb_logging


logger = tb_logging.get_logger()


def is_plugins_subdirectory(path):
    """Returns true if the path is a profile subdirectory."""
    if not tf.io.gfile.isdir(path):
        return False
    dirs = tf.io.gfile.listdir(path)
    return "plugins/" in dirs or "plugins" in dirs


def get_plugins_subdirectories(path):
    """Returns a list of plugins subdirectories within the given path."""
    if not tf.io.gfile.exists(path):
        # No directory to traverse.
        logger.warning("Directory does not exist: %s", str(path))
        return ()

    current_glob_string = os.path.join(path, "*")
    while True:
        globs = tf.io.gfile.glob(current_glob_string)

        if not globs:
            # This subdirectory level lacks files. Terminate.
            return

        for glob in globs:
            if is_plugins_subdirectory(glob):
                yield glob

        # Iterate to the next level of subdirectories.
        current_glob_string = os.path.join(current_glob_string, "*")


class LogdirLoader:
    """Loader for a root log directory, maintaining multiple DirectoryLoaders.

    This class takes a root log directory and a factory for DirectoryLoaders, and
    maintains one DirectoryLoader per "logdir subdirectory" of the root logdir.

    Note that this class is not thread-safe.
    """

    def __init__(self, logdir, directory_loader_factory):
        """Constructs a new LogdirLoader.

        Args:
          logdir: The root log directory to load from.
          directory_loader_factory: A factory for creating DirectoryLoaders. The
            factory should take a path and return a DirectoryLoader.

        Raises:
          ValueError: If logdir or directory_loader_factory are None.
        """
        if logdir is None:
            raise ValueError("A logdir is required")
        if directory_loader_factory is None:
            raise ValueError("A directory loader factory is required")
        self._logdir = logdir
        self._directory_loader_factory = directory_loader_factory
        # Maps run names to corresponding DirectoryLoader instances.
        self._directory_loaders = {}

    def synchronize_runs(self):
        """Finds new runs within `logdir` and makes `DirectoryLoaders` for them.

        In addition, any existing `DirectoryLoader` whose run directory
        no longer exists will be deleted.

        """
        logger.info("Starting logdir traversal of %s", self._logdir)
        runs_seen = set()
        for subdir in io_wrapper.GetLogdirSubdirectories(self._logdir):
            run = os.path.relpath(subdir, self._logdir)
            runs_seen.add(run)
            if run not in self._directory_loaders:
                logger.info("- Adding run for relative directory %s", run)
                self._directory_loaders[run] = self._directory_loader_factory(subdir)
        for subdir in get_plugins_subdirectories(self._logdir):
            run = os.path.relpath(subdir, self._logdir)
            runs_seen.add(run)
            if run not in self._directory_loaders:
                logger.info("- Adding run for relative directory %s", run)
                self._directory_loaders[run] = self._directory_loader_factory(subdir)
        stale_runs = set(self._directory_loaders) - runs_seen
        if stale_runs:
            for run in stale_runs:
                logger.info("- Removing run for relative directory %s", run)
                del self._directory_loaders[run]
        logger.info("Ending logdir traversal of %s", self._logdir)

    def get_run_events(self):
        """Returns tf.Event generators for each run's `DirectoryLoader`.

        Warning: the generators are stateful and consuming them will affect the
        results of any other existing generators for that run; calling code should
        ensure it takes events from only a single generator per run at a time.

        Returns:
          Dictionary containing an entry for each run, mapping the run name to a
          generator yielding tf.Event protobuf objects loaded from that run.
        """
        runs = list(self._directory_loaders)
        logger.info("Creating event loading generators for %d runs", len(runs))
        run_to_loader = collections.OrderedDict()
        for run_name in sorted(runs):
            loader = self._directory_loaders[run_name]
            run_to_loader[run_name] = self._wrap_loader_generator(loader.Load())
        return run_to_loader

    def _wrap_loader_generator(self, loader_generator):
        """Wraps `DirectoryLoader` generator to swallow
        `DirectoryDeletedError`."""
        try:
            for item in loader_generator:
                yield item
        except directory_watcher.DirectoryDeletedError:
            return
