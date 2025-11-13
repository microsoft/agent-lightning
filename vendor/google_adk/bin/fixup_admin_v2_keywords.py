#!/Users/sithoilam/miniforge3/bin/python3.9
# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import argparse
import os
import libcst as cst
import pathlib
import sys
from typing import (Any, Callable, Dict, List, Sequence, Tuple)


def partition(
    predicate: Callable[[Any], bool],
    iterator: Sequence[Any]
) -> Tuple[List[Any], List[Any]]:
    """A stable, out-of-place partition."""
    results = ([], [])

    for i in iterator:
        results[int(predicate(i))].append(i)

    # Returns trueList, falseList
    return results[1], results[0]


class adminCallTransformer(cst.CSTTransformer):
    CTRL_PARAMS: Tuple[str] = ('retry', 'timeout', 'metadata')
    METHOD_TO_PARAMS: Dict[str, Tuple[str]] = {
        'check_consistency': ('name', 'consistency_token', 'standard_read_remote_writes', 'data_boost_read_local_writes', ),
        'copy_backup': ('parent', 'backup_id', 'source_backup', 'expire_time', ),
        'create_app_profile': ('parent', 'app_profile_id', 'app_profile', 'ignore_warnings', ),
        'create_authorized_view': ('parent', 'authorized_view_id', 'authorized_view', ),
        'create_backup': ('parent', 'backup_id', 'backup', ),
        'create_cluster': ('parent', 'cluster_id', 'cluster', ),
        'create_instance': ('parent', 'instance_id', 'instance', 'clusters', ),
        'create_logical_view': ('parent', 'logical_view_id', 'logical_view', ),
        'create_materialized_view': ('parent', 'materialized_view_id', 'materialized_view', ),
        'create_table': ('parent', 'table_id', 'table', 'initial_splits', ),
        'create_table_from_snapshot': ('parent', 'table_id', 'source_snapshot', ),
        'delete_app_profile': ('name', 'ignore_warnings', ),
        'delete_authorized_view': ('name', 'etag', ),
        'delete_backup': ('name', ),
        'delete_cluster': ('name', ),
        'delete_instance': ('name', ),
        'delete_logical_view': ('name', 'etag', ),
        'delete_materialized_view': ('name', 'etag', ),
        'delete_snapshot': ('name', ),
        'delete_table': ('name', ),
        'drop_row_range': ('name', 'row_key_prefix', 'delete_all_data_from_table', ),
        'generate_consistency_token': ('name', ),
        'get_app_profile': ('name', ),
        'get_authorized_view': ('name', 'view', ),
        'get_backup': ('name', ),
        'get_cluster': ('name', ),
        'get_iam_policy': ('resource', 'options', ),
        'get_instance': ('name', ),
        'get_logical_view': ('name', ),
        'get_materialized_view': ('name', ),
        'get_snapshot': ('name', ),
        'get_table': ('name', 'view', ),
        'list_app_profiles': ('parent', 'page_size', 'page_token', ),
        'list_authorized_views': ('parent', 'page_size', 'page_token', 'view', ),
        'list_backups': ('parent', 'filter', 'order_by', 'page_size', 'page_token', ),
        'list_clusters': ('parent', 'page_token', ),
        'list_hot_tablets': ('parent', 'start_time', 'end_time', 'page_size', 'page_token', ),
        'list_instances': ('parent', 'page_token', ),
        'list_logical_views': ('parent', 'page_size', 'page_token', ),
        'list_materialized_views': ('parent', 'page_size', 'page_token', ),
        'list_snapshots': ('parent', 'page_size', 'page_token', ),
        'list_tables': ('parent', 'view', 'page_size', 'page_token', ),
        'modify_column_families': ('name', 'modifications', 'ignore_warnings', ),
        'partial_update_cluster': ('cluster', 'update_mask', ),
        'partial_update_instance': ('instance', 'update_mask', ),
        'restore_table': ('parent', 'table_id', 'backup', ),
        'set_iam_policy': ('resource', 'policy', 'update_mask', ),
        'snapshot_table': ('name', 'cluster', 'snapshot_id', 'ttl', 'description', ),
        'test_iam_permissions': ('resource', 'permissions', ),
        'undelete_table': ('name', ),
        'update_app_profile': ('app_profile', 'update_mask', 'ignore_warnings', ),
        'update_authorized_view': ('authorized_view', 'update_mask', 'ignore_warnings', ),
        'update_backup': ('backup', 'update_mask', ),
        'update_cluster': ('name', 'location', 'state', 'serve_nodes', 'node_scaling_factor', 'cluster_config', 'default_storage_type', 'encryption_config', ),
        'update_instance': ('display_name', 'name', 'state', 'type_', 'labels', 'create_time', 'satisfies_pzs', 'satisfies_pzi', ),
        'update_logical_view': ('logical_view', 'update_mask', ),
        'update_materialized_view': ('materialized_view', 'update_mask', ),
        'update_table': ('table', 'update_mask', 'ignore_warnings', ),
    }

    def leave_Call(self, original: cst.Call, updated: cst.Call) -> cst.CSTNode:
        try:
            key = original.func.attr.value
            kword_params = self.METHOD_TO_PARAMS[key]
        except (AttributeError, KeyError):
            # Either not a method from the API or too convoluted to be sure.
            return updated

        # If the existing code is valid, keyword args come after positional args.
        # Therefore, all positional args must map to the first parameters.
        args, kwargs = partition(lambda a: not bool(a.keyword), updated.args)
        if any(k.keyword.value == "request" for k in kwargs):
            # We've already fixed this file, don't fix it again.
            return updated

        kwargs, ctrl_kwargs = partition(
            lambda a: a.keyword.value not in self.CTRL_PARAMS,
            kwargs
        )

        args, ctrl_args = args[:len(kword_params)], args[len(kword_params):]
        ctrl_kwargs.extend(cst.Arg(value=a.value, keyword=cst.Name(value=ctrl))
                           for a, ctrl in zip(ctrl_args, self.CTRL_PARAMS))

        request_arg = cst.Arg(
            value=cst.Dict([
                cst.DictElement(
                    cst.SimpleString("'{}'".format(name)),
cst.Element(value=arg.value)
                )
                # Note: the args + kwargs looks silly, but keep in mind that
                # the control parameters had to be stripped out, and that
                # those could have been passed positionally or by keyword.
                for name, arg in zip(kword_params, args + kwargs)]),
            keyword=cst.Name("request")
        )

        return updated.with_changes(
            args=[request_arg] + ctrl_kwargs
        )


def fix_files(
    in_dir: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    transformer=adminCallTransformer(),
):
    """Duplicate the input dir to the output dir, fixing file method calls.

    Preconditions:
    * in_dir is a real directory
    * out_dir is a real, empty directory
    """
    pyfile_gen = (
        pathlib.Path(os.path.join(root, f))
        for root, _, files in os.walk(in_dir)
        for f in files if os.path.splitext(f)[1] == ".py"
    )

    for fpath in pyfile_gen:
        with open(fpath, 'r') as f:
            src = f.read()

        # Parse the code and insert method call fixes.
        tree = cst.parse_module(src)
        updated = tree.visit(transformer)

        # Create the path and directory structure for the new file.
        updated_path = out_dir.joinpath(fpath.relative_to(in_dir))
        updated_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate the updated source file at the corresponding path.
        with open(updated_path, 'w') as f:
            f.write(updated.code)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Fix up source that uses the admin client library.

The existing sources are NOT overwritten but are copied to output_dir with changes made.

Note: This tool operates at a best-effort level at converting positional
      parameters in client method calls to keyword based parameters.
      Cases where it WILL FAIL include
      A) * or ** expansion in a method call.
      B) Calls via function or method alias (includes free function calls)
      C) Indirect or dispatched calls (e.g. the method is looked up dynamically)

      These all constitute false negatives. The tool will also detect false
      positives when an API method shares a name with another method.
""")
    parser.add_argument(
        '-d',
        '--input-directory',
        required=True,
        dest='input_dir',
        help='the input directory to walk for python files to fix up',
    )
    parser.add_argument(
        '-o',
        '--output-directory',
        required=True,
        dest='output_dir',
        help='the directory to output files fixed via un-flattening',
    )
    args = parser.parse_args()
    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    if not input_dir.is_dir():
        print(
            f"input directory '{input_dir}' does not exist or is not a directory",
            file=sys.stderr,
        )
        sys.exit(-1)

    if not output_dir.is_dir():
        print(
            f"output directory '{output_dir}' does not exist or is not a directory",
            file=sys.stderr,
        )
        sys.exit(-1)

    if os.listdir(output_dir):
        print(
            f"output directory '{output_dir}' is not empty",
            file=sys.stderr,
        )
        sys.exit(-1)

    fix_files(input_dir, output_dir)
