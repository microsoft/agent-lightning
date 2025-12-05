# Copyright (c) Microsoft. All rights reserved.

"""Evaluation module for SWE-bench instance testing and grading.

This module provides core functionality for evaluating model predictions on SWE-bench
instances. It handles containerized execution of test scripts, patch application,
and generation of evaluation reports. The module orchestrates the complete evaluation
process including container management, patch application, test execution, and result
grading.
"""

import json
import traceback
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional

from docker.models.containers import ExecResult
from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    DOCKER_PATCH,
    DOCKER_USER,
    DOCKER_WORKDIR,
    INSTANCE_IMAGE_BUILD_DIR,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_INSTANCE,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
    RUN_EVALUATION_LOG_DIR,
    UTF8,
    SWEbenchInstance,
)
from swebench.harness.docker_build import close_logger  # type: ignore
from swebench.harness.docker_build import (
    BuildImageError,
    build_container,
    setup_logger,
)
from swebench.harness.docker_utils import cleanup_container  # type: ignore
from swebench.harness.docker_utils import exec_run_with_timeout  # type: ignore
from swebench.harness.docker_utils import remove_image  # type: ignore
from swebench.harness.docker_utils import should_remove  # type: ignore
from swebench.harness.docker_utils import (
    copy_to_container,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec
from swebench.harness.utils import EvaluationError

import docker

GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]


def run_instance(
    test_spec: TestSpec,
    pred: Dict[str, Any],
    rm_image: bool,
    force_rebuild: bool,
    client: docker.DockerClient,
    run_id: str,
    timeout: int | None = None,
    rewrite_reports: bool = False,
):
    """
    Run a single instance with the given prediction.

    Args:
        test_spec (TestSpec): TestSpec instance
        pred (dict): Prediction w/ model_name_or_path, model_patch, instance_id
        rm_image (bool): Whether to remove the image after running
        force_rebuild (bool): Whether to force rebuild the image
        client (docker.DockerClient): Docker client
        run_id (str): Run ID
        timeout (int): Timeout for running tests
        rewrite_reports (bool): True if eval run is just to reformat existing report
    """
    # Set up logging directory
    instance_id = test_spec.instance_id
    model_name_or_path = pred.get(KEY_MODEL, "None").replace("/", "__")
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / instance_id

    # Set up report file
    report_path = log_dir / LOG_REPORT
    if rewrite_reports:
        test_output_path = log_dir / LOG_TEST_OUTPUT
        if not test_output_path.exists():
            raise ValueError(f"Test output file {test_output_path} does not exist")
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )
        # Write report to report.json
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        return instance_id, report
    if report_path.exists():
        return instance_id, json.loads(report_path.read_text())

    if not test_spec.is_remote_image:
        # Link the image build dir in the log dir
        build_dir = INSTANCE_IMAGE_BUILD_DIR / test_spec.instance_image_key.replace(":", "__")
        image_build_link = log_dir / "image_build_dir"
        if not image_build_link.exists():
            try:
                # link the image build dir in the log dir
                image_build_link.symlink_to(build_dir.absolute(), target_is_directory=True)
            except:
                # some error, idk why
                pass

    # Set up logger
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_INSTANCE
    logger = setup_logger(instance_id, log_file)

    # Run the instance
    container = None
    try:
        # Build + start instance container (instance image should already be built)
        container = build_container(test_spec, client, run_id, logger, rm_image, force_rebuild)
        container.start()
        logger.info(f"Container for {instance_id} started: {container.id}")

        # Copy model prediction as patch file to container
        patch_file = Path(log_dir / "patch.diff")
        patch_file.write_text(pred[KEY_PREDICTION] or "")
        logger.info(f"Intermediate patch for {instance_id} written to {patch_file}, now applying to container...")
        copy_to_container(container, patch_file, PurePosixPath(DOCKER_PATCH))  # type: ignore

        # Attempt to apply patch to container (TODO: FIX THIS)
        val: Optional[ExecResult] = None
        for git_apply_cmd in GIT_APPLY_CMDS:
            val = container.exec_run(  # type: ignore
                f"{git_apply_cmd} {DOCKER_PATCH}",
                workdir=DOCKER_WORKDIR,
                user=DOCKER_USER,
            )
            if val.exit_code == 0:
                logger.info(f"{APPLY_PATCH_PASS}:\n{val.output.decode(UTF8)}")
                break
            else:
                logger.info(f"Failed to apply patch to container: {git_apply_cmd}")
        if val is not None:
            logger.info(f"{APPLY_PATCH_FAIL}:\n{val.output.decode(UTF8)}")
            raise EvaluationError(
                instance_id,
                f"{APPLY_PATCH_FAIL}:\n{val.output.decode(UTF8)}",
                logger,
            )

        # Get git diff before running eval script
        git_diff_output_before = (
            container.exec_run("git -c core.fileMode=false diff", workdir=DOCKER_WORKDIR).output.decode(UTF8).strip()  # type: ignore
        )
        logger.info(f"Git diff before:\n{git_diff_output_before}")

        eval_file = Path(log_dir / "eval.sh")
        eval_file.write_text(test_spec.eval_script)
        logger.info(f"Eval script for {instance_id} written to {eval_file}; copying to container...")
        copy_to_container(container, eval_file, PurePosixPath("/eval.sh"))  # type: ignore

        # Run eval script, write output to logs
        test_output, timed_out, total_runtime = exec_run_with_timeout(container, "/bin/bash /eval.sh", timeout)
        test_output_path = log_dir / LOG_TEST_OUTPUT
        logger.info(f"Test runtime: {total_runtime:_.2f} seconds")
        with open(test_output_path, "w") as f:
            f.write(test_output)
            logger.info(f"Test output for {instance_id} written to {test_output_path}")
            if timed_out:
                f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                raise EvaluationError(
                    instance_id,
                    f"Test timed out after {timeout} seconds.",
                    logger,
                )

        # Get git diff after running eval script (ignore permission changes)
        git_diff_output_after = (
            container.exec_run("git -c core.fileMode=false diff", workdir=str(DOCKER_WORKDIR)).output.decode(UTF8).strip()  # type: ignore
        )

        # Check if git diff changed after running eval script
        logger.info(f"Git diff after:\n{git_diff_output_after}")
        if git_diff_output_after != git_diff_output_before:
            logger.info("Git diff changed after running eval script")

        # Get report from test output
        logger.info(f"Grading answer for {instance_id}...")
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )
        logger.info(f"report: {report}\n" f"Result for {instance_id}: resolved: {report[instance_id]['resolved']}")

        # Write report to report.json
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        return instance_id, report
    except EvaluationError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
    except BuildImageError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
    except Exception as e:
        error_msg = f"Error in evaluating model for {instance_id}: {e}\n" f"{traceback.format_exc()}"
        logger.error(error_msg)
    finally:
        # Remove instance container + image, close logger
        cleanup_container(client, container, logger)
        if rm_image:
            remove_image(client, test_spec.instance_image_key, logger)
        close_logger(logger)
    return


def evaluate(
    prediction: Dict[str, Any],
    instance: SWEbenchInstance,
    cache_level: str,
    clean: bool,
    force_rebuild: bool,
    run_id: str,
    timeout: Optional[int],
    namespace: Optional[str],
    instance_image_tag: str,
    rewrite_reports: bool,
):
    client = docker.from_env()
    test_spec = make_test_spec(instance, namespace=namespace, instance_image_tag=instance_image_tag)

    instance_image_ids = {
        test_spec.instance_image_key,
    }
    existing_images = {tag for i in client.images.list(all=True) for tag in i.tags if tag in instance_image_ids}

    return run_instance(
        test_spec,
        prediction,
        should_remove(test_spec.instance_image_key, cache_level, clean, existing_images),
        force_rebuild,
        client,
        run_id,
        timeout,
        rewrite_reports,
    )
