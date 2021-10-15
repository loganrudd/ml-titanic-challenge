"""
Driver run that run all steps (build_features, train_lgbm) of mlflow pipeline and profiles
total CPU and memory usage of pipeline.

See MLproject file and README.md for more details.
"""

import os
import mlflow
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint
import matplotlib
import pandas as pd
import six
import time
from time import localtime, strftime
import multiprocessing as mp
import psutil
from mlflow.tracking.fluent import _get_experiment_id
import argparse


def _already_ran(entry_point_name, parameters, git_commit, experiment_id=None):
    """Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.
    """
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    client = mlflow.tracking.MlflowClient()
    all_run_infos = reversed(client.list_run_infos(experiment_id))
    for run_info in all_run_infos:
        full_run = client.get_run(run_info.run_id)
        tags = full_run.data.tags
        if tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None) != entry_point_name:
            continue
        match_failed = False
        for param_key, param_value in six.iteritems(parameters):
            run_value = full_run.data.params.get(param_key)
            if run_value != param_value:
                match_failed = True
                break
        if match_failed:
            continue

        if run_info.to_proto().status != RunStatus.FINISHED:
            # eprint writes to sys.stderr instead of sys.stdout
            eprint(("Run matched, but is not FINISHED, so skipping "
                    "(run_id=%s, status=%s)") % (run_info.run_id, run_info.status))
            continue

        previous_version = tags.get(mlflow_tags.MLFLOW_GIT_COMMIT, None)
        if git_commit != previous_version:
            eprint(("Run matched, but has a different source version, so skipping "
                    "(found=%s, expected=%s)") % (previous_version, git_commit))
            continue
        return client.get_run(run_info.run_id)
    eprint("No matching run has been found.")
    return None


# TODO: This is not great because it doesn't account for:
# - changes in code
# - changes in dependant steps
def _get_or_run(entrypoint, parameters, git_commit, use_cache=True):
    existing_run = _already_ran(entrypoint, parameters, git_commit)
    if use_cache and existing_run:
        print("Found existing run for entrypoint=%s and parameters=%s"
              % (entrypoint, parameters))
        return existing_run
    print("Launching new run for entrypoint=%s and parameters=%s"
          % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)


def workflow(split_prop, raw_data_path):
    # The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with mlflow.start_run() as active_run:
        git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)

        build_features_run = \
            _get_or_run("build_features", {"raw_data_path": raw_data_path}, git_commit)

        train_lgbm = _get_or_run("train_lgbm", {"split_prop": split_prop}, git_commit)


def monitor(target):
    worker_process = mp.Process(target=target)
    worker_process.start()
    p = psutil.Process(worker_process.pid)

    # log cpu usage of `worker_process` every 10 ms
    stats = []
    while worker_process.is_alive():
        stats.append((p.cpu_percent(),
                      p.memory_percent(),
                      strftime("%H:%M:%S", localtime())))
        time.sleep(0.01)
    worker_process.join()
    df = pd.DataFrame(stats, columns=['CPU', 'RAM', 'Time'])
    fig = df.set_index('Time').plot(figsize=(10, 5), grid=True,
                                    ylabel='% of Total').get_figure()
    fig.savefig('cpu-memory-profile.pdf')

    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split_prop', default=0.8, type=float)
    parser.add_argument('-d', '--raw_data_path')
    args = parser.parse_args()
    cpu_memory_profile = monitor(workflow(args.split_prop, args.raw_data_path))
