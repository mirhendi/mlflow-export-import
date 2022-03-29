"""
Imports models and their experiments and runs.
"""

import os
import time
import json
import click
import pickle
from concurrent.futures import ThreadPoolExecutor
import mlflow
from mlflow_export_import import utils, click_doc
from mlflow_export_import.common import filesystem as _filesystem
from mlflow_export_import.experiment.import_experiment import ExperimentImporter
from mlflow_export_import.model.import_model import AllModelImporter

print("MLflow Tracking URI:", mlflow.get_tracking_uri())

def _remap(run_info_map):
    res = {}
    for dct in run_info_map.values():
        for src_run_id,run_info in dct.items():
            res[src_run_id] = run_info
    return res

def _import_experiment(importer, exp_name, exp_input_dir):
    try:
        _run_info_map = importer.import_experiment(exp_name, exp_input_dir)
        return _run_info_map
    except Exception:
        import traceback
        traceback.print_exc()

def import_experiments(input_dir, experiment_name_prefix, use_src_user_id, import_metadata_tags, use_threads):
    start_time = time.time()
    manifest_path = os.path.join(input_dir,"experiments","manifest.json")
    manifest = utils.read_json_file(manifest_path)
    exps = manifest["experiments"]
    importer = ExperimentImporter(None, use_src_user_id, import_metadata_tags)
    print("Experiments:")
    for exp in exps: 
        print(" ",exp)

    if not use_threads:
        print("Not using threads ......")
        run_info_map = {}
        exceptions = []
        for exp in exps:
            exp_input_dir = os.path.join(input_dir, "experiments", exp["id"])
            try:
                exp_name = experiment_name_prefix + exp["name"] if experiment_name_prefix else exp["name"]
                _run_info_map = importer.import_experiment( exp_name, exp_input_dir)
                run_info_map[exp["id"]] = _run_info_map
            except Exception as e:
                exceptions.append(e)
                import traceback
                traceback.print_exc()

    if use_threads:
        print("Using threads ......")
        max_workers = os.cpu_count() or 4 if use_threads else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            thread_f = {}
            for exp in exps:
                exp_input_dir = os.path.join(input_dir, "experiments", exp["id"])
                exp_name = experiment_name_prefix + exp["name"] if experiment_name_prefix else exp["name"]
                thread_f[exp["id"]] = executor.submit(_import_experiment, importer, exp_name, exp_input_dir)
        run_info_map = {}
        exceptions = {}
        for each_f in thread_f:
            result = thread_f[each_f].result()
            if result is not None:
                run_info_map[each_f] = thread_f[each_f].result()

    duration = round(time.time() - start_time, 1)
    if len(exceptions) > 0:
        print(f"Errors: {len(exceptions)}")
    print(f"Duration: {duration} seconds")
    return run_info_map, { "experiments": len(exps), "exceptions": exceptions, "duration": duration }

def import_models(input_dir, run_info_map, delete_model, verbose, use_threads):
    max_workers = os.cpu_count() or 4 if use_threads else 1
    start_time = time.time()
    models_dir = os.path.join(input_dir, "models")
    manifest_path = os.path.join(models_dir,"manifest.json")
    manifest = utils.read_json_file(manifest_path)
    models = manifest["ok_models"]
    importer = AllModelImporter(run_info_map)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for model in models:
            dir = os.path.join(models_dir, model)
            executor.submit(importer.import_model, model, dir, delete_model, verbose)

    duration = round(time.time() - start_time, 1)
    return { "models": len(models), "duration": duration }

def import_all(input_dir, delete_model, use_src_user_id, import_metadata_tags, verbose, use_threads, experiment_name_prefix):
    start_time = time.time()
    exp_res = import_experiments(input_dir, experiment_name_prefix, use_src_user_id, import_metadata_tags)
    print("Saving run mapping info")
    with open('exp_res.pkl', 'wb') as f:
        pickle.dump(exp_res, f)

    print("Loading run mapping info")
    with open('exp_res.pkl', 'rb') as f:
        exp_results = pickle.load(f)
    run_info_map = _remap(exp_results[0])
    model_res = import_models(input_dir, run_info_map, delete_model, verbose, use_threads)
    duration = round(time.time() - start_time, 1)
    dct = { "duration": duration, "experiment_import": exp_results[1], "model_import": model_res }
    fs = _filesystem.get_filesystem(".")
    utils.write_json_file(fs, "import_report.json", dct)
    print("\nImport report:")
    print(json.dumps(dct,indent=2)+"\n")


@click.command()
@click.option("--input-dir", 
    help="Input directory.", 
    required=True, 
    type=str
)
@click.option("--delete-model", 
    help=click_doc.delete_model, 
    type=bool, 
    default=False, 
    show_default=True
)
@click.option("--verbose", 
    type=bool, 
    help="Verbose.", 
    default=False, 
    show_default=True
)
@click.option("--use-src-user-id", 
    help=click_doc.use_src_user_id, 
    type=bool, 
    default=False, 
    show_default=True
)
@click.option("--import-metadata-tags", 
    help=click_doc.import_metadata_tags, 
    type=bool, 
    default=False, 
    show_default=True
)
@click.option("--use-threads",
    help=click_doc.use_threads,
    type=bool,
    default=False,
    show_default=True
)
@click.option("--experiment-name-prefix",
    help="If specified, added as prefix to experiment name.",
    default=None,
    type=str,
   show_default=True
)

def main(input_dir, delete_model, use_src_user_id, import_metadata_tags, verbose, use_threads, experiment_name_prefix):
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")
    import_all(input_dir, 
        delete_model=delete_model, 
        use_src_user_id=use_src_user_id, 
        import_metadata_tags=import_metadata_tags, 
        verbose=verbose, 
        use_threads=use_threads,
        experiment_name_prefix=experiment_name_prefix)

if __name__ == "__main__":
    main()
