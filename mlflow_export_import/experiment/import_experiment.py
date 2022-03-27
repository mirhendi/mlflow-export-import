""" 
Exports an experiment to a directory.
"""

import os
import mlflow
import click
from mlflow_export_import import click_doc
from mlflow_export_import import peek_at_experiment
from mlflow_export_import.run.import_run import RunImporter
from mlflow_export_import import utils
from mlflow_export_import.common import mlflow_utils
from mlflow_export_import.common.http_client import DatabricksHttpClient

class ExperimentImporter():
    def __init__(self, mlflow_client=None, mlmodel_fix=True, use_src_user_id=False, import_metadata_tags=False):
        """
        :param mlflow_client: MLflow client or if None create default client.
        :param use_src_user_id: Set the destination user ID to the source user ID.
                                Source user ID is ignored when importing into
        :param import_metadata_tags: Import mlflow_export_import tags.
        """
        self.mlflow_client = mlflow_client or mlflow.tracking.MlflowClient()
        self.run_importer = RunImporter(self.mlflow_client, mlmodel_fix=mlmodel_fix, \
            use_src_user_id=use_src_user_id, \
            import_metadata_tags=import_metadata_tags, dst_notebook_dir_add_run_id=True)
        print("MLflowClient:",self.mlflow_client)
        self.dbx_client = DatabricksHttpClient()

    def import_experiment(self, exp_name, input_dir, dst_notebook_dir=None):
        """
        :param: exp_name: Destination experiment name.
        :param: input_dir: Source experiment directory.
        :return: A map of source run IDs and destination run.info.
        """
        exp_id = mlflow_utils.set_experiment(self.dbx_client, exp_name)
        manifest_path = os.path.join(input_dir,"manifest.json")
        dct = utils.read_json_file(manifest_path)
        run_ids = dct["export_info"]["ok_runs"]
        failed_run_ids = dct["export_info"]["failed_runs"]
        print(f"Importing {len(run_ids)} runs into experiment '{exp_name}' from {input_dir}")
        run_ids_map = {}
        run_info_map = {}
        for src_run_id in run_ids:
            dst_run, src_parent_run_id = self.run_importer.import_run(exp_name, os.path.join(input_dir,src_run_id), dst_notebook_dir)
            dst_run_id = dst_run.info.run_id
            run_ids_map[src_run_id] = { "dst_run_id": dst_run_id, "src_parent_run_id": src_parent_run_id }
            run_info_map[src_run_id] = dst_run.info
        print(f"Imported {len(run_ids)} runs into experiment '{exp_name}' from {input_dir}")
        if len(failed_run_ids) > 0:
            print(f"Warning: {len(failed_run_ids)} failed runs were not imported - see {manifest_path}")
        utils.nested_tags(self.mlflow_client, run_ids_map)

        try:
            self._import_permissions(exp_id, input_dir)
            print("Experiment permissions imported for exp", exp_id)
        except Exception as e:
            print("Experiment permissions NOT imported for exp", exp_id)
            print(e)
        return run_info_map

    def _import_permissions(self, dst_exp_id, input_dir):
        permissions_path = os.path.join(input_dir, "permissions.json")
        permissions_data = utils.read_json_file(permissions_path)

        for each_ac in permissions_data['access_control_list']:
            permission_dic = {}
            data = {}
            try:
                permission_dic['user_name'] = each_ac['user_name']
            except:
                permission_dic['group_name'] = each_ac['group_name']

            permission_dic['permission_level'] = each_ac['all_permissions'][0]['permission_level']
            data['access_control_list'] = [permission_dic]
            try:
                self.dbx_client.patch(resource="preview/permissions/experiments/{}".format(dst_exp_id), data=data)
                print("One permission imported")
            except Exception as e:
                print("One permission couldn't be imported", e)
                print(data)

@click.command()
@click.option("--input-dir",
    help="Input path - directory",
    type=str,
    required=True
)
@click.option("--experiment-name",
    help="Destination experiment name",
    type=str,
    required=True
)
@click.option("--just-peek",
    help="Just display experiment metadata - do not import",
    type=bool,
    default=False
)
@click.option("--use-src-user-id",
    help=click_doc.use_src_user_id,
    type=bool,
    default=False
)
@click.option("--import-metadata-tags",
    help="Import mlflow_export_import tags",
    type=bool,
    default=False
)
@click.option("--dst-notebook-dir",
    help="Databricks destination workpsace base directory for notebook. A run ID will be added to contain the run's notebook.",
    type=str,
    required=False,
    show_default=True
)

def main(input_dir, experiment_name, just_peek, use_src_user_id, import_metadata_tags, dst_notebook_dir):
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")
    if just_peek:
        peek_at_experiment(input_dir)
    else:
        importer = ExperimentImporter(
            mlflow_client=None,
            use_src_user_id=use_src_user_id,
            import_metadata_tags=import_metadata_tags)
        importer.import_experiment(experiment_name, input_dir, dst_notebook_dir)

if __name__ == "__main__":
    main()
