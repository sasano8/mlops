import importlib
import inspect
import json
from functools import partial
from pathlib import Path
from typing import Dict, Union
from urllib.parse import urlparse

from pydantic import BaseModel

import mlflow
from mlflow.entities import Experiment


class ExperimentConf(BaseModel):
    id: str
    name: str
    public_name: str
    description: str = ("",)
    tags: Dict[str, Union[str, str]] = {}
    params: dict = {}


class MlException(Exception):
    ...


def open_config(file_path=None) -> dict:
    try:
        with open(file_path) as f:
            conf = json.load(f)
    except FileNotFoundError as e:
        raise

    if not isinstance(conf, dict):
        raise TypeError()

    if "name" not in conf:
        raise TypeError()

    if "tags" not in conf:
        conf["tags"] = {}

    if not isinstance(conf["tags"], dict):
        raise TypeError()

    return conf


def write_conf(file_path=None):
    ...


def get_or_create_experiment(name: str) -> Experiment:
    experiment = mlflow.get_experiment_by_name(name)
    if experiment:
        return experiment

    experiment_id = mlflow.create_experiment("sklearn-elasticnet-wine")
    experiment = mlflow.get_experiment(experiment_id)
    return experiment


def normalize_tags(tags: Union[str, dict]):
    if isinstance(tags, str):
        return {tags: ""}
    else:
        return tags


def experiment(tags: Union[str, dict]):
    func = partial(ml, tags=tags)
    return func


def ml(func, *, tags: str = "default"):
    mod_path = inspect.getmodule(func).__file__  # noqa

    def wrapped(conf_path: Union[str, Path, None] = None):
        if conf_path is None:
            conf_path = Path(mod_path).absolute().parent / "mlconf.json"

        if isinstance(conf_path, (str, Path)):
            _conf = open_config(conf_path)
        else:
            raise MlException()

        experiment = get_or_create_experiment(_conf["name"])
        if _conf.get("id", None) != experiment.experiment_id:
            _conf["id"] = experiment.experiment_id

        nonlocal tags
        normalized_tags = normalize_tags(tags)
        conf = ExperimentConf(**_conf)
        with open(conf_path, "w") as f:
            json.dump(conf.dict(), f, ensure_ascii=False, indent=4)

        experiment.tags.update({**conf.tags, **normalized_tags})

        print(experiment)
        with mlflow.start_run(experiment_id=experiment.experiment_id, nested=True):
            mlflow.set_tags(experiment.tags)
            model = func(conf)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(
                    model, "model", registered_model_name=conf.public_name
                )
            else:
                mlflow.sklearn.log_model(model, "model")
            return model

    return wrapped


def run(package: str, entry_point: str = "main"):
    name = package + "." + entry_point
    mod = importlib.import_module(name)
    mod.run()
