
from pathlib import Path
import mlflow
import json
from typing import Union, Any, Dict
from pydantic import BaseModel
from functools import partial
import logging

from mlflow.entities import Experiment

class ExperimentConf(BaseModel):
    id: str
    name: str
    tags: Dict[str, Union[str, None]] = {}
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

    experiment_id = mlflow.create_experiment('sklearn-elasticnet-wine')
    experiment = mlflow.get_experiment(experiment_id)
    return experiment


def normalize_tags(tags: Union[str, dict]):
    if isinstance(tags, str):
        return {
            tags: None
        }
    else:
        return tags


def experiment(tags):
    func = partial(ml, tags=tags)
    return func
    


def ml(func, *, tags: str = "default"):
    def wrapped(conf_path: Union[str, Path, None] = None):
        if __name__ == "__main__":
            ...

        if conf_path is None:
            conf_path = Path(__file__).absolute().parent / "mlconf.json"

        if isinstance(conf_path, (str, Path)):
            conf = open_config(conf_path)
        else:
            raise MlException()

        experiment = get_or_create_experiment(conf["name"])
        if conf.get("id", None) != experiment.experiment_id:
            conf["id"] = experiment.experiment_id

        nonlocal tags
        tags = normalize_tags(tags)
        conf = ExperimentConf(**conf)
        with open(conf_path, "w") as f:
            json.dump(conf.dict(), f)

        experiment.tags.update({**conf.tags, **tags})
        mlflow.set_tags(experiment.tags)

        with mlflow.start_run(experiment_id=experiment.experiment_id, nested=True):
            results = func(conf)
            return results
    return wrapped



@experiment(tags="default")
def init(conf: ExperimentConf):
    from .process import run
    run(conf)


# if __name__ == "__main__":
#     print(init())
