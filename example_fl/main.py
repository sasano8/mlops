import warnings
from mlops import ExperimentConf, experiment
from example_fl.fl_server import FlServerConfig


@experiment(tags="default")
def run(conf: ExperimentConf):
    warnings.filterwarnings("ignore")
    return FlServerConfig().run()
    # predicted_qualities = lr.predict(test_x)

    # (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    # print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (p.alpha, p.l1_ratio))
    # print("  RMSE: %s" % rmse)
    # print("  MAE: %s" % mae)
    # print("  R2: %s" % r2)

    # mlflow.log_params(p.dict())
    # mlflow.log_metrics(dict(rmse=rmse, r2=r2, mae=mae))

    # return lr
