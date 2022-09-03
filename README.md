
# https://kakakakakku.hatenablog.com/entry/2022/06/14/095437

管理ツールをインストールする。
いらないかも？

```
git clone https://github.com/mlflow/mlflow
```

管理ツールを起動する。

```
mlflow ui
```

実験する。

```
python run.py example
```

推論エンドポイントを起動する。

```
mlflow models serve --no-conda -m file:///home/sasano8/projects/mlops/mlruns/1/fb8125400d4145ab9d197a844c2556d2/artifacts/model -p 1234
```

推論する。

```
curl -X POST -H 'Content-Type:application/json; format=pandas-split' http://127.0.0.1:1234/invocations\
  --data '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' 
[10.652032783691832]

curl -X POST -H 'Content-Type:application/json; format=pandas-split' http://127.0.0.1:1234/invocations\
  --data '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[15.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' 
[10.83593352318585]
```
