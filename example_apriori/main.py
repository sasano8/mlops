# import opendatasets as od


# def main():
#     od.download(
#         "https://www.kaggle.com/ekrembayar/apriori-association-rules-grocery-store"
#     )

import kaggle

kaggle.api.authenticate()
kaggle.api.dataset_download_files(
    "alexmiles/grocery-products-purchase-data",
    "datasets/grocery-products-purchase-data",
    unzip=True,
)

# https://di-acc2.com/marketing/16651/
import pandas as pd

data = pd.read_csv(
    "datasets/grocery-products-purchase-data/" + "Grocery Products Purchase.csv"
)

# レコード数
record_len = len(data)
# カラム数
column_len = len(data.columns)

# トランザクション形式に加工
transactions = []
for i in range(record_len):
    # データをリスト型に変更
    values = [str(data.values[i, j]) for j in range(column_len)]
    values_notnull = []
    for check in values:
        if check != "nan":
            values_notnull.append(check)
    transactions.append(values_notnull)

from mlxtend.preprocessing import TransactionEncoder

# データをテーブル形式に加工
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
print(df)


from mlxtend.frequent_patterns import apriori

freq_items = apriori(
    df,  # データフレーム
    min_support=0.01,  # 支持度(support)の最小値
    use_colnames=True,  # 出力値のカラムに購入商品名を表示
    max_len=None,  # 生成されるitemsetsの個数
    verbose=0,  # low_memory=Trueの場合のイテレーション数
    low_memory=False,  # メモリ制限あり＆大規模なデータセット利用時に有効
)

# 結果出力
freq_items = freq_items.sort_values("support", ascending=False).reset_index(drop=True)
print(freq_items)


from mlxtend.frequent_patterns import association_rules

# アソシエーション・ルール抽出
df_rules = association_rules(
    freq_items,  # supportとitemsetsを持つデータフレーム
    metric="confidence",  # アソシエーション・ルールの評価指標
    min_threshold=0.1,  # metricsの閾値
)

print(df_rules)


results = df_rules[
    (df_rules["confidence"] > 0.2) & (df_rules["lift"] > 3.0)  # 信頼度
]  # リフト値

# 結果出力
print(results.loc[:, ["antecedents", "consequents", "confidence", "lift"]])
