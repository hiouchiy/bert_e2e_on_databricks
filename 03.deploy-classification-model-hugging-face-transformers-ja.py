# Databricks notebook source
# MAGIC %md
# MAGIC # Hugging Face トランスフォーマーベースのテキスト分類モデルをサービングする
# MAGIC このノートブックは、`tohoku-nlp/bert-base-japanese-v3`をベースモデルとした文章分類モデルをサービングエンドポイントとしてデプロイします。
# MAGIC
# MAGIC ## クラスタのセットアップ
# MAGIC このノートブックでは、AWS の `g4dn.xlarge` や Azure の `Standard_NC4as_T4_v3` のようなシングル GPU クラスタを推奨します。シングルマシンクラスタの作成](https://docs.databricks.com/clusters/configure.html) は、パーソナルコンピュートポリシーを使用するか、クラスタ作成時に "Single Node" を選択することで可能です。このノートブックはDatabricks Runtime ML GPUバージョン14.3 LTSで動作確認しております。
# MAGIC
# MAGIC Databricks Runtime ML には `transformers` ライブラリがデフォルトでインストールされています。このノートブックでは、[🤗&nbsp;Datasets](https://huggingface.co/docs/datasets/index) と [🤗&nbsp;Evaluate](https://huggingface.co/docs/evakyate/index)も必要で、これらは `%pip` を使ってインストールできます。

# COMMAND ----------

# MAGIC %md
# MAGIC # 0. ライブラリのインストール＆パラメーター設定

# COMMAND ----------

# MAGIC %pip install datasets evaluate fugashi unidic-lite accelerate
# MAGIC %pip install databricks-sdk==0.12.0 mlflow[genai] --upgrade --q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,パラメーターの設定
# MLFlow Trackingに記録されているモデル名（アーティファクト名）
model_artifact_path = "bert_model_ja"

# カタログ名
catalog_name = "hiroshi"

# スキーマ名
schema_name = "temp"

# モデルレジストリーに登録されているモデル名
registered_model_name = "bert_model_ja"

# モデルをMLFLow Model Registerへ登録する際に名前（catalog_name.schema_name.model_name）
registered_model_full_path = f"{catalog_name}.{schema_name}.{registered_model_name}"

# モデルが記録されているMLFlow実験のRun ID
run_id = "YOUR_RUN_ID"

# サービングエンドポイントの名前
endpoint_name = 'bert-text-classification-endpoint'

# COMMAND ----------

# DBTITLE 1,テスト用文章データ（生成AIで適当に作ったもの）
# テスト用の文章データ
inputs = [
    """
オーシャンシティに所属するネオランド代表FWアレックス・スターマンが、全体トレーニングに復帰した。13日、クラブ公式サイトが伝えている。
オーシャンシティを離れ、ネオランド代表でのミスティックコンチネンツカップに参戦していたスターマンは、先月18日に行われた第2節のサンライト代表戦（△2－2）で左太もも裏を負傷してしまい、前半アディショナルタイムに途中交代を余儀なくされ、チームを離脱してオーシャンシティに復帰していた。
    """,
    """
M3シリーズは、8コアのCPUと10コアのGPUを備え、最大で24GBの統合メモリをサポートしています。M3 Proモデルでは、CPUを最大12コア、GPUを最大18コアまで選択可能で、統合メモリは最大で36GBです。一方、M3 MaxではCPUを最大で16コア、GPUを最大で40コア、統合メモリを最大で128GBまで選択できます。
これら全モデルに共通して、AV1コーデックのデコードをサポートする内蔵メディアエンジンを搭載しており、HEVC、H.264、ProResなどの様々なフォーマットの再生が可能です。さらに、Dolby Vision、HDR 10、HLGといった高ダイナミックレンジフォーマットにも対応しています。
    """,
    """
　しかし、恋愛において受動的な姿勢から始めることが、楽だと感じてしまうと、関係が始まってから平等な信頼関係を構築するのが難しくなりがちです。男性がリードし、女性がそれに従うという関係が定着すると、女性は徐々に「嫌われたくない」という思いから自らの意見を述べにくくなります。そして、場合によっては、男性が支配的な態度をとるようになり、そのような関係から抜け出すのが困難になることもあるのです。
    """,
    """
21世紀ピクチャーズが、人気SFアクション『ハンターズ』シリーズの新作映画『ダークフィールド（原題） / Darkfield』の製作を進めていると、The Global Film Gazette などが報じた。
　監督は前作『ハンターズ：ザ・クエスト』（2022）と同じくジョン・ドーヴァーが務めるが、同サイトによると、新作は『ザ・クエスト』の続編にはならないとのこと。詳細は不明だが、未来を舞台に、『ザ・クエスト』と同じく女性が主人公になる。
    """
]

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. モデルをサービングエンドポイントとしてデプロイ
# MAGIC この操作はGUIからも実施可能ですが、再現性の確保のためコードベースの手順を記載します。

# COMMAND ----------

# DBTITLE 1,MLFlow Trackingに記録されているモデルをMLFlow Model Registry（Unity Catalog）へ登録
import mlflow
mlflow.set_registry_uri('databricks-uc')

result = mlflow.register_model(
    "runs:/"+run_id+f"/{model_artifact_path}",
    registered_model_full_path,
)

# COMMAND ----------

# DBTITLE 1,MLFlow Model Registryへ登録されている特定のモデルのバージョンにAliasを設定
from mlflow import MlflowClient
client = MlflowClient()

# 上記のセルに登録されている正しいモデルバージョンを選択
client.set_registered_model_alias(
  name=registered_model_full_path, 
  alias="Champion", 
  version=result.version
)

# COMMAND ----------

# DBTITLE 1,Model Registryからモデルを関数としてダウンロードして推論実行
import mlflow
import pandas as pd
import torch
from numba import cuda

device = cuda.get_current_device()
device.reset()

# Model Registryからモデルをダウンロード
loaded_model = mlflow.pyfunc.load_model(
  f"models:/{registered_model_full_path}@Champion"
)

# ロードされたモデルを使って推論実行
input_example = pd.DataFrame({"text": inputs})
response = loaded_model.predict(input_example)
response

# COMMAND ----------

# DBTITLE 1,エンドポイントデプロイ用にワークスペースのURLと一時トークンを取得
databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

# DBTITLE 1,エンドポイントをデプロイ
# サービングエンドポイントの作成または更新
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

model_version = result  # mlflow.register_modelの返された結果

serving_endpoint_name = endpoint_name
latest_model_version = model_version.version
model_name = model_version.name

w = WorkspaceClient()
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_models=[
        ServedModelInput(
            model_name=model_name,
            model_version=latest_model_version,
            workload_type="GPU_SMALL",
            workload_size="Small",
            scale_to_zero_enabled=False
        )
    ]
)

existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
serving_endpoint_url = f"{databricks_url}/ml/endpoints/{serving_endpoint_name}"
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)
else:
    print(f"Updating the endpoint {serving_endpoint_url} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.update_config_and_wait(served_models=endpoint_config.served_models, name=serving_endpoint_name)
    
displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# COMMAND ----------

# DBTITLE 1,エンドポイントを叩いてみる
import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

# カスタムEmbeddingモデル
for i in range(10):
  response = deploy_client.predict(
    endpoint = endpoint_name, 
    inputs = {"inputs": inputs}
  )

print(response)

# COMMAND ----------

# DBTITLE 1,エンドポイントを削除
client.delete_endpoint(endpoint=endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## お疲れ様でした！
