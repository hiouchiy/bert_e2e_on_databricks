# Databricks notebook source
# MAGIC %md
# MAGIC # Hugging Face トランスフォーマーでテキスト分類モデルをチューニングする
# MAGIC このノートブックは、"cl-tohoku/bert-base-japanese-whole-word-masking"をベースモデルとした記事分類器をシングルGPUマシンで学習します。
# MAGIC Transformers](https://huggingface.co/docs/transformers/index)ライブラリを使います。
# MAGIC
# MAGIC まず小さなデータセットをダウンロードし、Spark DataFrameに変換し、Unity CatalogへDelta Tableとして保存します。トークン化までの前処理はSpark上で行われます。DBFSは、ドライバ上のローカルファイルとしてデータセットに直接アクセスするための便宜として使用されてますが、DBFSを使用しないように変更することもできます。
# MAGIC
# MAGIC 記事のテキストトークナイゼーションは、ベースモデルとのトークン化の一貫性を持たせるために、モデルのデフォルトトークナイザーの`transformers`で行われます。このノートブックはモデルをファインチューニングするために `transformers` ライブラリの [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) ユーティリティを使用します。このノートブックはトークナイザーと学習済みモデルをTransformersの`パイプライン`にラップし、パイプラインをMLflowモデルとしてログに記録します。
# MAGIC これにより、パイプラインを Spark DataFrame の文字列カラムの UDF として直接適用することが簡単になります。
# MAGIC
# MAGIC ## クラスタのセットアップ
# MAGIC このノートブックでは、AWS の `g4dn.xlarge` や Azure の `Standard_NC4as_T4_v3` のようなシングル GPU クラスタを推奨します。シングルマシンクラスタの作成](https://docs.databricks.com/clusters/configure.html) は、パーソナルコンピュートポリシーを使用するか、クラスタ作成時に "Single Node" を選択することで可能です。このノートブックはDatabricks Runtime ML GPUバージョン14.3 LTSで動作確認しております。
# MAGIC
# MAGIC Databricks Runtime ML には `transformers` ライブラリがデフォルトでインストールされています。このノートブックでは、[🤗&nbsp;Datasets](https://huggingface.co/docs/datasets/index) と [🤗&nbsp;Evaluate](https://huggingface.co/docs/evakyate/index)も必要で、これらは `%pip` を使ってインストールできます。

# COMMAND ----------

# MAGIC %md
# MAGIC # 0. ライブラリのインストールとインポート

# COMMAND ----------

# MAGIC %pip install datasets evaluate fugashi unidic-lite accelerate
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,パラメーターの設定
################################################
# ユーザー毎に変更が不要なパラメーター
################################################

# HuggingFace モデル名
base_model = "tohoku-nlp/bert-base-japanese-v3"

# BERTに投入する文章の最大長
max_length = 128

# データなどを格納するパス
# このデモではドライバーノードのローカルディスクに格納しますが、以下のドキュメントに則りDBFSなど他の適切な場所に格納も可能です
# https://docs.databricks.com/ja/files/write-data.html
tutorial_path = "/databricks/driver" # 
import os
os.environ['TUTORIAL_PATH']=tutorial_path # 後ほどShellコマンドからアクセスするため環境変数にセット

# MLFlow Trackingに記録されているモデル名（アーティファクト名）
model_artifact_path = "bert_model_ja"


################################################
# ユーザー毎に変更が必要なパラメーター
################################################

# Unity Catalog カタログ名
catalog_name = "YOUR_CATALOG_NAME"

# Unity Catalog スキーマ名
schema_name = "YOUR_SCHEMA_NAME"

# Unity Catalog テーブル名
table_name = "YOUR_TABLE_NAME"

# データを格納するためのテーブル名（catalog_name.schema_name.table_name）
table_name_full_path = f"{catalog_name}.{schema_name}.{table_name}"

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. BERTのファインチューニング

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1-1. データのダウンロードとロード
# MAGIC まずデータセットをダウンロードします。その後、Spark DataFrameとしてロードし、Delta TableとしてUnity Catalogへ保存します。
# MAGIC
# MAGIC 今回用いるデータセット[livedoor ニュースコーパス](https://www.rondhuit.com/download/ldcc-20140209.tar.gz)は、[Rondhuitのサイト](https://www.rondhuit.com/download.html)から入手できます。

# COMMAND ----------

# DBTITLE 1,データのダウンロード＆解凍（アタッチされたVMのローカルディスクに一時保存）
# MAGIC %sh
# MAGIC cd $TUTORIAL_PATH
# MAGIC
# MAGIC # データのダウンロード
# MAGIC wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz
# MAGIC
# MAGIC # ファイルの解凍
# MAGIC tar -zxf ldcc-20140209.tar.gz 

# COMMAND ----------

# DBTITLE 1,./text以下にそれぞれのメディアに対応するフォルダがあるのを確認
# MAGIC %sh
# MAGIC ls $TUTORIAL_PATH/text

# COMMAND ----------

# DBTITLE 1,適当なファイルの内容を確認（最初の3行はメタデータで、4行目から本文）
# MAGIC %sh
# MAGIC cat $TUTORIAL_PATH/text/it-life-hack/it-life-hack-6342280.txt

# COMMAND ----------

# DBTITLE 1,ファイル名とクラスの一覧をPandas Dataframeとして作成
import glob
import pandas as pd

# カテゴリーのリスト
category_list = [
    'dokujo-tsushin',
    'it-life-hack',
    'kaden-channel',
    'livedoor-homme',
    'movie-enter',
    'peachy',
    'smax',
    'sports-watch',
    'topic-news'
]

# 各データの形式を整える
columns = ['label', 'label_name', 'file_path']
dataset_label_text = pd.DataFrame(columns=columns)
id2label = {} 
label2id = {}
for label, category in enumerate(category_list):
  
  file_names_list = sorted(glob.glob(f'{tutorial_path}/text/{category}/{category}*'))#対象メディアの記事が保存されているファイルのlistを取得します。
  print(f"{category}の記事を処理しています。　{category}に対応する番号は{label}で、データ個数は{len(file_names_list)}です。")

  id2label[label] = category
  label2id[category] = label
  
  for file in file_names_list:
      list = [[label, category, file]]
      df_append = pd.DataFrame(data=list, columns=columns)
      dataset_label_text = pd.concat([dataset_label_text, df_append], ignore_index=True, axis=0)

dataset_label_text.head()

# COMMAND ----------

# DBTITLE 1,ファイルの内容をロードする関数（Pandas UDF）
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType

@pandas_udf(StringType())
def read_text(paths: pd.Series) -> pd.Series:

  all_text = []
  for index, file in paths.items():             #取得したlistに従って実際のFileからデータを取得します。
      lines = open(file).read().splitlines()
      text = '\n'.join(lines[3:])               # ファイルの4行目からを抜き出す。
      all_text.append(text)

  return pd.Series(all_text)

# COMMAND ----------

# DBTITLE 1,Pandas DataframeからSpark Dataframeに変換し、ファイルの内容を新規列として追加
from pyspark.sql.functions import col

dataset_df = spark.createDataFrame(dataset_label_text)
dataset_df = dataset_df.withColumn('text', read_text(col('file_path')))
display(dataset_df.head(5))

# COMMAND ----------

# DBTITLE 1,次のセル（SQL）に渡すために必要な変数をシステム変数としてセット
spark.conf.set("my.catalogName", catalog_name)
spark.conf.set("my.schemaName", schema_name)

# COMMAND ----------

# DBTITLE 1,Unity Catalogにカタログとスキーマを無ければ作る
# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS ${my.catalogName};
# MAGIC USE CATALOG ${my.catalogName};
# MAGIC CREATE SCHEMA IF NOT EXISTS ${my.catalogName}.${my.schemaName};
# MAGIC USE SCHEMA ${my.schemaName};

# COMMAND ----------

# DBTITLE 1,Spark DataframeをDelta TableとしてUnity Catalogへ保存
dataset_df.write.mode("overwrite").saveAsTable(table_name_full_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1-2. データ加工
# MAGIC
# MAGIC Spark DataframeからHugging Face Datasetsへ変換をし、text列（文章データ）の内容をトークン化して、新たな列として追加します。
# MAGIC
# MAGIC Hugging Face の `datasets` は `datasets.Dataset.from_spark` を使って Spark DataFrames からのロードをサポートしています。[from_spark()](https://huggingface.co/docs/datasets/use_with_spark) メソッドの詳細については Hugging Face のドキュメントを参照してください。また、Databricksの[こちらのドキュメント](https://docs.databricks.com/ja/machine-learning/train-model/huggingface/load-data.html)も参考にしてください。
# MAGIC
# MAGIC Dataset.from_sparkはデータセットをキャッシュします。この例では、モデルはドライバ上で学習され、キャッシュされたデータはSparkを使用して並列化されるため、`cache_dir`はドライバとすべてのワーカーからアクセス可能でなければなりません。Databricks File System (DBFS)のルート([AWS](https://docs.databricks.com/dbfs/index.html#what-is-the-dbfs-root)|[Azure](https://learn.microsoft.com/azure/databricks/dbfs/#what-is-the-dbfs-root)|[GCP](https://docs.gcp.databricks.com/dbfs/index.html#what-is-the-dbfs-root))やマウントポイント([AWS](https://docs.databricks.com/dbfs/mounts.html)|[Azure](https://learn.microsoft.com/azure/databricks/dbfs/mounts)|[GCP](https://docs.gcp.databricks.com/dbfs/mounts.html))を利用することができます。
# MAGIC
# MAGIC DBFSを使用することで、モデル学習に使用する`transformers`互換のデータセットを作成する際に、"ローカル"パスを参照することができます。

# COMMAND ----------

# DBTITLE 1,データを学習用／検証用に分割し、HuggigFace Datasetに変換
dataset_df = spark.read.table(table_name_full_path)
(train_df, test_df) = dataset_df.persist().randomSplit([0.8, 0.2], seed=47)

import datasets
train_dataset = datasets.Dataset.from_spark(train_df, cache_dir="/dbfs/cache/train")
test_dataset = datasets.Dataset.from_spark(test_df, cache_dir="/dbfs/cache/test")

# COMMAND ----------

# MAGIC %md
# MAGIC 学習用のデータセットをトークン化してシャッフルする。また、学習処理では`text`カラムを必要としないので、データセットから削除します。本質的には削除しなくとも問題ないのですが、なぜか列を削除しないとトレーニング中に警告が出るため、あえて削除しています。
# MAGIC このステップでは、`datasets`は変換されたデータセットをローカルディスクにキャッシュし、モデルの学習時に高速に読み込めるようにする。

# COMMAND ----------

# DBTITLE 1,テキスト列からトークン列を作成（最終的にテキスト列は削除）
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(base_model)

def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        max_length=max_length,
        padding='max_length', 
        truncation=True)

train_tokenized = train_dataset.map(tokenize_function, batched=True).remove_columns(["text"])
test_tokenized = test_dataset.map(tokenize_function, batched=True).remove_columns(["text"])
train_dataset = train_tokenized.shuffle(seed=47)
test_dataset = test_tokenized.shuffle(seed=47)

# COMMAND ----------

# DBTITLE 1,Datasetsから1レコード取り出し、中身を見てみる
train_dataset[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1-3. モデルトレーニング

# COMMAND ----------

# MAGIC %md
# MAGIC ログに記録する評価指標を定義します。今回はAccuracyを記録します。損失（Loss）は設定せずとも自動でログに記録されます。

# COMMAND ----------

import numpy as np
import evaluate

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# COMMAND ----------

# MAGIC %md
# MAGIC 学習用のパラメーターはほぼデフォルト値を使用しますが、Epoch数（デフォルトは3）とバッチサイズ（デフォルトは8）のみデフォルトから変更します。他にも必要に応じて学習率などの多くの学習パラメータを設定できます。詳細は
# MAGIC [transformersドキュメント](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) を参照ください。

# COMMAND ----------

from transformers import TrainingArguments

training_output_dir = f"{tutorial_path}/bert_trainer"
training_args = TrainingArguments(
  output_dir=training_output_dir, 
  logging_dir = f"{tutorial_path}/logs",    # TensorBoard用にログを記録するディレクトリ
  evaluation_strategy="epoch",
  num_train_epochs=5)

training_args.set_dataloader(train_batch_size=12, eval_batch_size=32)

print(f"学習時のバッチサイズは　{training_args.per_device_train_batch_size}、検証時のバッチサイズは　{training_args.per_device_eval_batch_size}　です。")


# COMMAND ----------

# MAGIC %md
# MAGIC ラベルマッピングとクラス数を指定して、ベースモデルから学習するモデルを作成します。

# COMMAND ----------

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
  base_model, 
  num_labels=len(category_list), 
  label2id=label2id, 
  id2label=id2label)

# COMMAND ----------

# MAGIC %md
# MAGIC [data collator](https://huggingface.co/docs/transformers/main_classes/data_collator)を使うことで、訓練データと評価データセットの入力をバッチ化することができる。`DataCollatorWithPadding`をデフォルトで使用すると、テキスト分類のベースライン性能が良くなる。

# COMMAND ----------

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer)

# COMMAND ----------

# MAGIC %md
# MAGIC 上記で作成したモデル、引数、データセット、照合器、メトリクスを用いて、トレーナーオブジェクトを構築します。

# COMMAND ----------

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# COMMAND ----------

# MAGIC %md
# MAGIC モデルをサービングエンドポイントとしてデプロイするためのMLFlow登録用ラッパークラスです。
# MAGIC 本来はMLFlowのTransformerフレーバーを使用することでより簡単に登録できるのですが、それだとサービング時にGPUが使用されないケースがあるため、GPU使用をするにはこの実装が必要です。

# COMMAND ----------

import mlflow
import torch

class TextClassificationPipelineModel(mlflow.pyfunc.PythonModel):
  
  def __init__(self, model, tokenizer):
    device = 0 if torch.cuda.is_available() else -1
    self.pipeline = pipeline(
      "text-classification", 
      model=model, 
      tokenizer=tokenizer,
      batch_size=1,
      device=device)
    self.tokenizer = tokenizer
    
  def predict(self, context, model_input): 
    messages = model_input["text"].to_list()
    answers = self.pipeline(messages, max_length=max_length, padding='max_length', truncation=True)

    label_list = []
    score_list = []
    for answer in answers:
      label_list.append(answer['label'])
      score_list.append(str(answer['score']))

    return {"label": label_list, "score": score_list}

# COMMAND ----------

# MAGIC %md
# MAGIC モデルをトレーニングし、メトリクスと結果を MLflow に記録します。
# MAGIC MLFlowのTransformerフレーバーを使用することで、簡単に記録できます。

# COMMAND ----------

from transformers import pipeline

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

model_output_dir = f"{tutorial_path}/trained_model"
pipeline_output_dir = f"{tutorial_path}/trained_pipeline"

with mlflow.start_run() as run:
  
  # 学習開始。学習のメトリックが自動的にMLFLowにロギングされる
  trainer.train()

  # 学習終了後にモデルを保存
  trainer.save_model(model_output_dir)
  
  # 学習済みモデルを読み込んでパイプライン化して、更に保存。
  bert = AutoModelForSequenceClassification.from_pretrained(model_output_dir)

  # pipe = pipeline(
  #   "text-classification", 
  #   model=bert, 
  #   batch_size=1, 
  #   tokenizer=tokenizer,
  #   device=0)
  # pipe.save_pretrained(pipeline_output_dir)
  
  #######################################
  # CPUのみでのサービングで良ければこちらでも可
  #######################################
  # # MLFlow Trackingにパイプラインを記録する。
  # mlflow.transformers.log_model(
  #   transformers_model=pipe, 
  #   artifact_path=model_artifact_path+"_CPU", 
  #   input_example=["これはサンプル１です。", "これはサンプル２です。"],
  #   pip_requirements=["torch", "transformers", "accelerate", "sentencepiece", "datasets", "evaluate", "fugashi", "ipadic", "unidic-lite"],
  #   # registered_model_name=registered_model_name,
  #   model_config={ 
  #     "max_length": max_length, 
  #     "padding": "max_length", 
  #     "truncation": True 
  #   }
  # )

  # エンドポイントの入力と出力のスキーマを定義
  input_schema = Schema([ColSpec(DataType.string, "text")])
  output_schema = Schema([ColSpec(DataType.string, "label"), ColSpec(DataType.double, "score")])
  signature = ModelSignature(inputs=input_schema, outputs=output_schema)

  # 入力データのサンプルを用意
  input_example = pd.DataFrame({"text": ["これはサンプル１です。", "これはサンプル２です。"]})
  
  # モデルをMLFlow Trackingに記録
  mlflow.pyfunc.log_model(
      artifact_path=model_artifact_path,
      python_model=TextClassificationPipelineModel(bert, tokenizer),
      pip_requirements=["torch", "transformers", "accelerate", "sentencepiece", "datasets", "evaluate", "fugashi", "unidic-lite"],
      input_example=input_example,
      signature=signature
  )

print(f"モデルはMLFlow実験のRun(ID:{run.info.run_id})に記録されました。このIDを記録しておいてください。")

# COMMAND ----------

# DBTITLE 1,TensorBoard用にログのディレクトリーを指定
# MAGIC %load_ext tensorboard
# MAGIC experiment_log_dir = f"{tutorial_path}/logs"

# COMMAND ----------

# DBTITLE 1,TensorBoard起動
# MAGIC %tensorboard --logdir $experiment_log_dir

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1-4. 学習済みモデルで推論

# COMMAND ----------

# DBTITLE 1,テスト用のデータセット
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

# DBTITLE 1,学習済みモデルを使ってパイプラインを作成し推論実行
trained_pipe = pipeline(
    "text-classification", 
    model=AutoModelForSequenceClassification.from_pretrained(model_output_dir), 
    batch_size=4, 
    top_k=3,
    tokenizer=tokenizer,
    device=0)
res = trained_pipe(inputs)
res

# COMMAND ----------

# MAGIC %md
# MAGIC ## お疲れ様でした！
