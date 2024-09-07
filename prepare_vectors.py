from pprint import pprint
import os
from params import Params
import pandas as pd
import numpy as np
import random
from datasets import load_dataset
#from sentence_transformers import SentenceTransformer, models
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import polars as pl
import argparse
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import json

from logging import config
import logging
config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)
progress = logging.getLogger("progress")

class PrepareVectors():
    """ PrepareVectors()

    """
    """
    params = Params("config.toml")
    # このクラスで出力する埋め込みベクトルの保存ファイル．通常はparquet
    input_data_filename_original = params.config["io"]["input_filename_original"]
    input_data_filename_finetuned = params.config["io"]["input_filename_finetuned"]
    random_vectors_by_original = params.config["io"]["random_vectors_by_original"]
    random_vectors_by_finetuned = params.config["io"]["random_vectors_by_finetuned"]
    language_model = params.config["io"]["language_model"]    
    input_model = params.config["io"]["input_model"]
    embedeing_vector_dimension = params.config["io"]["embedding_vector_dimension"]
    """
    def __init__(self, cfg: DictConfig, model_load=True):
        """ __init__
            args:
        """
        #print(json.dumps(cfg.__str__))
        print(cfg.__str__)
        self.input_model = cfg.io.input_model
        self.input_filename = cfg.io.input_filename
        self.random_vectors_filename = cfg.io.random_vectors_filename
        self.language_model = cfg.io.language_model
        self.revision = cfg.io.revision
        self.embedding_vector_dimension = cfg.io.embedding_vector_dimension
        #if os.path.isfile(self.input_data_filename):
        #    return
        revision = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
        """
        self.train_dataset = load_dataset("glue", "stsb", split="train",
                                          revision=revision)
        self.valid_dataset = load_dataset("glue", "stsb", split="validation",
                                          revision=revision)
        """
        self.train_dataset = load_dataset(self.input_model[0], self.input_model[1], split="train",
                                          revision=self.revision, num_proc=16)
        #self.valid_dataset = load_dataset(self.input_model[0], self.input_model[1], split="validation",
        #                                  revision=self.revision, num_proc=16)
        self.numRows = len(self.train_dataset)

        # メイン処理からみて入力ファイル名なので，準備処理としては出力ファイル名
        self.output = self.input_filename
        self.random_output = self.random_vectors_filename
        if model_load:
            progress.info(f"Model({self.language_model}) loading...")
            
            #self.model = SentenceTransformer(self.language_model,
            #                                 revision=self.revision   )
            #self.model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct")
            self.model = HuggingFaceEmbeddings(
                              model_name=self.language_model,
                              multi_process = True,
                              show_progress = True,
                              #trust_remote_code=True,
                              model_kwargs = {"trust_remote_code": True}
                              )
            logger.info(f"model={self.model}")
        
        logger.debug(f"train example={self.train_dataset[0]}")
        #logger.debug(f"valid example={self.valid_dataset[0]}")

    def getVectors(self,
                   num = -1,
                   cache_enable = True,
                   enable_multi_processing = True
                  ):
        """ getVectors
            self.modelの内容に基づくモデルを使い，
            self.input_modelの入力データから，
            numの数だけpolarsのデータフレームを返す．-1なら，モデルによってフルに返す
            args:
                num: int 返すベクトルの数
                cache_enable: bool ファイルキャッシュを使うかどうか
                reset: bool ファイルキャッシュを使う場合に，最初にリセットするかどうか
            return:
                pl.DataFrame(
                          schema=
                          [
                              ("sentence1", pl.String),
                              ("embedding1", pl.Array(pl.Float64, dim)),
                              ("sentence2", pl.String),
                              ("embedding2", pl.Array(pl.Float64, dim)),
                              ("label", pl.Float64)
                          ]
                         )
        """
        output_dir = os.path.splitext(os.path.dirname(self.output))[0]        
        if not os.path.exists(output_dir):
            progress.info(f"Creating output directory({output_dir})")
            os.makedirs(output_dir)
        if os.path.exists(self.output):
            progress.info(f"File found")
            logger.info(f"getVectors' results has been detected. load from ({self.output})")
            df = pl.read_parquet(self.output)
            logger.info(f"read df={df}")
            return df
        if num == -1:
            num = self.numRows
        #if os.path.isfile(self.input_data_filename):
        #    retukaburn
        if os.path.isfile(self.output):
            df = pl.read_parquet(self.output).limit(num)
            logger.debug(f"loaded: {df.count()} rows.")
            return df
        else:
            assert hasattr(self, 'model'), "language model should be defined"
        dim = self.embedding_vector_dimension
        df = pl.DataFrame(
                          schema=
                          [
                              ("sentence1", pl.String),
                              ("embedding1", pl.Array(pl.Float64, dim)),
                              ("sentence2", pl.String),
                              ("embedding2", pl.Array(pl.Float64, dim)),
                              ("label", pl.Float64)
                          ]
                         )
        logger.debug(f"df={df}")
        #for i in range(self.numRows):
        if enable_multi_processing:
            progress.info(f"Prepare vectors with multi processing")
            sentence1 = self.train_dataset[0:num]['sentence1']
            sentence2 = self.train_dataset[0:num]['sentence2']
            label = self.train_dataset[0:num]['label']
            """ # SentenceTransformer Version
            pool = self.model.start_multi_process_pool()
            embedding1 = self.model.encode_multi_process(sentence1,
                                                             pool)
            embedding2 = self.model.encode_multi_process(sentence2,
                                                             pool)
            self.model.stop_multi_process_pool(pool)
            """
            embedding1 = self.model.embed_documents(sentence1)
            embedding2 = self.model.embed_documents(sentence2)
            embsize = len(embedding1[0])
            df = pl.DataFrame(
                    {
                        "sentence1": sentence1,
                        #"embedding1": embedding1.tolist(),
                        "embedding1": embedding1,
                        "sentence2": sentence2,
                        #"embedding2": embedding2.tolist(),
                        "embedding2": embedding2,
                        "label": label
                    },
                    schema=
                    [
                        ("sentence1", pl.String),
                        ("embedding1", pl.Array(pl.Float64, embsize)),
                        ("sentence2", pl.String),
                        ("embedding2", pl.Array(pl.Float64, embsize)),
                        ("label", pl.Float64),
                    ]

                )

            logger.debug(f"df={df}")
        else:
            progress.info(f"Prepare vectors with sequentialnaru processing")            
            for i in range(num):
                sentence1 = self.train_dataset[i]['sentence1']
                sentence2 = self.train_dataset[i]['sentence2']
                label = self.train_dataset[i]['label']
                embedding1 = self.model.encode(sentence1)
                embedding2 = self.model.encode(sentence2)
                row = pl.DataFrame(
                        {
                            "sentence1": sentence1,
                            "embedding1": [embedding1],
                            "sentence2": sentence2,
                            "embedding2": [embedding2],
                            "label": label
                        },
                        schema=
                        [
                            ("sentence1", pl.String),
                            ("embedding1", pl.Array(pl.Float64, len(embedding1))),
                            ("sentence2", pl.String),
                            ("embedding2", pl.Array(pl.Float64, len(embedding2))),
                            ("label", pl.Float64),
                        ]

                    )
                logger.debug(f"row={row}")
                df = df.vstack(row)

        #print(f"df={df}")
        logger.info(f"df={df}, shape={df.shape}")
        output_dir = os.path.splitext(os.path.dirname(self.output))[0]
        logger.info(f"output_dir={output_dir}")
        # ディレクトリが存在しない場合は作成
        if not os.path.exists(output_dir):
            progress.info(f"Creating output directory({output_dir})")            
            os.makedirs(output_dir)

        progress.info(f"Writing  output file({self.output})")    
        df.write_parquet(self.output)
        #return df.write_parquet(self.output)
        return df
    def createRandomSentence(self, num = 1):
        if os.path.isfile(self.random_output):
            df = pl.read_parquet(self.random_output)
            logger.debug(f"loaded: {df.count()} rows.")
            return df
        pldataset = self.getVectors(num)
        print("dataset=", pldataset)
        plsentence1 = pldataset.select("sentence1")
        plsentence2 = pldataset.select("sentence2")
        sentence_list1 = sum(plsentence1.to_numpy().tolist(), [])
        sentence_list2 = sum(plsentence2.to_numpy().tolist(), [])

        dict = []
        sentence_list1 = sum([x.split(" ") for x in sentence_list1], [])
        sentence_list2 = sum([x.split(" ") for x in sentence_list2], [])
        dict.append(sentence_list1)
        dict.append(sentence_list2)
        dict = sum(dict, [])
        dict = list(set(dict))
        dict_num = len(dict)
        random_sentence1 = []
        random_sentence2 = []
        rng = np.random.default_rng(1234)
        dim = 384
        df = pl.DataFrame(
                          schema=
                          [
                              ("sentence1", pl.String),
                              ("embedding1", pl.Array(pl.Float64, dim)),
                              ("sentence2", pl.String),
                              ("embedding2", pl.Array(pl.Float64, dim)),
                              ("label", pl.Float64)
                          ]
                         )
        for i in range(num):
            sentence1 = plsentence1[i].to_numpy().tolist()[0][0].split(" ")
            print("sentence1=", sentence1)
            rnd_words = [dict[random.randint(0, dict_num - 1)] for x in  sentence1]
            print("rnd words=", rnd_words)
            rnd_words1 = " ".join(rnd_words)
            embedding1 = self.model.encode(rnd_words1)
            random_sentence1.append(rnd_words)
            sentence2 = plsentence2[i].to_numpy().tolist()[0][0].split(" ")
            rnd_words = [dict[random.randint(0, dict_num - 1)] for x in  sentence2]
            random_sentence2.append(rnd_words)
            rnd_words2 = " ".join(rnd_words)
            embedding2 = self.model.encode(rnd_words2)
            label = 5.0
            row = pl.DataFrame(
                    {
                        "sentence1": rnd_words1,
                        "embedding1": [embedding1],
                        "sentence2": rnd_words2,
                        "embedding2": [embedding2],
                        "label": label
                    },
                    schema=
                    [
                        ("sentence1", pl.String),
                        ("embedding1", pl.Array(pl.Float64, len(embedding1))),
                        ("sentence2", pl.String),
                        ("embedding2", pl.Array(pl.Float64, len(embedding2))),
                        ("label", pl.Float64),
                    ]

                )
            print("row=", row)
            df = df.vstack(row)
        df.write_parquet(self.random_output)
        return df
    def getRandomVectors():
        dim = 384
        df = pl.DataFrame(
                          schema=
                          [
                              ("sentence1", pl.String),
                              ("embedding1", pl.Array(pl.Float64, dim)),
                              ("sentence2", pl.String),
                              ("embedding2", pl.Array(pl.Float64, dim)),
                              ("label", pl.Float64)
                          ]
                         )
        logger.debug(f"df={df}")
        sentence_list = self.train_dataset[:, "sentence1"]
        sentence_list += self.train_dataset[:, "sentence2"]
        #for i in range(self.numRows):
        for i in range(num):
            sentence1 = self.train_dataset[i]['sentence1']
            sentence2 = self.train_dataset[i]['sentence2']
            label = self.train_dataset[i]['label']
            embedding1 = self.model.encode(sentence1)
            logger.debug(f"embedding1={embedding1}")
            logger.debug(f"len embedding1={len(embedding1)}")
            embedding2 = self.model.encode(sentence2)
            row = pl.DataFrame(
                    {
                        "sentence1": sentence1,
                        "embedding1": [embedding1],
                        "sentence2": sentence2,
                        "embedding2": [embedding2],
                        "label": label
                    },
                    schema=
                    [
                        ("sentence1", pl.String),
                        ("embedding1", pl.Array(pl.Float64, len(embedding1))),
                        ("sentence2", pl.String),
                        ("embedding2", pl.Array(pl.Float64, len(embedding2))),
                        ("label", pl.Float64),
                    ]

                )
            logger.debug(f"row={row}")
            df = df.vstack(row)
        
    def describe_each_row(self, vec):
        describedf = None
        for row in range(vec.shape[0]):
            #print(vec[row])
            df = pd.DataFrame(pd.Series(vec[row]).describe()).transpose()
            # print("describe=", df)
            if describedf is None:
                describedf = df
            else:
                describedf = pd.concat([describedf, df])
        return describedf

"""
@hydra.main(config_name="config", version_base=None, config_path="conf")
def load_config(cfg: DictConfig) -> DictConfig:
    return cfg
"""
if __name__ == "__main__":
    with initialize(version_base=None, config_path="conf", job_name=__file__):
        #cfg = compose(config_name="qwen2")
        #cfg = compose(config_name="sentence-transformes")
        cfg = compose(config_name="config.yaml",
                      overrides=["hydra.job.chdir=True"],
                      return_hydra_config=True
                      )
    logger.info(f"initial cfg=\n{OmegaConf.to_yaml(cfg)}")
    vecs = PrepareVectors(cfg, model_load=True)
    # embeddings = vecs.getVectors(7, cache_enable=False)
    #embeddings = vecs.getVectors(1000, cache_enable=False, enable_multi_processing=True)
    embeddings = vecs.getVectors(-1, cache_enable=False, enable_multi_processing=True)
    print("getVectors", embeddings)
    sample_idx = [0, 6]
    emb1 = embeddings.select("embedding1")[sample_idx].to_numpy()[:, 0]
    emb2 = embeddings.select("embedding2")[sample_idx].to_numpy()[:, 0]
    print("emb1 shape=", emb1.shape)
    #print("emb1=", emb1)
    dif = np.abs(emb1-emb2)
    rel_tol = 1e-1
    #idx = np.where(dif <= threshhold)[0]
