from pprint import pprint
import os
from params import Params
import pandas as pd
import numpy as np
import random
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, models
import polars as pl
import argparse
from logging import config
import logging
config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)


class PrepareVectors():
    """
        PrepareVectors(mode)
            mode: "original" or "finetuned"
    """
    params = Params("config.toml")
    input_data_filename_original = params.config["io"]["input_filename_original"]
    input_data_filename_finetuned = params.config["io"]["input_filename_finetuned"]
    random_vectors_by_original = params.config["io"]["random_vectors_by_original"]
    random_vectors_by_finetuned = params.config["io"]["random_vectors_by_finetuned"]
    original_model = params.config["io"]["original_model"]    
    finetuned_model = params.config["io"]["finetuned_model"]    
    def __init__(self, mode, model_load=True):
        #if os.path.isfile(self.input_data_filename):
        #    return
        revision = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
        self.train_dataset = load_dataset("glue", "stsb", split="train",
                                          revision=revision)
        self.valid_dataset = load_dataset("glue", "stsb", split="validation",
                                          revision=revision)
        self.numRows = len(self.train_dataset)

        if mode == "original":
            self.output = self.input_data_filename_original
            self.random_output = self.random_vectors_by_original
            if model_load:
                self.model = SentenceTransformer(self.original_model)
        elif mode == "finetuned":
            self.output = self.input_data_filename_finetuned
            self.random_output = self.random_vectors_by_finetuned
            if model_load:
                self.model = SentenceTransformer(self.finetuned_model)
        
        logger.debug(f"train example={self.train_dataset[0]}")
        logger.debug(f"valid example={self.valid_dataset[0]}")

    def getVectors(self, num):
        """
            numの数だけpolarsのデータフレームを返す
            args:
                num: int
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
        #if os.path.isfile(self.input_data_filename):
        #    return
            
        if os.path.isfile(self.output):
            df = pl.read_parquet(self.output).limit(num)
            logger.debug(f"loaded: {df.count()} rows.")
            return df
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
        logger.debug(f"df={df}")
        print(f"df={df}")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="original",
        help="original: original model, finetuned: fine-tuned model"
        )
    opt = parser.parse_args()
    mode = opt.mode
    
    vecs = PrepareVectors(mode, model_load=False)
    embeddings = vecs.getVectors(7)
    print("getVectors", embeddings)
    sample_idx = [0, 6]
    emb1 = embeddings.select("embedding1")[sample_idx].to_numpy()[:, 0]
    emb2 = embeddings.select("embedding2")[sample_idx].to_numpy()[:, 0]
    print("emb1 shape=", emb1.shape)
    print("emb1=", emb1)
    dif = np.abs(emb1-emb2)
    rel_tol = 1e-1
    #idx = np.where(dif <= threshhold)[0]
    idx = np.where(np.isclose(emb1, emb2, rtol=rel_tol))
    elm = dif[idx]
    print(idx)
    print(elm)
    print("original values")
    print(emb1[idx])
    print(emb2[idx])
