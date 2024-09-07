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
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import json

from prepare_vectors import PrepareVectors

from logging import config
import logging
config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)
progress = logging.getLogger("progress")

if __name__ == "__main__":    

    with initialize(version_base=None, config_path="conf", job_name=__file__):
        #cfg = compose(config_name="qwen2")
        #cfg = compose(config_name="sentence-transformes")
        #cfg = compose(config_name="config.yaml", overrides=["+io=qwen2", "hydra.job.chdir=True", "hydra.run.dir=./outputtest"], return_hydra_config=True)
        #cfg = compose(config_name="config.yaml", overrides=[f"+io={lang_model}"], return_hydra_config=True)
        cfg = compose(config_name="config.yaml", return_hydra_config=True)
    logger.info(f"cfg={OmegaConf.to_yaml(cfg)}")

    # sentence-transformer バージョン
    input_model = cfg.io.input_model # glue
    input_filename = cfg.io.input_filename
    language_model = cfg.io.language_model
    revision = cfg.io.revision
    embedding_vector_dimension = cfg.io.embedding_vector_dimension
    train_dataset = load_dataset(input_model[0], input_model[1], split="train",
                                          revision=revision, num_proc=16)
    # メイン処理からみて入力ファイル名なので，準備処理としては出力ファイル名
    output = input_filename
    output_dir = os.path.splitext(os.path.dirname(f"{output}-langchaintest"))[0]        
    if not os.path.exists(output_dir):
        progress.info(f"Creating output directory({output_dir})")
        os.makedirs(output_dir)
    else:
        progress.info(f"{output_dir} found.")
    dim = embedding_vector_dimension
    #for i in range(self.numRows):
    sentence_transformers = True
    #sentence_transformers = False
    num = 10
    if sentence_transformers:
        progress.info("Sentence Transformers START")
        progress.info(f"Model({language_model}) loading...")
        model = SentenceTransformer(language_model,
                                     revision=revision,
                                    trust_remote_code=True,
                                    )
        logger.info(f"model={model}")
        progress.info(f"Prepare vectors with multi processing")
        sentence1 = train_dataset[0:num]['sentence1']
        sentence2 = train_dataset[0:num]['sentence2']
        label = train_dataset[0:num]['label']
        pool = model.start_multi_process_pool()
        embedding1 = model.encode_multi_process(sentence1,
                                                         pool)
        embedding2 = model.encode_multi_process(sentence2,
                                                         pool)
        model.stop_multi_process_pool(pool)
        df_st = pl.DataFrame(
                {
                    "sentence1": sentence1,
                    "embedding1": embedding1.tolist(),
                    "sentence2": sentence2,
                    "embedding2": embedding2.tolist(),
                    "label": label
                },
                schema=
                [
                    ("sentence1", pl.String),
                    ("embedding1", pl.Array(pl.Float64, dim)),
                    ("sentence2", pl.String),
                    ("embedding2", pl.Array(pl.Float64, dim)),
                    ("label", pl.Float64),
                ]

            )

        progress.info(f"df={df_st}")
        progress.info("Sentence Transformers END")

    langchain = True
    if langchain:
    # langchain version
        progress.info("LangChain START")
        from langchain_huggingface.embeddings import HuggingFaceEmbeddings
        model_kwargs = {"device": "gpu"}
        model = HuggingFaceEmbeddings(
            model_name=language_model,
            multi_process = True,
            show_progress = True,
        #    trust_remote_code=True,
        )
        sentence1 = train_dataset[0:num]['sentence1']
        sentence2 = train_dataset[0:num]['sentence2']
        label = train_dataset[0:num]['label']
        embedding1 = model.embed_documents(sentence1)
        embedding2 = model.embed_documents(sentence2)
        df_lc = pl.DataFrame(
                {
                    "sentence1": sentence1,
                    "embedding1": embedding1,
                    "sentence2": sentence2,
                    "embedding2": embedding2,
                    "label": label
                },
                schema=
                [
                    ("sentence1", pl.String),
                    ("embedding1", pl.Array(pl.Float64, dim)),
                    ("sentence2", pl.String),
                    ("embedding2", pl.Array(pl.Float64, dim)),
                    ("label", pl.Float64),
                ]

            )
        progress.info(f"df={df_lc}")
        progress.info("LangChain END")

    progress.info(f"Comparing...")
    progress.info(f"SentenceTransformer[0]={df_st[0]}")
    progress.info(f"LangChain[0]={df_lc[0]}")
