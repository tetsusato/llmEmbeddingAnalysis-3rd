from pprint import pprint
import os
from params import Params
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, models
import polars as pl
from logging import config
import logging
config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)


class PrepareVectors():
    params = Params("config.toml")
    input_data_filename = params.config["io"]["input_filename"]
    
    def __init__(self):
        if os.path.isfile(self.input_data_filename):
            return
        revision = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
        self.train_dataset = load_dataset("glue", "stsb", split="train",
                                          revision=revision)
        self.valid_dataset = load_dataset("glue", "stsb", split="validation",
                                          revision=revision)
        self.numRows = len(self.train_dataset)

        logger.debug(f"train example={self.train_dataset[0]}")
        logger.debug(f"valid example={self.valid_dataset[0]}")

    def getVectors(self):
        if os.path.isfile(self.input_data_filename):
            return
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
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        #for i in range(self.numRows):
        for i in range(100):
            sentence1 = self.train_dataset[i]['sentence1']
            sentence2 = self.train_dataset[i]['sentence2']
            label = self.train_dataset[i]['label']
            embedding1 = model.encode(sentence1)
            logger.debug(f"embedding1={embedding1}")
            logger.debug(f"len embedding1={len(embedding1)}")
            embedding2 = model.encode(sentence2)
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
        return df.write_parquet(self.input_data_filename)

        

if __name__ == "__main__":        
    vecs = PrepareVectors()
    embeddings = vecs.getVectors()
    print(embeddings)
    
