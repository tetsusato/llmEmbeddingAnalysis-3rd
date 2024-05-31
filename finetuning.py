from datasets import load_dataset
from sentence_transformers.losses import CoSENTLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
    models,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator

import os
from params import Params

from logging import config
import logging
config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)

class Finetuning():
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

if __name__ == "__main__":
    ft = Finetuning()
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model.to("cuda:0")
    loss = CoSENTLoss(model)
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir="models/mpnet-base-all-nli-triplet",
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # losses that use "in-batch negatives" benefit from no duplicates
        # Optional tracking/debugging parameters:
        #eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        run_name="mpnet-base-all-nli-triplet",  # Will be used in W&B if `wandb` is installed
    )
