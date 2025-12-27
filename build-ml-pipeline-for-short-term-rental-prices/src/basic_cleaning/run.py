#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd
from pandas import DataFrame as df


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    run = wandb.init(project="nyc_airbnb", group="eda", save_code=True)
    local_path = wandb.use_artifact("sample.csv:latest").file()
    df = pd.read_csv(local_path)

    # Drop outliers
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # upload to wandb
    df.to_csv("clean_sample.csv", index=False)
    artifact = wandb.Artifact(
    args.output_artifact,
    type=args.output_type,
    description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Input file artifact - e.g. sample.csv:latest",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Cleaned output file name - e.g. clean_sample.csv",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="WandB artifact type - e.g. clean_sample",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="WandB description of the output",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=int,
        help="Minimum cut off price",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=int,
        help="Maximum cut off price",
        required=True
    )


    args = parser.parse_args()

    go(args)
