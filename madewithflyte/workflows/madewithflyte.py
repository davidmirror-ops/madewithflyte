import pandas as pd
import flytekit
from flytekit import task, workflow, ImageSpec, Resources, Secret
from flytekit.core.workflow import WorkflowFailurePolicy # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from typing import Tuple
from flytekitplugins.ray import HeadNodeConfig, RayJobConfig, WorkerNodeConfig
from flytekit.testing import SecretsManager
import nltk, re, ray, openai, json, time, os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from transformers import BertTokenizer
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv


secret=Secret(
    group="openai-key",
    key="OPENAI_API_KEY",
    mount_requirement=Secret.MountType.ENV_VAR,
)

SECRET_GROUP = "openai-key"
SECRET_NAME="OPENAI_API_KEY"



ray.data.DatasetContext.get_current().execution_options.preserve_order = True  # deterministic

ray_config = RayJobConfig(
    head_node_config=HeadNodeConfig(ray_start_params={"log-color": "True"}),
    worker_node_config=[WorkerNodeConfig(group_name="ray-group", replicas=1)],
    runtime_env="../requirements.txt",
    enable_autoscaling=True,
    shutdown_after_job_finishes=True,
    ttl_seconds_after_finished=3600,
)
sklearn_image_spec = ImageSpec(
    base_image="ghcr.io/flyteorg/flytekit:py3.9-latest",
    requirements="../requirements.txt",
    registry="ghcr.io/davidmirror-ops/images",
    #platform="linux/arm64", 
)

DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"
HOLDOUT_LOC = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/holdout.csv"

@task(container_image=sklearn_image_spec)
def data_ingestion(DATASET_LOC: str, HOLDOUT_LOC:str) -> pd.DataFrame:
    df = pd.read_csv(DATASET_LOC)
    test_df = pd.read_csv(HOLDOUT_LOC)
    print(df.head(), flush=True)
    return df

@task(container_image=sklearn_image_spec)
def split_dataset(df:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame]:
    test_size=0.2
    train_df, val_df = train_test_split(df, stratify=df.tag,test_size=test_size, random_state=1234)
    train_value_counts=train_df.tag.value_counts()
    validation_value_counts=val_df.tag.value_counts()*int((1-test_size)/test_size)
    return train_df, val_df

nltk.download("stopwords")
STOPWORDS = stopwords.words("english")
def clean_text(text, stopwords=STOPWORDS):
    """Clean raw text string."""
    # Lower
    text = text.lower()

    # Remove stopwords
    pattern = re.compile(r'\b(' + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub('', text)

    # Spacing and filters
    text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)  # add spacing
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()  # strip white space at the ends
    text = re.sub(r"http\S+", "", text)  #  remove links
    return text

@task(container_image=sklearn_image_spec)
def clean_data(df:pd.DataFrame)->pd.DataFrame:
    df["text"] = df.title + " " + df.description
    original_df = df.copy()
    df.text = df.text.apply(clean_text)
    # DataFrame cleanup
    df = df.drop(columns=["id", "created_on", "title", "description"], errors="ignore")  # drop cols
    df = df.dropna(subset=["tag"])  # drop nulls
    df = df[["text", "tag"]]  # rearrange cols
    return df

@task(container_image=sklearn_image_spec)
def encoding(df:pd.DataFrame,train_df:pd.DataFrame)->pd.Series:
    # Label to index
    tags = train_df.tag.unique().tolist()
    num_classes = len(tags)
    class_to_index = {tag: i for i, tag in enumerate(tags)}
    # Encode labels
    df["tag"] = df["tag"].map(class_to_index)
    return df["tag"]

def decode(indices:int, index_to_class:dict)->dict:
    return [index_to_class[index] for index in indices]

@task(container_image=sklearn_image_spec)
def tokenize(batch:pd.DataFrame)->dict:
    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    encoded_inputs = tokenizer(batch["text"].tolist(), return_tensors="np", padding="longest")
    return {
        "ids": encoded_inputs["input_ids"].tolist(),  # Convert numpy array to list
        "masks": encoded_inputs["attention_mask"].tolist(),  # Convert numpy array to list
        "targets": batch["tag"].tolist()  # Convert pandas Series to list
    }
@task(container_image=sklearn_image_spec,secret_requests=[Secret(group=SECRET_GROUP,key=SECRET_NAME)])
def query_openai_endpoint()->dict:
    context = flytekit.current_context()
    client=OpenAI(api_key=context.secrets.get(SECRET_GROUP, SECRET_NAME))
    system_content = "you only answer in rhymes"  # system content (behavior)
    assistant_content = ""  # assistant content (context)
    user_content = "how are you"  # user content (message)
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": system_content},
        {"role": "assistant", "content": assistant_content},
        {"role": "user", "content": user_content},
    ],
)
    print (response.to_dict()["choices"][0].to_dict()["message"]["content"], flush=True)

@workflow
def complete_workflow(failure_policy=WorkflowFailurePolicy.FAIL_AFTER_EXECUTABLE_NODES_COMPLETE)->Tuple[dict,dict]:
    ingest_data= data_ingestion(DATASET_LOC=DATASET_LOC, HOLDOUT_LOC=HOLDOUT_LOC)
    train_df, val_df = split_dataset(df=ingest_data)
    cleaned_data = clean_data(df=ingest_data)
    encoded_data=encoding(df=ingest_data, train_df=train_df)
    preprocessed_data=tokenize(batch=cleaned_data)
    model_response=query_openai_endpoint()
    return preprocessed_data, model_response


