import json,  re, time, openai, ray, nltk, flytekit
from collections import Counter
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from openai import OpenAI
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split  # type: ignore
from tqdm import tqdm
from transformers import BertTokenizer
from flytekit import ImageSpec, Resources, Secret, task, workflow
from flytekit.core.workflow import WorkflowFailurePolicy  # type: ignore
from flytekit.testing import SecretsManager
from flytekitplugins.ray import HeadNodeConfig, RayJobConfig, WorkerNodeConfig

sns.set_theme()

#  Mounts the openai-key K8s secret as an environment variable. 
#Learn more: https://docs.flyte.org/en/latest/user_guide/productionizing/secrets.html#id4
secret=Secret(
    group="openai-key",
    key="OPENAI_API_KEY",
    mount_requirement=Secret.MountType.ENV_VAR,
)

SECRET_GROUP = "openai-key"
SECRET_NAME="OPENAI_API_KEY"
context = flytekit.current_context()
client=OpenAI(api_key=context.secrets.get(SECRET_GROUP, SECRET_NAME))

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
#@task(container_image=sklearn_image_spec,secret_requests=[Secret(group=SECRET_GROUP, key=SECRET_NAME)])
#def query_openai_endpoint()->str:
 #   context = flytekit.current_context()
    #   client=OpenAI(api_key=context.secrets.get(SECRET_GROUP, SECRET_NAME))
    #system_content = "you only answer in rhymes"  # system content (behavior)
    #assistant_content = ""  # assistant content (context)
    #user_content = "how are you"  # user content (message)
    #response = client.chat.completions.create(
    #model="gpt-4o-2024-08-06",
    #messages=[
    #    {"role": "system", "content": system_content},
    #    {"role": "assistant", "content": assistant_content},
    #    {"role": "user", "content": user_content},
    #],    
   # )
    #model_answer = model_response = response.choices[0].message.content
    #return model_answer

#Predict tags for a given sample    
def get_tag(model, system_content="", assistant_content="", user_content="")->str:
    try:
        response=client.chat.completions.create(
            model=model,        
            messages=[
                {"role": "system", "content": system_content},
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": user_content},
        ],
    )
        predicted_tag = response.choices[0].message.content
        return predicted_tag

    except (openai.error.ServiceUnavailableError, openai.error.APIError) as e:
        print(f"Error: {e}")
        return "error"

def get_predictions(inputs, model, system_content, assistant_content="")->list:
    y_pred=[]
    for item in tqdm(inputs):
        user_content=str(item)
        predicted_tag=get_tag(model=model, system_content=system_content, assistant_content=assistant_content, user_content=user_content)
        
        retry_count = 0
        max_retries = 5
        while predicted_tag is None and retry_count < max_retries:
            sleep_time = 2 ** retry_count  # Exponential backoff
            time.sleep(sleep_time)
            retry_count += 1
            predicted_tag=get_tag(model=model, system_content=system_content, assistant_content=assistant_content, user_content=user_content)
        
        if predicted_tag is None:
            predicted_tag = "error"  # Handle case where retries are exhausted

        y_pred.append(predicted_tag)
    return y_pred

def clean_predictions(y_pred, tags, default="other")->list:
    for i, item in enumerate(y_pred):
        if item not in tags:
            y_pred[i] = default
        if item.startswith("'") and item.endswith("'"):
            y_pred[i] = item[1:-1]
    return y_pred

def plot_tag_dist(y_true, y_pred):
    true_tag_freq = dict(Counter(y_true))
    pred_tag_freq = dict(Counter(y_pred))
    df_true = pd.DataFrame({"tag": list(true_tag_freq.keys()), "freq": list(true_tag_freq.values()), "source": "true"})
    df_pred = pd.DataFrame({"tag": list(pred_tag_freq.keys()), "freq": list(pred_tag_freq.values()), "source": "pred"})
    df = pd.concat([df_true, df_pred], ignore_index=True)
    #Plot the distribution of tags
    plt.figure(figsize=(10, 3))
    plt.title("Tag Distribution", fontsize=14)
    ax = sns.barplot(x="tag", y="freq", hue="source", data=df)
    ax.set_xticklabels(list(true_tag_freq.keys()), rotation=0, fontsize=8)
    plt.legend()
    plt.show()

@task(container_image=sklearn_image_spec,secret_requests=[Secret(group=SECRET_GROUP, key=SECRET_NAME)])
def evaluate(test_df, model, system_content, tags, assistant_content="")->Tuple[list,dict]:
    #Predictions
    y_test = test_df.tag.tolist()
    test_samples = test_df[["title", "description"]].to_dict(orient="records")
    y_pred = get_predictions(inputs=test_samples, model=model, system_content=system_content, assistant_content=assistant_content)
    y_pred = clean_predictions(y_pred=y_pred, tags=tags)
    
    #Performance metrics
    metrics = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    performance = {"precision": metrics[0], "recall": metrics[1], "f1-score": metrics[2]}
    print(json.dumps(performance, indent=2))
    plot_tag_dist(y_true=y_test, y_pred=y_pred)
    return y_pred, performance


@workflow
def complete_workflow()->Tuple[dict,str]:
    ingest_data= data_ingestion(DATASET_LOC=DATASET_LOC, HOLDOUT_LOC=HOLDOUT_LOC)
    train_df, val_df = split_dataset(df=ingest_data)
    cleaned_data = clean_data(df=ingest_data)
    encoded_data=encoding(df=ingest_data, train_df=train_df)
    preprocessed_data=tokenize(batch=cleaned_data)
    model_response=query_openai_endpoint()
    return preprocessed_data, model_response


