import pandas as pd
from flytekit import task, workflow, ImageSpec, Resources # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from typing import Tuple
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

sklearn_image_spec = ImageSpec(
    base_image="ghcr.io/flyteorg/flytekit:py3.9-latest",
    packages=["scikit-learn","pandas","pyarrow","fastparquet","flytekit","nltk"],
    registry="ghcr.io/davidmirror-ops/images",
    platform="linux/arm64", 
)

DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"

@task(container_image=sklearn_image_spec)
def data_ingestion(DATASET_LOC: str) -> pd.DataFrame:
    df = pd.read_csv(DATASET_LOC)
    print(df.head())
    return df

@task(container_image=sklearn_image_spec)
def split_dataset(df:pd.DataFrame)->Tuple[pd.Series,pd.Series]:
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
def preprocessing(df:pd.DataFrame)->pd.DataFrame:
    df["text"] = df.title + " " + df.description
    original_df = df.copy()
    df.text = df.text.apply(clean_text)
    # DataFrame cleanup
    df = df.drop(columns=["id", "created_on", "title", "description"], errors="ignore")  # drop cols
    df = df.dropna(subset=["tag"])  # drop nulls
    df = df[["text", "tag"]]  # rearrange cols
    return df

@task(container_image=sklearn_image_spec)
def encoding(df:pd.DataFrame,train_df:pd.Series)->pd.DataFrame:
    # Label to index
    tags = train_df.tag.unique().tolist()
    num_classes = len(tags)
    class_to_index = {tag: i for i, tag in enumerate(tags)}
    # Encode labels
    df["tag"] = df["tag"].map(class_to_index)
    return df

@workflow
def complete_workflow()->pd.DataFrame:
    ingest_data= data_ingestion(DATASET_LOC=DATASET_LOC)
    train_df, val_df = split_dataset(df=ingest_data)
    cleaned_data = preprocessing(df=ingest_data)
    encoded_data=encoding(df=ingest_data, train_df=train_df)
    return encoded_data


