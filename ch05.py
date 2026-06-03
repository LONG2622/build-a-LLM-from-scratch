#开始微调6.3
#指令微调和分类微调
#准备数据集：下载和解压
import urllib.request
import zipfile
import os
import pandas as pd
from pathlib import Path
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollecion.tsv"

def download_and_unzip_spam_data(
        url, zip_path, extracted_path, data_tile_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping downloadand extraction.")
        return 
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File Downloaded and saved as {data_file_path}")
download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
df = pd.read_csv(
    data_file_path, sep = "\t", header = None, names = ["Label", "Text"]
)
df
print(df["Label"].value_counts())
#创建一个平衡的数据集
def create_balanced_dataset(df):
    num_spam = df[df["Label"]=="sapm"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state = 123
    )
    balanced_df = pd.concat([
        ham_subset, df[df["Label"] == "spam"]
    ])
    return balanced_df
balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())

#划分数据集
def random_split(df, train_frac, validation_frac):
    df = df.sample(
        frac = 1, random_state = 123
    ).reset_index(drop = True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df
train_df, validation_df, test_df= random_split(
    balanced_df, 0.7, 0.1)
#创建dateloader
import tiktoken
tokenizer = tiktoken.get_encoding()