import pandas as pd
import torch
from transformers import AutoTokenizer

class DataLoader:
    def __init__(self, train_path, valid_path, model_name, max_length=128):
        self.train_path = train_path
        self.valid_path = valid_path
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("device:", self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def load_data(self):
        train_data = pd.read_csv(self.train_path)
        valid_data = pd.read_csv(self.valid_path)
        print('중복 제거 전 학습 데이터셋 : {}'.format(len(train_data)))
        print('중복 제거 전 테스트 데이터셋 : {}'.format(len(valid_data)))

        train_data.drop_duplicates(subset=["comments"], inplace=True)
        valid_data.drop_duplicates(subset=["comments"], inplace=True)
        print('중복 제거 후 학습 데이터셋 : {}'.format(len(train_data)))
        print('중복 제거 후 테스트 데이터셋 : {}'.format(len(valid_data)))

        return train_data, valid_data

    def tokenize_data(self, data):
        return self.tokenizer(
            list(data["comments"]),
            return_tensors="pt",
            max_length=self.max_length,
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )

    def prepare_datasets(self, train_data, valid_data):
        tokenized_train = self.tokenize_data(train_data)
        tokenized_valid = self.tokenize_data(valid_data)

        train_dataset = CourseDataset(tokenized_train, train_data["label"].values)
        valid_dataset = CourseDataset(tokenized_valid, valid_data["label"].values)

        return train_dataset, valid_dataset


class CourseDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["label"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def main():
    MODEL_NAME = "beomi/KcELECTRA-base"
    TRAIN_PATH = '../Profanity_Filter/data/datasets/Train_data.csv'
    VALID_PATH = '../Profanity_Filter/data/datasets/Valid_data.csv'

    data_loader = DataLoader(TRAIN_PATH, VALID_PATH, MODEL_NAME)
    train_data, valid_data = data_loader.load_data()
    train_dataset, valid_dataset = data_loader.prepare_datasets(train_data, valid_data)
    return train_dataset, valid_dataset, MODEL_NAME


# Example usage
if __name__ == "__main__":
    main()




