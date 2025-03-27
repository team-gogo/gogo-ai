import pandas as pd
import re
import os
 
class CDDataProcessor:
    def __init__(self, input_file, output_file, column_names):
        self.input_file = input_file
        self.output_file = output_file
        self.column_names = column_names

    def preprocess_cd_data(self):
        with open(self.input_file, 'r', encoding='utf-8') as f_in, \
             open(self.output_file, 'w', encoding='utf-8', newline='') as f_out:
            for line in f_in:
                line = line.strip()
                line = line.replace('"', '').replace(',', '')
                last_pipe_index = line.rfind('|')
                if last_pipe_index != -1:
                    line_replace = line[:last_pipe_index] + ',' + line[last_pipe_index + 1:]
                f_out.write(line_replace + '\n')

    def load_and_split_data(self, train_frac=0.8, random_state=42):
        df = pd.read_csv(self.output_file, names=self.column_names)
        train_data = df.sample(frac=train_frac, random_state=random_state)
        valid_data = df.drop(train_data.index)
        return train_data, valid_data

    def preprocess_and_split_cd_data(self):
        self.preprocess_cd_data()
        return self.load_and_split_data()


class KHDataProcessor:
    @staticmethod
    def num_padding(filedir):
        df = pd.read_csv(filedir)
        df['label'] = df['label'].replace({'none': 0, 'hate': 1, 'offensive': 2})
        return df

    @staticmethod
    def load_kh_data(train_file, valid_file):
        KH_train_data = KHDataProcessor.num_padding(filedir=train_file)
        KH_valid_data = KHDataProcessor.num_padding(filedir=valid_file)
        return KH_train_data, KH_valid_data


class DataMerger:
    @staticmethod
    def merge_and_save_data(CD_train_data, CD_valid_data, KH_train_data, KH_valid_data, train_output, valid_output):
        Train_data = pd.concat([CD_train_data, KH_train_data], ignore_index=True)
        Valid_data = pd.concat([CD_valid_data, KH_valid_data], ignore_index=True)
        Train_data.to_csv(train_output, index=False)
        Valid_data.to_csv(valid_output, index=False)


def main():
    input_file = '../Profanity_Filter/data/raw_datasets/CD_data/dataset.txt'
    output_file = '../Profanity_Filter/data/raw_datasets/CD_data/CD_data.csv'
    column_names = ['comments', 'label']

    cd_processor = CDDataProcessor(input_file, output_file, column_names)
    CD_train_data, CD_valid_data = cd_processor.preprocess_and_split_cd_data()

    KH_train_data, KH_valid_data = KHDataProcessor.load_kh_data(
        '../Profanity_Filter/data/raw_datasets/KH_data/train.hate.csv',
            '../Profanity_Filter/data/raw_datasets/KH_data/dev.hate.csv'
)

    DataMerger.merge_and_save_data(
        CD_train_data, CD_valid_data, KH_train_data, KH_valid_data,
            '../Profanity_Filter/data/datasets/Train_data.csv',
            '../Profanity_Filter/data/datasets/Valid_data.csv'
        )

    Test_data = pd.read_csv('../Profanity_Filter/data/raw_datasets/KH_data/test.hate.no_label.csv')
    Test_data.to_csv('../Profanity_Filter/data/datasets/Test.csv', index=False)


if __name__ == '__main__':
    main()