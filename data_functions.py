import pandas as pd


def get_data(url):
    raw_dataset = pd.read_csv(url)
    dataset = raw_dataset.dropna()
    dataset = dataset.reset_index(drop=True)
    return dataset


def inspect_data(dataset):
    print('Dataset shape:')
    print(dataset.shape)

    print('Tail:')
    print(dataset.tail())

    print('Statistics:')
    print(dataset.describe().transpose())