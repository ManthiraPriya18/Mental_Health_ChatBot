from datasets import load_dataset
import os

def prepare_data():
    dataset = load_dataset("ShenLab/MentalChat16K")
    dataset['train'].to_csv('data/raw/mentalchat16k.csv', index=False)

if __name__ == '__main__':
    prepare_data()
