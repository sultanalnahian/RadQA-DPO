import pandas as pd
import csv
import numpy as np
import random
import argparse
from util import compute_f1_score


def get_preference_data(contexts, predicted_answers, original_answers, threshold):
    preference_data = []
    for i, context in enumerate(contexts):
        pred = predicted_answers[i]
        org_answer = original_answers[i]
        try:
            f1_score = compute_f1_score([pred], [org_answer])
            if pred != org_answer:
                print(i, f1_score)

            if f1_score <= threshold:
                item = dict()
                item['prompt'] = context
                item['chosen'] = original_answers[i]
                item['rejected'] = predicted_answers[i]
                preference_data.append(item)
        except Exception as e:
            print("prediction: ", pred)
    
    return preference_data

def is_in_dictionary(source, new_item):
    for each_item in source:
        if each_item['prompt'] == new_item['prompt'] and each_item['chosen'] == new_item['chosen'] and each_item['rejected'] == new_item['rejected']:
            return True
    
    return False
    


def create_preference_files_from_threshold(source_file, output_dir):
    data = pd.read_csv(open(source_file,'r'), delimiter="\t")
    contexts = data["prompt"]
    predicted_answers = data['rejected']
    original_answers = data['chosen']
    thresholds = [0.90, 0.80, 0.70, 0.60, 0.50]

    for threshold in thresholds:
        preference_data = get_preference_data(contexts, predicted_answers, original_answers, threshold)
        random.seed(42)
        random.shuffle(preference_data)
        fields = ['prompt', 'chosen', 'rejected']
        th1 = int(threshold*100)
        filepath = "{}/preference_threshold_{}.tsv".format(output_dir, th1)
        with open(filepath, 'w') as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=fields, delimiter='\t')
            csvwriter.writeheader()
            csvwriter.writerows(preference_data)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", type=str)
    parser.add_argument("--output_dir", type=str, default="dataset/")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    create_preference_files_from_threshold(args.source_file)
    