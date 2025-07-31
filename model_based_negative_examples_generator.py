import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from radqa import RadQA
import pandas as pd
from tqdm import tqdm
import csv
import random
import argparse
from util import compute_f1_score

class LMInference:

    def __init__(self, pretrained_model) -> None:

        self.SEQ_LENGTH = 1024

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        context: str):

        # print("Generating questions...\n")
        encoded_input = self.tokenizer(
            context,
            padding='max_length',
            max_length=self.SEQ_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model.generate(input_ids=encoded_input["input_ids"], max_length = 128)
        answers = []
        for each_output in outputs:
            relation = self.tokenizer.decode(each_output, skip_special_tokens=True)
            relation = relation.replace("<pad>", "")
            relation = relation.strip()
            answers.append(relation)
        
            
        return answers

def get_preference_data(contexts, predicted_answers, original_answers, threshold=0.9):
    preference_data = []
    for i, context in enumerate(contexts):
        pred = predicted_answers[i]
        org_answer = original_answers[i]
        f1_score = compute_f1_score([pred], [org_answer])
        if pred != org_answer:
            print(i, f1_score)
        # if f1_score > threshold and f1_score <= (threshold + 0.20):
        if f1_score <= threshold:
            item = dict()
            item['prompt'] = context
            item['chosen'] = original_answers[i]
            item['rejected'] = predicted_answers[i]
            preference_data.append(item)
    
    return preference_data

def is_in_dictionary(source, new_item):
    for each_item in source:
        if each_item['prompt'] == new_item['prompt'] and each_item['chosen'] == new_item['chosen'] and each_item['rejected'] == new_item['rejected']:
            return True
    
    return False
    

def merge_preference_file(file_list, directory, output_file):
    # read csv
    file_path = directory +"/" + file_list[0]
    data = pd.read_csv(open(file_path,'r'), delimiter="\t")
    # Convert the DataFrame to a Dictionary
    data_dict = data.to_dict(orient='records')
    for i in range(1,len(file_list)):
        file_path = directory +"/" + file_list[i]
        data = pd.read_csv(open(file_path,'r'), delimiter="\t")
        _data_dict = data.to_dict(orient='records')
        for each_item in _data_dict:
            if not is_in_dictionary(data_dict, each_item):
                data_dict.append(each_item)
    
    random.shuffle(data_dict)
    fields = ['prompt', 'chosen', 'rejected']
    filepath = directory +"/" + output_file
    with open(filepath, 'w') as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=fields, delimiter='\t')
        csvwriter.writeheader()
        csvwriter.writerows(data_dict)


def create_preference_files_from_threshold(source_file):
    data = pd.read_csv(open(source_file,'r'), delimiter="\t")
    contexts = data["prompt"]
    predicted_answers = data['rejected']
    original_answers = data['chosen']
    thresholds = [0.80, 0.70, 0.60, 0.50]
    for threshold in thresholds:
        preference_data = get_preference_data(contexts, predicted_answers, original_answers, threshold)
        random.seed(42)
        random.shuffle(preference_data)
        fields = ['prompt', 'chosen', 'rejected']
        th1 = int(threshold*100)
        th2 = th1+20
        filepath = "output/preference_dataset/threshold/train_preference_{}_{}.tsv".format(th1, th2)
        # filepath = "output/preference_dataset/threshold/train_preference_{}.tsv".format(th1)
        with open(filepath, 'w') as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=fields, delimiter='\t')
            csvwriter.writeheader()
            csvwriter.writerows(preference_data)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/dpo/checkpoint-1829")
    parser.add_argument("--input_file", type=str, default="dataset/train.json")
    parser.add_argument("--output_file", type=str, default="dataset/preference_train.tsv")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    model = LMInference(args.model)
    radQA = RadQA(args.input_file)
    
    results = []
    predicted_answers = []
    original_answers = []
    contexts = []
    for item in tqdm(radQA.data):
        context = item['context']
        output = model.generate(context)
        output_item = dict()
        output_item['context'] = context
        context_arr = context.split("<context>")
        context_arr = context_arr[0].split("<question>")
        question = context_arr[1].strip()
        output_item['question'] = question
        output_item['original_answer'] = item['answer']
        output_item['predicted_answer'] = output[0]
        results.append(output_item)
        original_answer = item['answer'].replace("\n","")
        predicted_answers.append(output[0])
        original_answers.append(original_answer)
        contexts.append(context)
        

    preference_data = get_preference_data(contexts, predicted_answers, original_answers)
    fields = ['prompt', 'chosen', 'rejected']
    filepath = args.output_file
    with open(filepath, 'w') as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=fields, delimiter='\t')
        csvwriter.writeheader()
        csvwriter.writerows(preference_data)
        

