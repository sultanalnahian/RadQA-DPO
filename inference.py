import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from radqa import RadQA
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score
from util import compute_f1_score
import argparse

def compute_metrics(decoded_labels,decoded_preds):

    accuracy = accuracy_score(decoded_labels, decoded_preds)
    f1_score = compute_f1_score(decoded_preds, decoded_labels)
   
    return {"accuracy": accuracy, "f1_score": f1_score}

class LMInference:

    def __init__(self, pretrained_model) -> None:

        self.SEQ_LENGTH = 1152

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

        outputs = self.model.generate(input_ids=encoded_input["input_ids"], max_length=128)
        answers = []
        for each_output in outputs:
            relation = self.tokenizer.decode(each_output, skip_special_tokens=True)
            relation = relation.replace("<pad>", "")
            relation = relation.strip()
            answers.append(relation)
        
            
        return answers

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/sft")
    parser.add_argument("--input_file", type=str, default="dataset/test.json")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    radQA = RadQA(args.input_file)
    model = LMInference(args.model_path)
    results = []
    predicted_answers = []
    original_answers = []
    wrong_no_prediction = []
    for item in tqdm(radQA.data):
        context = item['context']
        output = model.generate(context)
        original_answer = item['answer'].replace("\n","")
        predicted_answer = output[0]

        predicted_answers.append(predicted_answer)
        original_answers.append(original_answer)
        
    print(compute_metrics(original_answers, predicted_answers))