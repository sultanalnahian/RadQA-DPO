import json
from tqdm import tqdm
import pandas as pd
import random

random.seed(42)

class RadQA:
    def __init__(self, file):
        data = json.load(open(file, 'r'))
        self.data = self.load_data(data['data'])

    def load_data(self, data):
        qa_data = []
        for item in tqdm(data):
            paragraphs = item['paragraphs']
            for paragraph in paragraphs:
                questions = paragraph['qas']
                context = paragraph['context']
                for qas in questions:
                    question = qas['question']
                    answers = qas['answers']
                    possible = qas['is_impossible']
                    if len(answers) == 0:
                        new_item = dict()
                        new_item['context'] = "<context> "+ context + " "+ "<question> " + question
                        new_item['answer'] = "no_answer"
                        qa_data.append(new_item)
                    else:
                        for answer in answers:
                            new_item = dict()
                            new_item['context'] = "<context> "+ context + " "+ "<question> " + question
                            new_item['answer'] = answer['text']
                            qa_data.append(new_item)
        
        return qa_data
    
         

if __name__ == "__main__":
    radqa = RadQA("dataset/train.json")
    print(len(radqa.data))