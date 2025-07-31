import itertools
import re
import random
import pandas as pd
from tqdm import tqdm
import csv
import json
from nltk.tokenize import word_tokenize, sent_tokenize

def get_random_relation(data):
    r_in = random.randint(0, len(data)-1)
    item = data.iloc[r_in]
    return item.iloc[1]

def create_preference_data(sequence, original_relation, negative_relations):
    data = []
    for neg_rel in negative_relations:
        new_item = dict()    
        new_item['prompt'] = sequence
        new_item['chosen'] = original_relation
        new_item['rejected'] = neg_rel
        data.append(new_item)
    
    return data
def load_data(file):
    data = json.load(open(file, 'r'))
    qa_data = []
    max_token = 400
    answer_tokens = []
    i = 0
    data_len = int((len(data['data']) * 25)/100)
    for item in tqdm(data['data'][:data_len]):
        paragraphs = item['paragraphs']
        for paragraph in paragraphs:
            paragraph_data = []
            questions = paragraph['qas']
            context = paragraph['context']
            tokenized_context = word_tokenize(context)
            num_tokens = len(tokenized_context)
            if num_tokens>max_token:
                i+=1
                max_token = num_tokens
                print(i)
            for qas in questions:
                question = qas['question']
                answers = qas['answers']
                
                possible = qas['is_impossible']
                if len(answers) == 0:
                    new_item = dict()
                    # new_item['context'] = question + " <context> " + context
                    # new_item['question'] = question
                    new_item['context'] = context
                    new_item['question'] = question
                    new_item['answer'] = "no_answer"
                    paragraph_data.append(new_item)
                else:
                    for answer in answers:
                        tokenized_answers = word_tokenize(answer['text'])
                        answer_tokens.append(len(tokenized_answers))
                        # if max_answer_token < len(tokenized_answers):
                        #     max_answer_token = len(tokenized_answers)
                        new_item = dict()
                        # new_item['context'] = question + " <context> " + context
                        new_item['context'] = context
                        new_item['question'] = question
                        new_item['answer'] = answer['text']
                        paragraph_data.append(new_item)
            qa_data.append(paragraph_data)
    
    print(max_token)
    
    return qa_data

def answer_combination_with_previous_words(answer, context, k = 10, start=4):
    _range = random.randint(6,max(k,6))
    splitted_context = context.split(answer)
    splitted_words = splitted_context[0].split(" ")
    total_words = len(splitted_words)
    new_answers = []
    # new_answers.append(answer)
    for i in range(min(start,total_words),min(_range, total_words+1)):
        words = splitted_words[total_words-i:]
        new_words = " ".join(words)
        if new_words.strip() != "":
            new_answer = " ".join(words) + answer
            new_answers.append(new_answer)
    
    if len(new_answers)>0:
        rand_in = random.randint(0,len(new_answers)-1)
        return [new_answers[rand_in]]
    return new_answers

def answer_combination_with_next_words(answer, context, k = 10, start=3):
    _range = random.randint(6,max(k,6))
    splitted_context = context.split(answer)
    splitted_words = splitted_context[1].split(" ")
    total_words = len(splitted_words)
    new_answers = []
    # new_answers.append(answer)
    for i in range(min(start,total_words),min(_range, total_words+1)):
        words = splitted_words[:i]
        new_words = " ".join(words)
        if new_words.strip() != "":
            new_answer = answer+" ".join(words)
            new_answers.append(new_answer)
    
    if len(new_answers)>0:
        rand_in = random.randint(0,len(new_answers)-1)
        return [new_answers[rand_in]]
    return new_answers

def remove_begining_words_from_answer(answer, k = 6, start = 2):
    splitted_words = answer.split(" ")
    max_remove = min(k, len(splitted_words)-2)
    new_answers = []
    for i in range(start,max_remove):
        new_answer = " ".join(splitted_words[i:])
        new_answers.append(new_answer)
    
    if len(new_answers)>0:
        rand_in = random.randint(0,len(new_answers)-1)
        return [new_answers[rand_in]]
    return new_answers

def remove_end_words_from_answer(answer, k = 6, start = 2):
    splitted_words = answer.split(" ")
    max_remove = min(k, len(splitted_words)-2)
    new_answers = []
    for i in range(start, max_remove):
        new_answer = " ".join(splitted_words[:len(splitted_words)-i])
        new_answers.append(new_answer)
    
    if len(new_answers)>0:
        rand_in = random.randint(0,len(new_answers)-1)
        return [new_answers[rand_in]]
    return new_answers

def is_sentence_in_answers(sentence, answers):
    for answer in answers:
        if answer in sentence:
            return True
        if sentence in answer:
            return True
    
    return False
def get_random_sentence_from_context(answers, context):
    sentences = sent_tokenize(context)
    n = len(sentences)
    k = 0
    sentence = ""
    while k < 10:
        rand_i = random.randint(0,n-1)
        sentence = sentences[rand_i]
        words = word_tokenize(sentence)
        if len(words)>3 and not is_sentence_in_answers(sentence, answers):
            return sentence
        k +=1
        sentence = ""
    return sentence

def get_preference_data_item(context, question, chosen_answer, rejected_answer):
    item = dict()
    item['prompt'] = "<context> "+context + " "+ "<question> " + question
    item['chosen'] = chosen_answer
    item['rejected'] = rejected_answer
    return item

def create_negative_answers_combination(new_answer, context, remove_k = 6, add_k = 8):
    all_answers = []
    # 1. Remove few words from the end of the answer text
    removed_end_words_answers = remove_end_words_from_answer(new_answer, remove_k)
    all_answers.extend(removed_end_words_answers)
                        
    # 2. Add few words with the begining of the answer text
    previous_answer_combination = answer_combination_with_previous_words(new_answer, context, add_k)
    all_answers.extend(previous_answer_combination)
    for each_answer in removed_end_words_answers:
        previous_answer_combination = answer_combination_with_previous_words(each_answer, context, add_k)
        all_answers.extend(previous_answer_combination)
                        
    # 3. Remove few words from the beginning of the answer text
    removed_beginning_words_answers = remove_begining_words_from_answer(new_answer, remove_k)
    all_answers.extend(removed_beginning_words_answers)

    # 4. Add few words with the end of the answer text
    next_answer_combination = answer_combination_with_next_words(new_answer,context, add_k)
    all_answers.extend(next_answer_combination)
    for each_answer in removed_beginning_words_answers:
        next_answer_combination = answer_combination_with_next_words(each_answer, context, add_k)
        all_answers.extend(next_answer_combination)

    random.shuffle(all_answers)
    return all_answers[:max(len(all_answers), 2)]

def create_positive_answers_combination(new_answer, context, remove_k = 4, add_k = 4):
    
    def positive_examples_answer_combination_with_previous_words(answer, context, k = 4, start=1):
        _range = random.randint(1,k)
        splitted_context = context.split(answer)
        splitted_words = splitted_context[0].split(" ")
        total_words = len(splitted_words)
        new_answers = []
        # new_answers.append(answer)
        for i in range(min(start,total_words),min(_range, total_words+1)):
            words = splitted_words[total_words-i:]
            new_words = " ".join(words)
            if new_words.strip() != "":
                new_answer = " ".join(words) + answer
                new_answers.append(new_answer)
        
        if len(new_answers)>0:
            rand_in = random.randint(0,len(new_answers)-1)
            return [new_answers[rand_in]]
        return new_answers

    def positive_examples_answer_combination_with_next_words(answer, context, k = 4, start=1):
        _range = random.randint(2,k)
        splitted_context = context.split(answer)
        splitted_words = splitted_context[1].split(" ")
        total_words = len(splitted_words)
        new_answers = []
        # new_answers.append(answer)
        for i in range(min(start,total_words),min(_range, total_words+1)):
            words = splitted_words[:i]
            new_words = " ".join(words)
            if new_words.strip() != "":
                new_answer = answer+" ".join(words)
                new_answers.append(new_answer)
        
        if len(new_answers)>0:
            rand_in = random.randint(0,len(new_answers)-1)
            return [new_answers[rand_in]]
        return new_answers

    def positive_examples_remove_begining_words_from_answer(answer, k = 4, start = 1):
        splitted_words = answer.split(" ")
        max_remove = min(k, len(splitted_words)-2)
        new_answers = []
        for i in range(start,max_remove):
            new_answer = " ".join(splitted_words[i:])
            new_answers.append(new_answer)
        
        if len(new_answers)>0:
            rand_in = random.randint(0,len(new_answers)-1)
            return [new_answers[rand_in]]
        return new_answers

    def positive_examples_remove_end_words_from_answer(answer, k = 4, start = 1):
        splitted_words = answer.split(" ")
        max_remove = min(k, len(splitted_words)-2)
        new_answers = []
        for i in range(start, max_remove):
            new_answer = " ".join(splitted_words[:len(splitted_words)-i])
            new_answers.append(new_answer)
        
        if len(new_answers)>0:
            rand_in = random.randint(0,len(new_answers)-1)
            return [new_answers[rand_in]]
        return new_answers

    all_answers = []
    # 1. Remove few words from the end of the answer text
    removed_end_words_answers = positive_examples_remove_end_words_from_answer(new_answer, remove_k)
    all_answers.extend(removed_end_words_answers)
                        
    # 2. Add few words with the begining of the answer text
    previous_answer_combination = positive_examples_answer_combination_with_previous_words(new_answer, context)
    all_answers.extend(previous_answer_combination)
    for each_answer in removed_end_words_answers:
        previous_answer_combination = positive_examples_answer_combination_with_previous_words(each_answer, context)
        all_answers.extend(previous_answer_combination)
                        
    # 3. Remove few words from the beginning of the answer text
    removed_beginning_words_answers = positive_examples_remove_begining_words_from_answer(new_answer, remove_k)
    all_answers.extend(removed_beginning_words_answers)

    # 4. Add few words with the end of the answer text
    next_answer_combination = positive_examples_answer_combination_with_next_words(new_answer,context, add_k)
    all_answers.extend(next_answer_combination)
    for each_answer in removed_beginning_words_answers:
        next_answer_combination = positive_examples_answer_combination_with_next_words(each_answer, context, add_k)
        all_answers.extend(next_answer_combination)

    max_answer = 6
    if len(all_answers)> max_answer:
        random.shuffle(all_answers)
        all_answers = all_answers[:max_answer]
    return all_answers


def assign_answers_on_no_answer(data):
    new_data = []
    total_no_anser = 0
    for paragraphs in data:
        original_answers = []
        for item in paragraphs:
            original_answers.append(item['answer'])
        for item in paragraphs:
            context = item["context"]
            question = item["question"]
            answer = item['answer']
            if answer == "no_answer":
                total_no_anser +=1
                new_answer = ""
                no_item = len(paragraphs)
                all_answers = []
                for i in range(no_item):
                    new_answer = paragraphs[i]["answer"]
                    if new_answer != "no_answer":
                        # Get negative answer combination using an answer of a different question
                        combinations_from_different_answer = create_negative_answers_combination(new_answer, context, 4, 7)
                        all_answers.extend(combinations_from_different_answer[:1])
                        
                # Get negative answer combination from a random sentence
                random_sentence = get_random_sentence_from_context(original_answers, context)
                if random_sentence != "":
                    new_data.append(get_preference_data_item(context, question, answer, random_sentence))
                
                for each_new_answer in all_answers:
                    if new_answer != each_new_answer:
                        new_data.append(get_preference_data_item(context, question, answer, each_new_answer))

    print("number of no answer: ",total_no_anser)
    return new_data


def assign_no_answer_on_answers(data):
    new_data = []
    for paragraphs in data:
        for item in paragraphs:
            context = item["context"]
            question = item["question"]
            answer = item['answer']
            if answer != "no_answer":
                new_answer = "no_answer"
                new_data.append(get_preference_data_item(context, question, answer, new_answer))
                
    return new_data

def assign_other_answers(data):
    new_data = []
    for i_text, paragraphs in enumerate(data):
        original_answers = []
        for item in paragraphs:
            original_answers.append(item['answer'])
        for item in paragraphs:
            context = item["context"]
            question = item["question"]
            answer = item['answer']
            all_answers = []
            if answer != "no_answer":
                # Get different text combinations using the original answer text
                
                no_item = len(paragraphs)
                # Get text combinations using answer text of different questions
                for i in range(no_item):
                    new_answer = paragraphs[i]["answer"]
                    if new_answer != "no_answer" and new_answer != answer and new_answer not in all_answers:
                        all_answers.append(new_answer)
                        combinations_from_different_answer = create_negative_answers_combination(new_answer, context, 4, 7)
                        all_answers.extend(combinations_from_different_answer)
                
                # Get text combinations using a random sentence for answer
                if i_text == 1324:
                    print(i_text)
                random_sentence = get_random_sentence_from_context(original_answers, context)
                if random_sentence != "":
                    if random_sentence not in all_answers:
                        all_answers.append(random_sentence)
                    combinations_from_random_sentence = create_negative_answers_combination(random_sentence, context, 3, 2)
                    all_answers.extend(combinations_from_random_sentence)

            for each_new_answer in all_answers:
                if answer != each_new_answer:
                    new_data.append(get_preference_data_item(context, question, answer, each_new_answer))
    return new_data

if __name__ == "__main__":
    # 1. Multiple relations @GDA
    # filePath = "data/gda/train.tsv"
    qa_data = load_data("dataset/train.json")
    random.shuffle(qa_data)
    all_preference_data = []
    no_answer_data = assign_answers_on_no_answer(qa_data)
    assign_no_answer_data = assign_no_answer_on_answers(qa_data)
    other_answer_data = assign_other_answers(qa_data)
    random.shuffle(no_answer_data)
    random.shuffle(assign_no_answer_data)
    random.shuffle(other_answer_data)
    all_preference_data.extend(no_answer_data[:1000])
    all_preference_data.extend(assign_no_answer_data[:1000])
    all_preference_data.extend(other_answer_data[:8000])

    random.shuffle(all_preference_data)
    all_preference_data = all_preference_data[:4000]
    fields = ['prompt', 'chosen', 'rejected']
    filepath = "dataset/rule_based_preference_train.tsv"
    with open(filepath, 'w') as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=fields, delimiter='\t')
        csvwriter.writeheader()
        csvwriter.writerows(all_preference_data)
