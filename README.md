This repository contains the PyTorch implementation of our paper **''RadQA-DPO: A Radiology Question Answering System with Encoder-Decoder Models Enhanced by Direct Preference Optimization''** (accepted at BioNLP 2025). It includes the code, the original RadQA dataset, and the preference dataset used in the experiments presented in the paper.

## System Requirement

* NVIDIA H100 GPUs
* Python >= 3.10.13
* torch, torchvision, torchaudio
* transformers
* datasets
* accelerate
* joblib = 1.3.2, numpy = 1.26.4, scikit-learn = 1.3.0, pandas = 2.2.1
sentencepiece = 0.2.0, nltk

## Training

### SFT Model
Run the train_sft.py file to train the Supervised Fine Tuned (SFT) model. 

Example train command:
```
python train_sft.py --epochs 8
```

### DPO Model
To train a model using DPO, run the train_dpo.py file. Please, provide the path of the SFT model in the argument that you want to optimize. Also provide the path of the training and validation preference data file in the argument. For example:

```
python train_dpo.py --model_name_or_path models/sft/ --train_file dataset/preference_dataset/t5-3b/train_preference_90.tsv
```

## Inference
To generate answer using the trained model, use the inference.py file. We can execute the inference in two ways:

### Run inference on o file:
Run the inference.py file and give the path of the model and the test file in the argument list.

For example:
```
python inference.py --model_path models/sft --input_file dataset/test.json
```

The above command run the inference code on the test.json file using the pretrained model and provide the accuracy and f1-score of the prediction.

### Run inference on a single input

We can run inference on a single input by importing the inference class and call the generate function. 

Example code:
```
from inference import LMInference

model_path = "models/sft"
input_text = "<context> IMPRESSION:  Right mid lung opacity is concerning for early pneumonia. <question> Can the patient's shortness of breath be explained by any infiltrations in the lungs?"
model = LMInference(model_path)
output = model.generate(input_text)[0]
```
