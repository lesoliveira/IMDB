#!/usr/bin/env python3

## Transformers to classify the IMDB dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm
from sklearn.metrics import classification_report
from datasets import load_dataset
from tqdm.auto import tqdm

## loads the dataset from the csv file
dataset = load_dataset ('csv', data_files = 'imdb_master-UTF.csv')

## selects the 25000 testing samples
testset = dataset['train'].filter(lambda x: x['type']=='test')

pipe = pipeline(model="textattack/bert-base-uncased-imdb",  device=0)

i =0 
y_pred =[]
y = []

for out in tqdm(pipe(KeyDataset(testset, "review"), batch_size=32,truncation="only_first")):
    if (testset['label'][i] == 'neg'):
        y.append(0)
    else:
        y.append(1)
	
	
    if (out['label'] == 'LABEL_0' ):
        y_pred.append(0)
    else:
        y_pred.append(1)
		
    i = i+1
    if (i%100==0):
        print(i)
		
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))


