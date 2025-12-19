## Feature Extraction Using BERT
## It generates a feature vector of 

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertModel
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased', quantization_config=bnb_config)

data = pd.read_csv('imdb_master-UTF.csv' )

data['label'] = data['label'].replace(['neg'],'0')
data['label'] = data['label'].replace(['pos'],'1')

dfdrop = data.iloc[:-50000]

texts = dfdrop['review']
labels = dfdrop['label']

batch_size = 256  # Set an appropriate batch size based on memory capacity

# Open file in write mode to save embeddings
with open('embeddings.txt', 'w') as f:
    for i in range(0, len(texts), batch_size):
        # Batch processing
        batch_texts = texts[i:i+batch_size].tolist()

        # Tokenize the batch
        encoded_texts = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')

        # Move tensors to GPU if available
        if torch.cuda.is_available():
            encoded_texts = {key: value.cuda() for key, value in encoded_texts.items()}
            #bert_model = bert_model.cuda()

        # Get BERT embeddings
        with torch.no_grad():
            outputs = bert_model(**encoded_texts)
            embeddings = outputs.last_hidden_state

        # Pooling strategy: Using [CLS] token
        pooled_embeddings = embeddings[:, 0, :].cpu().numpy()  # Move to CPU to prevent memory overload on GPU

        print (i)

        # Append embeddings to the file in ASCII format
        np.savetxt(f, pooled_embeddings, fmt='%.6f', delimiter=' ')


