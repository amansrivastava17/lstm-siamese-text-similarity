##Text Similarity Using Siamese Deep Neural Network

Siamese neural network is a class of **neural network architectures that contain two or more** **identical** **subnetworks**. *identical* here means they have the same configuration with the same parameters 
and weights. Parameter updating is mirrored across both subnetworks.

It is a keras based implementation of deep siamese Bodirectional LSTM network to capture phrase/sentence similarity using word embeddings.

Below is the architecture description for the same.

![rch_imag](images/arch_image.png)



### Usage



#### Training

```python
from siamese_model import train_seimese_model
from inputHandler import word_embed_meta_data, create_test_data
from config import siamese_config
import pandas as pd

########################################
############ Data Preperation ##########
########################################


df = pd.read_csv('sample_data.csv')

sentences1 = list(df['sentences1'])
sentences2 = list(df['sentences2'])
is_similar = list(df['is_similar'])
del df

####################################
######## Word Embedding ############
####################################

# creating word embedding meta data for word embedding 
tokenizer, embedding_matrix = word_embed_meta_data(sentences1 + sentences2,  siamese_config['EMBEDDING_DIM'])

embedding_meta_data = {
	'tokenizer': tokenizer,
	'embedding_matrix': embedding_matrix
}

## creating sentence pairs
sentences_pair = [(x1, x2) for x1, x2 in zip(sentences1, sentences2)]
del sentences1
del sentences2

##########################
######## Training ########
##########################

best_model_path = train_seimese_model(sentences_pair, is_similar, embedding_meta_data, model_save_directory='./')
```

```python
from operator import itemgetter
from keras.models import load_model

model = load_model(best_model_path)

test_sentence_pairs = [('What can make Physics easy to learn?','How can you make physics easy to learn?'),
					   ('How many times a day do a clocks hands overlap?','What does it mean that every time I look at the clock the numbers are the same?')]

test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer,test_sentence_pairs,  siamese_config['MAX_SEQUENCE_LENGTH'])

preds = list(model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1).ravel())
results = [(x, y, z) for (x, y), z in zip(test_sentence_pairs, preds)]
results.sort(key=itemgetter(2), reverse=True)
print results
```

### References:

1. [Siamese Recurrent Architectures for Learning Sentence Similarity (2016)](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12195)

2. Inspired from https://github.com/dhwajraj/deep-siamese-text-similarity

   ​