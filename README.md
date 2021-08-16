# clip-finetune-ade20k
Caption Generation and VQA on ADE20K using CLIP


### Preprocessing Captions from Localized Narratives
To convert the paragraphs from Localized Narratives into meaningful sentences, we apply the following preprocessing scripts in a pipeline:

1. ```utils/rule_based_sentence_split.py```: POS Tag-based split. First applies a BERT-based punctuation model to add missing punctuations, then applies POS rules to replace misplaced commas with fullstops. 
2. ```utils/coreference.py```: Coreference Resolution to add missing context from pronouns.  
3. ```utils/ml_based_sentence_split.py```: BERT model trained on the wikisplit dataset that takes one long sentence and splits it into two shorter sentences. 


### CLIP Finetuning



### CLIP Caption Generation


### Synthetic Question Generation for VQA using T5


### Evaluation
#### Image Captioning
#### VQA