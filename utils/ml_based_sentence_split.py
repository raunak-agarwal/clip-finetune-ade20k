"""
python ml_based_sentence_split.py --indir INPUT_DIR_WITH_TXT_FILES --outdir EMPTY_DIR_TO_OUTPUT_TXT_FILES
"""
import json
import glob
from argparse import ArgumentParser

import torch
import spacy

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

nlp = spacy.load("en_core_web_lg")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
  
tokenizer = AutoTokenizer.from_pretrained("flax-community/t5-v1_1-base-wikisplit")

model = AutoModelForSeq2SeqLM.from_pretrained("flax-community/t5-v1_1-base-wikisplit")

model = model.to(device)

def read_file(fname):
    with open(fname, 'r') as f:
        text = f.read()
    return text

def read_dir(dirname):
    fnames = glob.glob(dirname+"*")
    names, captions = [], []
    for fname in fnames:
        n = fname.split("/")[-1].split(".")[0]
        caption = read_file(fname)
        
        names.append(n)
        captions.append(caption)
        
    return (names, captions)

def t5_split(doc):
    docs = []
    
    tokenized = tokenizer(doc, return_tensors="pt").to(device)
    answer = model.generate(tokenized['input_ids'], 
                            attention_mask = tokenized['attention_mask'], 
                            max_length=256, num_beams=20)
    
    generated = tokenizer.decode(answer[0], skip_special_tokens=True)
#     print(doc)
    return [i.strip() for i in generated.split(".") if len(i) > 0]

def call_condition(x):
    nlp_doc = nlp(x)
    tags = [i.tag_ for i in nlp_doc]
    text = [i.text for i in nlp_doc]
    cond = 'CC' or 'IN' or 'EX' in tags
    return len(x) > 10 and cond

def final_clean(doc):
    #Condition if doc starts with 'and'
    #Condition if doc has a capitalized This in the middle of a sentence
    #Conditon to capitalize first letter
    #Strip extra whitespace
    

def main():
    # Args
    parser = ArgumentParser()
    parser.add_argument('--indir', type=str)
    parser.add_argument('--outdir', type=str)
    args = parser.parse_args()
    
    imgs, captions = read_dir(args.indir)
    
    for img, caption in zip(imgs, captions):
        caption = get_splits(caption)
        with open(args.outdir+img+".txt", 'w') as outfile:
            outfile.writelines(caption)
            print("Successfully writing: ", img)
            
            
if __name__ == '__main__':
    main()