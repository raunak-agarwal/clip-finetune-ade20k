"""
python ml_based_sentence_split.py --indir INPUT_DIR_WITH_TXT_FILES --outdir EMPTY_DIR_TO_OUTPUT_TXT_FILES --read_range ADE20K_RANGE (eg. 1-2000)
"""
import json
import glob
from argparse import ArgumentParser

from anytree import Node, RenderTree

import torch
import spacy

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

nlp = spacy.load("en_core_web_lg")
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
  
tokenizer = AutoTokenizer.from_pretrained("flax-community/t5-v1_1-base-wikisplit")

model = AutoModelForSeq2SeqLM.from_pretrained("flax-community/t5-v1_1-base-wikisplit")

model = model.to(device)

def read_file(fname):
    with open(fname, 'r') as f:
        text = f.read()
    return text

def read_dir(dirname, readrange):
    fnames = glob.glob(dirname+"*")
    names, captions = [], []
    readrange = readrange.split("-")
    begin_range = int(readrange[0])
    end_range = int(readrange[1])
    for fname in fnames:
        name = fname.split("/")[-1].split(".")[0]
        n = int(name.split("_")[-1])
        
        if begin_range <= n <= end_range:
            caption = read_file(fname)
            names.append(name)
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
    cond = 'CC' in tags #or 'EX' in tags or 'IN' in tags
    return len(text) > 10 and cond

def check_doc(d):
    toks = d.lower().split(" ")
    total_toks = len(toks)
    done=False
    for i, tok in enumerate(toks):
        if tok in ["in","on","and","of","as","is","are","to","with"]:
            try:
                cond1 = toks[i-1] == toks[i+1]
                if i > 2 and i < total_toks - 2:
                    cond2 = toks[i-2] + toks[i-1] == toks[i+1] + toks[i+2]
                if i > 3 and i < total_toks - 3:
                    cond3 = toks[i-3] + toks[i-2] + toks[i-1] == \
                    toks[i+1] + toks[i+2] + toks[i+3]
                if i > 4 and i < total_toks - 4:
                    cond4 = toks[i-4] + toks[i-3] + toks[i-2] + toks[i-1] == \
                    toks[i+1] + toks[i+2] + toks[i+3] + toks[i+4]
                if i > 5 and i < total_toks - 5:
                    cond5 = toks[i-5] + toks[i-4] + toks[i-3] + toks[i-2] + toks[i-1] \
                    == toks[i+1] + toks[i+2] + toks[i+3] + toks[i+4] + toks[i+5]
                if i > 6 and i < total_toks - 6:
                    cond6 = toks[i-6] + toks[i-5] + toks[i-4] + toks[i-3] + \
                    toks[i-2] + toks[i-1] == toks[i+1] + toks[i+2] + \
                    toks[i+3] + toks[i+4] + toks[i+5] + toks[i+6]
                if i > 7 and i < total_toks - 7:
                    cond7 = toks[i-7] + toks[i-6] + toks[i-5] + toks[i-4] + toks[i-3] + \
                    toks[i-2] + toks[i-1] == toks[i+1] + toks[i+2] + \
                    toks[i+3] + toks[i+4] + toks[i+5] + toks[i+6] + toks[i+7]
                if i > 8 and i < total_toks - 8:
                    cond8 = toks[i-8] + toks[i-7] + toks[i-6] + toks[i-5] +\
                    toks[i-4] + toks[i-3] + \
                    toks[i-2] + toks[i-1] == toks[i+1] + toks[i+2] + \
                    toks[i+3] + toks[i+4] + toks[i+5] + toks[i+6] + toks[i+7] + toks[i+8]
                if cond1 or cond2 or cond3 or cond4 or cond5 or cond6 or cond7 or cond8:
                    done=True
                    d = " ".join(toks[:i])
            except Exception as e:
                pass
#                 d = " ".join(toks[:i])
#                     done=True
#                     print(e, " ", i, d)
        if done:
            break

    d0 = d.split(" and ")
    if len(d0) > 1:
        d1 = d0[0].strip().lower()
        d2 = d0[1].strip().lower()
        if d1.endswith(d2):
            d=d1
    return d

def split_sent(text, p = False):
    root = Node(name = text, split = call_condition(text))
    split = root.split
    
    if not split:
        return [root.name]
    
    splits = t5_split(root.name)
    
#     print("\nSplits 1: ", splits)
    

    try:
        cond1 = call_condition(splits[0])
        cond2 = call_condition(splits[1])  
        l1 = Node(name = splits[0], split = cond1, parent = root)
        l2 = Node(name = splits[1], split = cond2, parent = root)
        
        if l1.split:
            splits = t5_split(l1.name)
    #         print("\nSplit l1: ", splits)
            cond1 = call_condition(splits[0])
            cond2 = call_condition(splits[1])
            l11 = Node(name = splits[0], split = cond1, parent = l1)
            l12 = Node(name = splits[1], split = cond2, parent = l1)   

        if l2.split:
            splits = t5_split(l2.name)
    #         print("\nSplit l2: ", splits)
            cond1 = call_condition(splits[0])
            cond2 = call_condition(splits[1])
            l21 = Node(name = splits[0], split = cond1, parent = l2)
            l22 = Node(name = splits[1], split = cond2, parent = l2)
    
    except Exception as e:
        return [root.name]

    try:
        if l11.split:
            splits = t5_split(l11.name)
#             print("\nSplit l11: ", splits)
            cond1 = call_condition(splits[0])
            cond2 = call_condition(splits[1])
            l111 = Node(name = splits[0], split = cond1, parent = l11)
            l112 = Node(name = splits[1], split = cond2, parent = l11)
    except Exception as e:
        pass
    
    try:
        if l12.split:
            splits = t5_split(l12.name)
#             print("\nSplit l12: ", splits)
            cond1 = call_condition(splits[0])
            cond2 = call_condition(splits[1])
            l121 = Node(name = splits[0], split = cond1, parent = l12)
            l122 = Node(name = splits[1], split = cond2, parent = l12)
    except Exception as e:
        pass
    
    try:
        if l21.split:
            splits = t5_split(l21.name)
#             print("\nSplit l21: ", splits)
            cond1 = call_condition(splits[0])
            cond2 = call_condition(splits[1])
            l121 = Node(name = splits[0], split = cond1, parent = l21)
            l122 = Node(name = splits[1], split = cond2, parent = l21)
    except Exception as e:
        pass
    
    try:
        if l22.split:
            splits = t5_split(l22.name)
#             print("\nSplit l22: ", splits)
            cond1 = call_condition(splits[0])
            cond2 = call_condition(splits[1])
            l121 = Node(name = splits[0], split = cond1, parent = l22)
            l122 = Node(name = splits[1], split = cond2, parent = l22) 
    except Exception as e:
        pass
    
    if p:
        print("\n\n")
        for pre, fill, node in RenderTree(root):
            print("%s%s" % (pre, node.name))
    
    return [i.name for i in root.leaves]

def split_paragraph(text):
    text = nlp(text)
    sents = []
    for sent in text.sents:
        splits = split_sent(sent.text)
        for s in splits:
            sents.append(check_doc(s))
        
    return sents

def final_clean(text):
    sents = split_paragraph(text)
    final_sents = []
    for s in sents:
        s = s.strip()
        s = s.replace("This", "this")
        if s.lower().startswith("and"):
            s = s[3:]
        s = s.strip(",").strip()
        s = s[:1].upper() + s[1:]
        if len(s.split()) > 5:
            final_sents.append(s)
    return final_sents


def main():
    # Args
    parser = ArgumentParser()
    parser.add_argument('--indir', type=str)
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--read_range', type=str)
    args = parser.parse_args()
    
    imgs, captions = read_dir(args.indir, args.read_range)
    
    for img, caption in zip(imgs, captions):
        caption = final_clean(caption)
        caption = list(set(caption))
        with open(args.outdir+img+".txt", 'w') as outfile:
            for line in caption:
                outfile.write('%s\n' % line)
            print("Successfully writing: ", img)
            
            
if __name__ == '__main__':
    main()