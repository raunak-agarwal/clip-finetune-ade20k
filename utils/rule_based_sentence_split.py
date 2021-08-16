"""
python rule_based_sentence_split.py --infile INPUT_JSONL_PATH --outdir OUTPUT_DIR_PATH
"""
import json
import logging
from argparse import ArgumentParser

import spacy
from spacy.lang.en import English

from nemo.collections.nlp.models import PunctuationCapitalizationModel
logging.getLogger('nemo_logger').setLevel(logging.ERROR)

basic_nlp = English()  # just the language with no pipeline
basic_nlp.add_pipe("sentencizer") # add simple punctuation-based split (Splits on .,!?)
nlp = spacy.load("en_core_web_lg") # bigger model for better POS tags

punctuation_model = PunctuationCapitalizationModel.from_pretrained("punctuation_en_bert")


def read_captions(fname):
    img_paths = []
    captions = []
    with open(fname, 'r') as json_file:
        json_list = list(json_file)
    for json_str in json_list:
        result = json.loads(json_str)
        img_path = result['image_id']
        caption = result['caption'] 
        img_paths.append(img_path)
        captions.append(caption)
    return (img_paths, captions)

def simple_split(doc):
    doc = basic_nlp(doc)
    return [sent.text for sent in doc.sents]

def add_punctuation(doc):
    doc = punctuation_model.add_punctuation_capitalization([doc])[0]
    return doc.replace(',,',',').replace('.,','.').replace(',.','.').replace('..','.')

def rulebook(prev_tag, next_tag, next_next_tag):
    prev_tag = prev_tag.replace('NNS', 'NN')
    next_tag = next_tag.replace('NNS', 'NN')
    
    duals = [('NN','IN'),('PRP','IN'), ('RB','IN')]
             
    triples = [('NN','CC','IN'), ('NN','CC','PRP'), ('NN','CC','EX'),\
                ('PRP','CC','EX'), ('PRP','CC','PRP'), ('PRP','CC','DT')]
    
    d = (prev_tag, next_tag)
    t = (prev_tag, next_tag, next_next_tag)
    
    return d in duals or t in triples
        

def rule_based_split_sent(doc):
    doc = nlp(doc)
    toks, tags = [], []
    output_toks = []
    
    for tok in doc:
        toks.append(tok.text)
        tags.append(tok.tag_)
    
    for i, tok in enumerate(toks):
        if tok == ',':
            tag = tags[i]
            
            if i > 1 and i < len(toks) - 1:
                prev_tag = tags[i-1]
                next_tag = tags[i+1]
                next_next_tag = tags[i+2]
            else:
                output_toks.append(tok)
                continue
            
            if rulebook(prev_tag, next_tag, next_next_tag):
                output_toks.append(".")
            else:
                output_toks.append(tok)
        else:
            output_toks.append(tok)
    return ' '.join(output_toks).replace(" .", ".").replace(" ,",",")

def rule_based_split_paragraph(doc):
    sents = add_punctuation(doc)
    sents = simple_split(sents)
    final_sents = []
    for sent in sents:
        final_sents.append(rule_based_split_sent(sent))
    return ' '.join(final_sents)

def main():
    # Args
    parser = ArgumentParser()
    parser.add_argument('--infile', type=str)
    parser.add_argument('--outdir', type=str)
    args = parser.parse_args()
    
    imgs, captions = read_captions(args.infile)
    
    for img, caption in zip(imgs, captions):
        caption = rule_based_split_paragraph(caption)
        with open(args.outdir+img+".txt", 'w') as outfile:
            outfile.write(caption)
            print("Successfully writing: ", img)
            

if __name__ == '__main__':
    main()