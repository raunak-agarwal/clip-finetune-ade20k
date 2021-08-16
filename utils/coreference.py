'''
python coreference.py --indir INPUT_DIR_WITH_TXT_FILES --outdir EMPTY_DIR_TO_OUTPUT_TXT_FILES
'''
import json
import glob
from typing import List
from argparse import ArgumentParser

import spacy
from spacy.tokens import Doc, Span

from allennlp.predictors.predictor import Predictor

nlp = spacy.load("en_core_web_sm")

allen_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"

# allen_url = 'https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz'

# allen_url = "data/coref/"
# allen_url = allen_url + "coref-spanbert-large-2021.03.10.tar.gz"

predictor = Predictor.from_path(allen_url)


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

def get_cluster_head_idx(doc, cluster):
    noun_indices = IntersectionStrategy.get_span_noun_indices(doc, cluster)
    return noun_indices[0] if noun_indices else 0


def print_clusters(doc, clusters):
    def get_span_words(span, allen_document):
        return ' '.join(allen_document[span[0]:span[1]+1])

    allen_document, clusters = [t.text for t in doc], clusters
    for cluster in clusters:
        cluster_head_idx = get_cluster_head_idx(doc, cluster)
        if cluster_head_idx >= 0:
            cluster_head = cluster[cluster_head_idx]
            print(get_span_words(cluster_head, allen_document) + ' - ', end='')
            print('[', end='')
            for i, span in enumerate(cluster):
                print(get_span_words(span, allen_document) + ("; " if i+1 < len(cluster) else ""), end='')
            print(']')


def core_logic_part(document: Doc, coref: List[int], resolved: List[str], mention_span: Span):
    final_token = document[coref[1]]
    if final_token.tag_ in ["PRP$", "POS"]:
        resolved[coref[0]] = mention_span.text + "'s" + final_token.whitespace_
    else:
        resolved[coref[0]] = mention_span.text + final_token.whitespace_
    for i in range(coref[0] + 1, coref[1] + 1):
        resolved[i] = ""
    return resolved


def original_replace_corefs(document: Doc, clusters: List[List[List[int]]]) -> str:
    resolved = list(tok.text_with_ws for tok in document)

    for cluster in clusters:
        mention_start, mention_end = cluster[0][0], cluster[0][1] + 1
        mention_span = document[mention_start:mention_end]

        for coref in cluster[1:]:
            core_logic_part(document, coref, resolved, mention_span)

    return "".join(resolved)

def get_span_noun_indices(doc: Doc, cluster: List[List[int]]) -> List[int]:
    spans = [doc[span[0]:span[1]+1] for span in cluster]
    spans_pos = [[token.pos_ for token in span] for span in spans]
    span_noun_indices = [i for i, span_pos in enumerate(spans_pos)
        if any(pos in span_pos for pos in ['NOUN', 'PROPN'])]
    return span_noun_indices



def get_cluster_head(doc: Doc, cluster: List[List[int]], noun_indices: List[int]):
    head_idx = noun_indices[0]
    head_start, head_end = cluster[head_idx]
    head_span = doc[head_start:head_end+1]
    return head_span, [head_start, head_end]


def is_containing_other_spans(span: List[int], all_spans: List[List[int]]):
    return any([s[0] >= span[0] and s[1] <= span[1] and s != span for s in all_spans])


def improved_replace_corefs(document, clusters):
    resolved = list(tok.text_with_ws for tok in document)
    all_spans = [span for cluster in clusters for span in cluster]  # flattened list of all spans

    for cluster in clusters:
        noun_indices = get_span_noun_indices(document, cluster)

        if noun_indices:
            mention_span, mention = get_cluster_head(document, cluster, noun_indices)

            for coref in cluster:
                if coref != mention and not is_containing_other_spans(coref, all_spans):
                    core_logic_part(document, coref, resolved, mention_span)

    return "".join(resolved)

def print_comparison(resolved_original_text, resolved_improved_text):
    print(f"~~~ AllenNLP original replace_corefs ~~~\n{resolved_original_text}")
    print(f"\n~~~ Our improved replace_corefs ~~~\n{resolved_improved_text}")
        
def get_corefs(text):
    # text = "We want to take our code and create a game. Let's remind ourselves how to do that."
    clusters = predictor.predict(text)['clusters']
    doc = nlp(text)

#     print_comparison(original_replace_corefs(doc, clusters), improved_replace_corefs(doc, clusters))
    return improved_replace_corefs(doc, clusters)
    
def main():
    # Args
    parser = ArgumentParser()
    parser.add_argument('--indir', type=str)
    parser.add_argument('--outdir', type=str)
    args = parser.parse_args()
    
    imgs, captions = read_dir(args.indir)
    
    for img, caption in zip(imgs, captions):
        caption = get_corefs(caption)
        with open(args.outdir+img+".txt", 'w') as outfile:
            outfile.write(caption)
            print("Successfully writing: ", img)
            
            
if __name__ == '__main__':
    main()
    
# text = "In this image there are a few food items are arranged in the rack, behind it there is a person standing and holding some object in his hand, there are a few objects on the table, beside that there is like a refrigerator and there are a few bottles and other objects are arranged in the rack"
# get_corefs(text)


# text = "This picture describes about inside view of a hall. in this we can find group of people,\
# few are sitting on the chairs, in the background we can see lights."
# print("~~~ original ~~~\n", text)
# print()
# print(get_corefs(text))



