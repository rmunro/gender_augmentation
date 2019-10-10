import sys
import re
import collections
from collections import defaultdict    
import bert_predictions

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertForMaskedLM
# import logging
# logging.basicConfig(level=logging.INFO)

# Load pre-trained model, tokenizer, and vocabulary
bt = BertTokenizer("bert-large-uncased-whole-word-masking-vocab.txt")
tokenizer = bt.from_pretrained('bert-large-uncased-whole-word-masking')
model = BertForMaskedLM.from_pretrained('bert-large-uncased-whole-word-masking')


pronouns = [
{"hers":"hers", "she":"she","Gender=Fem":"Gender=Fem"},
{"hers":"his", "she":"he","Gender=Fem":"Gender=Masc"},
{"hers":"theirs", "she":"they", "Gender=Fem":"Gender=Neut"},
{"hers":"mine", "she":"me","Gender=Fem":"Gender=Neut", "Person=3":"Person=1", "Alex's":"your"},
{"hers":"yours", "she":"you", "Gender=Fem":"Gender=Neut", "Person=3":"Person=2", "Alex's":"my"}
]

objects = [
    {"car":"cat",  # noun
    "drive":"sleep",  # irregular verb pres
    "drove":"slept",  # irregular verb past
    "responsive":"quiet",  # adj2
    "accelerated":"played",   # regular verb past
    "accelerate":"play",  # regular verb pres
    "hit":"chase", # regular verb trans
    "bump":"toy", # noun
    "paint":"food", # noun2
    "dealer":"vet", # noun agent
    "Dealer":"Vet", # noun agent (sentence initial)
    "clean":"brush",  # regular verb trans pres
    "easily":"regularly", # adv
    "easy":"regular", # adv root
    "quick":"obedient",  # adj
    "sell":"come", # irregular verb pres
    "sold":"came" # irregular verb past
    },
    {"car":"singer",  # noun
    "drive":"weep",  # irregular verb pres
    "drove":"wept",  # irregular verb past
    "responsive":"open",  # adj2
    "accelerated":"practiced",   # regular verb past
    "accelerate":"practive",  # regular verb pres
    "hit":"book", # regular verb trans pres
    "bump":"show", # noun
    "paint":"music", # noun2
    "dealer":"producer", # noun agent
    "Dealer":"Producer", # noun agent (sentence initial)
    "clean":"train",  # regular verb 
    "easily":"diligently", # adv
    "easy":"diligent", # adv root
    "quick":"loud",  # adj
    "sell":"sing", # irregular verb pres
    "sold":"sang" # irregular verb past
    }    
        
]


all_sentences = "" # all sentences, old and new

def replace_pronouns(lines):
    new_lines = ""
    for pronoun in pronouns:
        for line in lines:
            for old_pronoun in pronoun:
                new_pronoun = pronoun[old_pronoun]
                line = line.replace(old_pronoun, new_pronoun)
            new_lines += line
    return new_lines


def replace_objects(lines):
    new_lines = ""
    sets = objects
    for set in sets:
        for line in lines:
            for substitute in set:
                new = set[substitute]
                line = line.replace(substitute, new)
            new_lines += line
    return new_lines


       

sent_id = 0 # current sentence id
text = ""
previous = ""

current_lines = []

hers_counts = defaultdict(lambda: 0)
his_counts = defaultdict(lambda: 0)
diff_counts = defaultdict(lambda: 0)

for line in sys.stdin:
    # print(line)
    if line.startswith("# sent_id = "):
        if sent_id > 0: 
            all_sentences += "".join(current_lines)
            # all_sentences += replace_pronouns(current_lines)
            # all_sentences += replace_objects(current_lines)
            
            if "car " in text:
            
                for pronoun in pronouns:
                    new_pronoun = pronoun["hers"]
                    new_text = text.replace("hers", new_pronoun)
                    '''
                    print("TEXT: "+new_text)
                    variations = extract_bert_predictions(new_text, previous, "car", True)
                    '''

                    if new_pronoun == "his":
                        variations = bert_predictions.extract_bert_differences(text, previous, new_text, previous, "car", True)
                    

                        print("VARIATIONS: "+str(variations))
                        for key in variations:
                            diff_counts[key] += variations[key]
                    
                    '''
                    if new_pronoun == "hers":
                        for key in variations:
                            hers_counts[key] += variations[key]
                    if new_pronoun == "his":
                        for key in variations:
                            his_counts[key] += variations[key]
                    '''
                    
                    
            # print(new_lines)
        
        # get next sentence id
        sent_id = int(line.lstrip("# sent_id = ").strip())
        current_lines = [line]
        # print(sent_id)
    else:
        current_lines.append(line)
        if line.startswith("# previous = "):
            previous = line.lstrip("# previous = ").strip()
        if line.startswith("# text = "):
            text = line.lstrip("# text = ").strip()

all_sentences += "".join(current_lines)
# all_sentences += replace_pronouns(current_lines) # get the last sentence

# print(all_sentences)

lines = all_sentences.split("\n")
sent_id = 1 # current new sentence id

for i in range(0, len(lines)):
    line = lines[i]    
    if line.startswith("# sent_id = "):
        line = "# sent_id = "+str(sent_id)
        sent_id += 1
    # updated_sentences += line+"\n"    
    lines[i] = line


all_sentences = "\n".join(lines)

# print(all_sentences, end = "")
print(hers_counts)
print(his_counts)
for key in diff_counts:
	value = diff_counts[key]
	if value > 1:
		print(key+" "+value)	
	
	