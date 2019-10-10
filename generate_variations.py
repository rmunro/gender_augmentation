import sys
import re
import collections
from collections import defaultdict    

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


def extract_bert_predictions(text, previous, tomask, monte_carlo=True, max_samples=100, good_turing_threshold=0.05):
    """Get predictions for new words in the given contexts
        
    Keyword arguments:
        text -- the sentence with the word to mask
        previous -- the previous sentence
        tomask -- the word to mask
        monte_carlo -- whether to use Monte-Carlo Dropouts to get multiple predictions
        max_samples -- hard limit on the number of samples from Monte Carlo Dropouts
        good_turing_threshold -- threshold for when to stop looking for new words, as a percent likelihood that a novel prediction will be discovered      
    """ 
    
    word_counts = defaultdict(lambda: 0)
    seen_total = 0
    seen_once = 0 # number seen only once
    good_turing = 1 # current Good-Turing estimate

    # Tokenize the two sentences for BERT
    input = "[CLS] "+previous+" [SEP] "+text+" [SEP]"
    tokenized_text = tokenizer.tokenize(input)
    
    # MASK the variable
    mask_index = tokenized_text.index(tomask)
    tokenized_text[mask_index] = '[MASK]'

    # Convert the tokenized sentences to tensors for pytorch  
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    sep_index = tokenized_text.index("[SEP]")
    segments_ids = [1] * len(tokenized_text)
    for i in range(0, sep_index+1):
        segments_ids[i] = 0
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
 
     # Get predictions for masked word
    with torch.no_grad():
        if monte_carlo:
            model.train() # include dropouts 
            while good_turing >= good_turing_threshold and seen_total < max_samples:
                outputs = model(tokens_tensor, token_type_ids=segments_tensors) 
                predictions = outputs[0]
                predicted_index = torch.argmax(predictions[0, mask_index]).item()
                predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
                # print("Predicted token:")
                # print(predicted_token)
                
                # get Smoothed Good-Turing estimate to decide when to keep searching
                seen_total += 1
                word_counts[predicted_token] += 1
                if word_counts[predicted_token] == 1:
                    seen_once += 1
                elif word_counts[predicted_token] == 2:
                    seen_once -= 1
                    
                good_turing = (seen_once + len(word_counts)) / seen_total
                # print("Predicted score:")
                # print(predictions[0, mask_index])
        else:
            model.eval()
            outputs = model(tokens_tensor, token_type_ids=segments_tensors) 
            predictions = outputs[0]
            predicted_index = torch.argmax(predictions[0, mask_index]).item()
            predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
            # print("Predicted token:")
            # print(predicted_token)
            word_counts[predicted_token] += 1          
            # print("Predicted score:")
            # print(predictions[0, mask_index])
            
    return word_counts
       
       
def extract_bert_differences(text1, previous1, text2, previous2, tomask, monte_carlo=True, max_samples=100, good_turing_threshold=0.05):
    """Get the words with the greatest difference in prediction score 
        
    Keyword arguments:
        text1 -- the first sentence with the word to mask
        previous1 -- the previous sentence to the first one
        text2 -- the second sentence with the word to mask
        previous2 -- the previous sentence to the second one
        tomask -- the word to mask
        monte_carlo -- whether to use Monte-Carlo Dropouts to get multiple predictions
        max_samples -- hard limit on the number of samples from Monte Carlo Dropouts
        good_turing_threshold -- threshold for when to stop looking for new words, as a percent likelihood that a novel prediction will be discovered      
    """ 
    
    word_counts = defaultdict(lambda: 0)
    seen_total = 0
    seen_once = 0 # number seen only once
    good_turing = 1 # current Good-Turing estimate

    # Tokenize the first pair of sentences for BERT
    input1 = "[CLS] "+previous1+" [SEP] "+text1+" [SEP]"
    tokenized_text1 = tokenizer.tokenize(input1)
    
    # MASK the variable
    mask_index1 = tokenized_text1.index(tomask)
    tokenized_text1[mask_index1] = '[MASK]'

    # Convert the tokenized sentences to tensors for pytorch  
    indexed_tokens1 = tokenizer.convert_tokens_to_ids(tokenized_text1)
    sep_index1 = tokenized_text1.index("[SEP]")
    segments_ids1 = [1] * len(tokenized_text1)
    for i in range(0, sep_index1+1):
        segments_ids1[i] = 0
        
    # Tokenize the second pair of sentences for BERT
    input2 = "[CLS] "+previous2+" [SEP] "+text2+" [SEP]"
    tokenized_text2 = tokenizer.tokenize(input2)
    
    # MASK the variable
    mask_index2 = tokenized_text2 .index(tomask)
    tokenized_text2[mask_index2] = '[MASK]'

    # Convert the tokenized sentences to tensors for pytorch  
    indexed_tokens2 = tokenizer.convert_tokens_to_ids(tokenized_text2)
    sep_index2 = tokenized_text2.index("[SEP]")
    segments_ids2  = [1] * len(tokenized_text2)
    for i in range(0, sep_index2+1):
        segments_ids2[i] = 0
        


    tokens_tensor1 = torch.tensor([indexed_tokens1])
    segments_tensors1 = torch.tensor([segments_ids1])
 
    tokens_tensor2 = torch.tensor([indexed_tokens2])
    segments_tensors2 = torch.tensor([segments_ids2])
 
     # Get predictions for masked word
    with torch.no_grad():
        if monte_carlo:
            model.train() # include dropouts 
            while good_turing >= good_turing_threshold and seen_total < max_samples:
                outputs1 = model(tokens_tensor1, token_type_ids=segments_tensors1) 
                predictions1 = outputs1[0]
                outputs2 = model(tokens_tensor2, token_type_ids=segments_tensors2) 
                predictions2 = outputs2[0]
            
                max_diff1 = 0.0
                max_token1 = ""
                max_diff2 = 0.0
                max_token2 = ""
            
                pred_count = len(predictions1[0, mask_index1])            
            
                for i in range(0, pred_count):
                    pred1 = predictions1[0, mask_index1][i].item()
                    pred2 = predictions2[0, mask_index2][i].item()
                    token = tokenizer.convert_ids_to_tokens([i])[0]
                
                    diff1 = pred1 - pred2
                    if diff1 > max_diff1 and not token.startswith('['):
                        max_diff1 = diff1
                        max_token1 = token

                    diff2 = pred2 - pred1
                    if diff2 > max_diff2 and not token.startswith('['):
                        max_diff2 = diff2
                        max_token2 = token

                word_counts[max_token1+" 1"] += 1
                word_counts[max_token2+" 2"] += 1
                
                # get Smoothed Good-Turing estimate to decide when to keep searching
                seen_total += 1
                word_counts[max_token1] += 1
                if word_counts[max_token1] == 1:
                    seen_once += 1
                elif word_counts[max_token1] == 2:
                    seen_once -= 1
                word_counts[max_token2] += 1
                if word_counts[max_token2] == 1:
                    seen_once += 1
                elif word_counts[max_token2] == 2:
                    seen_once -= 1
                    
                good_turing = (seen_once + len(word_counts)) / seen_total
                # print("Predicted score:")
                # print(predictions[0, mask_index])
        else:
            model.eval()
            outputs1 = model(tokens_tensor1, token_type_ids=segments_tensors1) 
            predictions1 = outputs1[0]
            outputs2 = model(tokens_tensor2, token_type_ids=segments_tensors2) 
            predictions2 = outputs2[0]
            
            max_diff1 = 0.0
            max_token1 = ""
            max_diff2 = 0.0
            max_token2 = ""
            
            pred_count = len(predictions1[0, mask_index1])            
            
            for i in range(0, pred_count):
                pred1 = predictions1[0, mask_index1][i].item()
                pred2 = predictions2[0, mask_index2][i].item()
                token = tokenizer.convert_ids_to_tokens([i])[0]
                
                diff1 = pred1 - pred2
                if diff1 > max_diff1 and not token.startswith('['):
                    max_diff1 = diff1
                    max_token1 = token

                diff2 = pred2 - pred1
                if diff2 > max_diff2 and not token.startswith('['):
                    max_diff2 = diff2
                    max_token2 = token

            word_counts[max_token1+" 1"] += 1
            word_counts[max_token2+" 2"] += 1
            
    return word_counts
           
    
       

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
                        variations = extract_bert_differences(text, previous, new_text, previous, "car", True)
                    

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
print(diff_counts)