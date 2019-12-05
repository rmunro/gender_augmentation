#!/usr/bin/env python

"""BERT PREDICTIONS

A collection of functions to probe BERT for bias

"""

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


def extract_bert_predictions(text, previous, tomask, also_mask=[], monte_carlo=True, max_samples=100, good_turing_threshold=0.05):
    """Get predictions for new words in the given contexts
        
    Keyword arguments:
        text -- the sentence with the word to mask
        previous -- the previous sentence
        tomask -- the word to mask and predict against
        also_mask -- words to also mask
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
    if tomask not in tokenized_text:
        # we got lost in bert tokenization: skip it
        print("WARNING: could not find "+tomask+" in "+str(tokenized_text))        
        return
    
    mask_index = 0
    for i in range(0, len(tokenized_text)): 
        if tokenized_text[i] == tomask:
            tokenized_text[i] = '[MASK]'
            mask_index = i

    for i in range(0, len(tokenized_text)): 
        if tokenized_text[i] in also_mask:
            tokenized_text[i] = '[MASK]'

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

def get_hers_his_theirs_difference(text, previous):
    """ Get the difference in prediction for "hers" and "his" in a sentence with "hers"
    
    """
    
    input = "[CLS] "+previous+" [SEP] "+text+" [SEP]"
    tokenized_text = tokenizer.tokenize(input)
    
    # MASK the variable
    mask_index = tokenized_text.index("hers")
    tokenized_text[mask_index] = '[MASK]'
        
    # Convert the tokenized sentences to tensors for pytorch  
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    sep_index = tokenized_text.index("[SEP]")
    segments_ids = [1] * len(tokenized_text)
    for i in range(0, sep_index+1):
        segments_ids[i] = 0
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
 
    hers_ind, his_ind, theirs_ind = tokenizer.convert_tokens_to_ids(["hers", "his", "theirs"])
      
     # Get predictions for masked word
    with torch.no_grad():
        model.eval()
        outputs = model(tokens_tensor, token_type_ids=segments_tensors) 
        predictions = outputs[0]
        
        pron = predictions[0, mask_index]
        
        prob_dist = torch.nn.functional.softmax(predictions[0, mask_index],dim=0)
        hers_pred = prob_dist[hers_ind].item()
        his_pred = prob_dist[his_ind].item()
        theirs_pred = prob_dist[theirs_ind].item()
        
        diff = hers_pred - his_pred
        if diff < 0:
            diff = 0-diff
            
        return [diff, hers_pred, his_pred, theirs_pred]       
               
       
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
                if word_counts[max_token1+" 1"] == 1:
                    seen_once += 1
                elif word_counts[max_token1+" 1"] == 2:
                    seen_once -= 1
                if word_counts[max_token2+" 2"] == 1:
                    seen_once += 1
                elif word_counts[max_token2+" 2"] == 2:
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

       