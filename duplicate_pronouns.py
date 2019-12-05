import sys
import re
import collections
import os
from collections import defaultdict    

seed_data = "./UD_English-Pronouns/en_pronouns-ud-test.conllu"

# all replacements for the example sentences
pronouns = [
{"hers":"hers", "Hers":"Hers", "she":"she","Gender=Fem":"Gender=Fem"},
{"hers":"his", "Hers":"His", "she":"he","Gender=Fem":"Gender=Masc"},
{"hers":"theirs", "Hers":"Theirs", "she":"they", "Gender=Fem":"Gender=Neut"},
{"hers":"mine", "Hers":"Mine", "she":"me","Gender=Fem":"Gender=Neut", "Person=3":"Person=1", "Alex's":"your"},
{"hers":"yours", "Hers":"Yours", "she":"you", "Gender=Fem":"Gender=Neut", "Person=3":"Person=2", "Alex's":"my"}
]


all_sentences = "" # all sentences, old and new
current_lines = []

def replace_pronouns(lines):
    ''' Code to replace all pronouns
    
    '''
    new_lines = ""
    for pronoun in pronouns:
        for line in lines:
            for old_pronoun in pronoun:
                new_pronoun = pronoun[old_pronoun]
                line = line.replace(old_pronoun, new_pronoun)
            new_lines += line
    return new_lines


fp = open(seed_data, 'r')
prev = ""
line = fp.readline()
while line:
    # print(line)
    if line == "\n" and prev != "\n":         
        all_sentences += replace_pronouns(current_lines)
        current_lines = ["\n"]
    else:
        current_lines.append(line)

    prev = line
    line = fp.readline()
fp.close()


#update sentence numbers 

lines = all_sentences.split("\n")
sent_id = 1 # current new sentence id

for i in range(0, len(lines)):
    line = lines[i]    
    if line.startswith("# sent_id = "):
        line = "# sent_id = "+str(sent_id)
        sent_id += 1
    elif line.startswith("# newdoc") and i > 0:
    	line = ""
    # updated_sentences += line+"\n"    
    lines[i] = line

all_sentences = "\n".join(lines)

print(all_sentences)
