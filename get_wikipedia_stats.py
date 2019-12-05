import json
import re
import sys
from glob import glob
import os
import gzip


def load_to_array(file_path):
    with open(file_path) as my_file:
        arr = my_file.readlines()
        arr = map(lambda s: s.strip(), arr)
        filter(None, arr)
        
        return list(arr)


words_to_test = load_to_array("./words_to_test.txt")

fem_words = load_to_array("./fem_words.txt")
masc_words = load_to_array("./masc_words.txt")

def convert_into_sentences(lines):
    new_lines = []
    for line in lines:
        data = json.loads(line)
        if "source_text" in data:
            source = data["source_text"]
            source = re.sub('([\.?!]+ )',r'\1   ', source)
            sents = source.split("  ")
            for sent in sents:
                sent = " "+sent+" "
                sent = re.sub('([^A-Za-z0-9]+) ', r' \1 ', sent)
                sent = re.sub(' ([^A-Za-z0-9]+)', r' \1 ', sent)
                sent = re.sub('(A-Za-z)\'s ', r' \1 \'s', sent)
                sent = re.sub('\s+', ' ', sent)
                if sent.strip() != "":
                    new_lines.append(sent)
    return new_lines
                
# file_list = list(sorted(glob(os.path.join(file_dir, '*.json'))))

total_fem = 0
total_masc = 0

fem_counts = {} # total count per fem sentence
masc_counts = {}
fem_poss_counts = {} # total count per fem possessive construction
masc_poss_counts = {}
hers_near_counts = {}
his_near_counts = {}

for word in words_to_test:
    fem_counts[word] = 0.1
    masc_counts[word] = 0.1
    fem_poss_counts[word] = 0.1
    masc_poss_counts[word] = 0.1
    hers_near_counts[word] = 0.1
    his_near_counts[word] = 0.1

c=0
s=0
with gzip.open('enwiki-20191125-cirrussearch-content.json.gz','rt') as f:
    for line in f:
        # print('got line', line)
        sents = convert_into_sentences([line])

        for sent in sents:
            fem = False
            masc = False
            for fem_word in fem_words:
                if " "+fem_word+" " in sent:
                    fem = True
                    total_fem += 1
                    break
            for masc_word in masc_words:
                if " "+masc_word+" " in sent:
                    masc = True
                    total_masc += 1
                    break

            for word in words_to_test:
                if " "+word+" " in sent:
                    if fem:
                        fem_counts[word] += 1
                    if masc:
                        masc_counts[word] += 1
                if " her "+word in sent or " Her "+word in sent:
                    fem_poss_counts[word] += 1
                if " his "+word	in sent	or " His "+word	in sent:
                    masc_poss_counts[word] += 1

                if " hers " in sent or " Hers " in sent:
                    if re.search(" "+word+" [^ ]* ?hers ",sent) or re.search(" [Hh]ers [^ ]* ?"+word,sent):
                        hers_near_counts[word] +=1
                if " his " in sent or " His " in sent:
                    if re.search(" "+word+" [^ ]* ?his ",sent)	or re.search(" [Hh]is [^ ]* ?"+word,sent):
                        his_near_counts[word] +=1

                    
                if " her "+word in sent or " Her "+word in sent:
                    fem_poss_counts[word] += 1
                if " his "+word in sent or " His "+word in sent:
                    masc_poss_counts[word] += 1

        
        s += len(sents)
        if c > 0 and c % 10 == 0:
            print(c,s)
            for word in words_to_test:
                bias_sent = str(masc_counts[word] / (fem_counts[word] + masc_counts[word]))
                bias_poss = str(masc_poss_counts[word] / (fem_poss_counts[word] +masc_counts[word] ))
                bias_near = str(his_near_counts[word] / (his_near_counts[word] + hers_near_counts[word] ))

                sent_count = str(fem_counts[word] + masc_counts[word])
                poss_count = str(fem_poss_counts[word] + masc_poss_counts[word])
                near_count = str(his_near_counts[word] + hers_near_counts[word])
                print("\t".join([word, bias_sent, bias_poss, bias_near, sent_count, poss_count, near_count]))

        c+=1


for word in words_to_test:
    bias_sent = str(masc_counts[word] / (fem_counts[word] + masc_counts[word]))
    bias_poss = str(masc_poss_counts[word] / (fem_poss_counts[word] + masc_counts[word] ))
    bias_near = str(his_near_counts[word] / (his_near_counts[word] + hers_near_counts[word] ))

    sent_count = str(fem_counts[word] + masc_counts[word])
    poss_count = str(fem_poss_counts[word] + masc_poss_counts[word])
    near_count = str(his_near_counts[word] + hers_near_counts[word])
    print("\t".join([word, bias_sent, bias_poss, bias_near, sent_count, poss_count, near_count]))
    
print(total_fem)
print(total_masc)
print(s)

        
'''
for i, file_path in enumerate(file_list):
    sents = convert_into_sentences(open(file_path).readlines())
    print('\n'.join(sents))
    print('\n\n\n\n')
'''

    

