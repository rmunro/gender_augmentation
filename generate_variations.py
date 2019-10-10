import sys
import re
import collections
from collections import defaultdict    
import bert_predictions


seed_data = "new_sentences.conllu"
reference_data = "Universal Dependencies 2.4/ud-treebanks-v2.4/UD_English-LinES/en_lines-ud-train.conllu"

# all replacements for the example sentences
pronouns = [
{"hers":"hers", "she":"she","Gender=Fem":"Gender=Fem"},
{"hers":"his", "she":"he","Gender=Fem":"Gender=Masc"},
{"hers":"theirs", "she":"they", "Gender=Fem":"Gender=Neut"},
{"hers":"mine", "she":"me","Gender=Fem":"Gender=Neut", "Person=3":"Person=1", "Alex's":"your"},
{"hers":"yours", "she":"you", "Gender=Fem":"Gender=Neut", "Person=3":"Person=2", "Alex's":"my"}
]

all_sentences = "" # all sentences, old and new

sent_id = 0 # current sentence id
text = ""
previous = ""

current_lines = []

hers_counts = defaultdict(lambda: 0)
his_counts = defaultdict(lambda: 0)
diff_counts = defaultdict(lambda: 0)


def get_parent(word, lines):
    '''Gets the token (string) of the parent of the word in lines
    assumes that lines is one entry in dependency file, as an array of each line
    '''
    parent = 0 
    
    for line in lines:
        fields = line.split("\t")
        if len(fields) > 5 and fields[1].lower() == word.lower():            
            parent = fields[6]
        
    if parent == "0":
        return word # field is already the root of the tree
    
    for line in lines:
        fields = line.split("\t")
        if len(fields) > 5 and fields[0] == parent:
            return fields[1]
            
    
    


def get_dependents(word, lines, word_number="-1"):
    '''Gets the token (string) of word in lines
    assumes each line is the line from one entry in dependency file
    '''
    dependents = []
    
    if word_number == "-1":        
        # work out the words position in the sentence if it wasnt passed
        for line in lines:
            fields = line.split("\t")
            if len(fields) > 5 and fields[1].lower() == word.lower():
                word_number = fields[0]
                break
            
    for line in lines:
        fields = line.split("\t")
        if len(fields) > 5 and fields[6] == word_number:
            new_word = fields[1]
            dependents.append(new_word)
            dependents.extend(get_dependents(new_word, lines, fields[0]))
    
    return dependents


reference_dependencies = defaultdict(lambda: defaultdict(lambda: 0))

fp = open(reference_data, 'r')
line = fp.readline()
c=1
while line:
    fields = line.split("\t")
    if len(fields) > 5 :
        word = fields[1]
        constants = fields[3]+"\t"+fields[4]+"\t"+fields[5]
        reference_dependencies[word][constants]+=1
    c+=1
    line = fp.readline()
fp.close()


seen_sentences = defaultdict(lambda: 0)

fp = open(seed_data, 'r')
line = fp.readline()
while line:
    # print(line)
    if line.startswith("# sent_id = "):
        if sent_id > 0: 
            all_sentences += "".join(current_lines)
            # all_sentences += replace_pronouns(current_lines)
            # all_sentences += replace_objects(current_lines)
            
            parent = get_parent("hers", current_lines)     
            relations = get_dependents(parent, current_lines)
            relations.append(parent)
            print("relations: "+str(relations)+" in "+text)

            # new varia
            sentence_variations = defaultdict(lambda: defaultdict(lambda: 0))
           
            for pronoun in pronouns:
                new_text = text
                new_previous = previous
                for replacement in pronoun:
                    new_word = pronoun[replacement]
                    new_text = new_text.replace(replacement, new_word)
                    new_previous = new_previous.replace(replacement, new_word)
                    seen_sentences[new_text] += 1

                new_relations = []
                for relation in relations:
                    if relation != "hers":
                        #skip the actual pronoun: we are controlling for its variation
                        new_relations.append(relation.lower())
                                    
                for relation in new_relations:
                    variations = bert_predictions.extract_bert_predictions(new_text, new_previous, relation, True)
                    
                    if variations == None:
                        variations = []
                 
                    for variation in variations:
                        sentence_variations[relation][variation] += 1
               
               
            for relation in sentence_variations:                
                # for each word in the sentence related to this one
                variations = sentence_variations[relation]
                for variation in variations:
                    # for each proposed variation on the related words
                    for pronoun in pronouns:
                        # for each pronoun variation
                        new_text = text
                        new_previous = previous
                        for replacement in pronoun:
                            new_word = pronoun[replacement]
                            new_text = new_text.replace(replacement, new_word)
                            new_previous = new_previous.replace(replacement, new_word)
                            
                        # TODO REPLACE WITH TOKEN-LEVEL SUBSTITUTION
                        new_text = new_text.replace(relation, variation)
                        new_previous = new_previous.replace(relation, variation)
                        if seen_sentences[new_text] == 0:
                            seen_sentences[new_text] = 1
                            
                            print("NEW WORD: "+variation+" to replace "+relation)
                            print("NEW CANDIDATE: "+new_text)
                        
                            
                    
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
            
    line = fp.readline()
fp.close()

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
        print(key+" "+str(value))    
    
print(seen_sentences)
  