import sys
import re
import collections
import os
from collections import defaultdict    
#import bert_predictions


his = ['night', 'girl', 'baby', 'money', 'world', 'key', 'boy', 'rest', 'name', 'future', 'choice', 'child', 'power', 'car', 'pleasure', 'fault', 'fun', 'it', 'room', 'job', 'town', 'body', 'gun', 'pain', 'life', 'man', 'crown', 'lord', 'answer', 'one', 'work', 'business', 'piece', 'alex', 'story', 'product', 'head', 'mother', 'face', 'place', 'father', 'soul', 'now', 'door', 'hand', 'sun', 'heart', 'kid', 'house', 'dealer', 'painting', 'paint', 'fresh', 'side', 'way', 'hair', 'shoes', 'hands', 'smell', 'ass', 'stuff', 'clothes', 'sold', 'price', 'selling', 'sale', 'blood', 'past', 'scent', 'deals', 'reputation', 'sales', 'deal', 'drugs', 'up', 'his', 'butt', 'mess', 'stock', 'goods', 'bid', 'buy', 'cash', 'shop', 'floor', 'streets', 'fingers', 'muscles', 'rhythm', 'length', 'shaft', 'strength', 'back', 'line', 'pockets', 'bottles', 'barrel', 'glass', 'inside', 'bottle', 'front', 'nose', 'bed', 'shirt', 'other', 'wound', 'knife', 'first', 'mom', 'arm', 'parents', 'eyes', 'last', 'mouth', 'top', 'coat', 'him', 'people', 'faces', 'things', 'bodies', 'doing', 'everything', 'scars', 'them', 'with', 'paintings', 'get']

hers = ['money', 'girl', 'baby', 'name', 'necklace', 'voice', 'night', 'house', 'boy', 'rest', 'body', 'world', 'child', 'car', 'decision', 'choice', 'man', 'pleasure', 'power', 'kid', 'island', 'one', 'castle', 'bar', 'bed', 'food', 'room', 'town', 'music', 'land', 'place', 'book', 'heart', 'product', 'toy', 'jewelry', 'perfume', 'business', 'story', 'work', 'father', 'mother', 'head', '?', '.', 'mom', 'soul', 'spirit', 'sister', 'camera', 'sun', 'hand', 'door', 'table', 'phone', 'gun', 'blonde', 'face', 'painting', 'paint', 'way', 'side', 'hair', 'hands', 'wife', 'blood', 'stuff', 'cars', 'past', 'daughter', 'ability', 'methods', 'life', 'easily', 'talent', 'deal', 'scars', 'hers', 'home', 'up', 'clothes', 'stock', 'goods', 'shoes', 'paintings', 'shop', 'floor', 'fingers', 'legs', 'mind', 'strength', 'lips', 'eyes', 'rhythm', 'speed', 'muscles', 'mouth', 'curves', 'engines', 'thighs', 'heat', 'sound', 'back', 'glass', 'bottle', 'kitchen', 'box', 'pockets', 'drawers', 'other', 'bathroom', 'shirt', 'dress', 'leg', 'family', 'front', 'mess', 'alex', 'bodies', 'things', 'everything', 'them', 'her', 'it', 'faces', 'women', 'people', 'girls', 'account']

theirs = ['theirs', 'action', 'answer', 'baby', 'back', 'best', 'blood', 'bodies', 'body', 'box', 'boy', 'business', 'camera', 'car', 'city', 'clothes', 'crew', 'customers', 'deal', 'dealer', 'door', 'drawers', 'drivers', 'drugs', 'engines', 'everything', 'eye', 'face', 'family', 'father', 'first', 'fish', 'floor', 'friends', 'front', 'girl', 'glass', 'goods', 'hair', 'hand', 'head', 'heart', 'horses', 'house', 'innocence', 'instincts', 'island', 'jewelry', 'job', 'junk', 'kid', 'land', 'last', 'leg', 'life', 'likes', 'lot', 'men', 'mess', 'minds', 'mom', 'money', 'mother', 'name', 'night', 'one', 'paint', 'painting', 'parents', 'party', 'past', 'people', 'place', 'pleasure', 'pockets', 'power', 'product', 'rest', 'room', 'same', 'scent', 'sheriff', 'ship', 'shit', 'shoes', 'shop', 'soul', 'streets', 'stuff', 'sun', 'sword', 'table', 'team', 'things', 'tires', 'town', 'toys', 'tracks', 'two', 'water', 'way', 'wheels', 'windows', 'work', 'world']


all = hers + his + theirs
all.sort()

seen = {}

hr = []
hi = []
th = []
hrhi = []
hrth = []
hith = []
hrhith = []

for word in all:
    if word in seen:
        continue

    if word in hers:
        if word in his:
            if word in theirs:
                hrhith.append(word)
            else:
                hrhi.append(word)
        elif word in theirs:
            hrth.append(word)
        else:
            hr.append(word)
    elif word in his:
        if word in theirs:
            hith.append(word)
        else:
            hi.append(word)
    else:
        th.append(word)
    
    seen[word] = True
        
print("\t".join(hr)+"\n")
print("\t".join(hi)+"\n")
print("\t".join(th)+"\n")
print("\t".join(hrhi)+"\n")
print("\t".join(hrth)+"\n")
print("\t".join(hith)+"\n")
print("\t".join(hrhith)+"\n")

    
    
    




exit()


seed_data = "./UD_English-Pronouns/en_pronouns-ud-test.conllu"
reference_data = "./ud-treebanks-v2.4/UD_English-LinES/en_lines-ud-train.conllu"

if not os.path.exists(seed_data):
    print("No file found: "+seed_data+"\nThis should contain your starting files")

if not os.path.exists(reference_data):
    print("No file found: "+reference_data+"\nTry downloading via:\ncurl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2988{/ud-treebanks-v2.4.tgz,/ud-documentation-v2.4.tgz,/ud-tools-v2.4.tgz}")



variations = bert_predictions.extract_bert_predictions("Their name is Griffin mask Monarch", "What is their name?", "mask", [],True, max_samples=1000, good_turing_threshold=0.01)

print(variations)

variations = bert_predictions.extract_bert_predictions("Dr Griffin mask Monarch", "Who is it?", "mask", [],True, max_samples=1000, good_turing_threshold=0.01)

print(variations)


