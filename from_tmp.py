import torch
import pickle

path = "/tmp/lll.pickle"

with open(path, 'rb') as f:
    corpus = pickle.load(f)
a = dict(corpus)
source = []
target = []

for k, v in a.items():
    print("key:", k)
    for sub in v:
        if isinstance(sub[1], str) and isinstance(sub[2], str) and isinstance(sub[4], str):
            sub_title = sub[1].strip().replace("\n", " ").lower()
            sub_text = sub[2].strip().replace("\n", " ").lower()
            sub_question = sub[4].strip().replace("\n", " ").lower()
            if sub_title[-1] == ".":
                sub_title = sub_title[:-1]
            if sub_question[-1] != "?":
                sub_question += "?"
            print("sub_title:", sub_title)
            print("sub_text:", sub_text)
            print("sub_question:", sub_question)
            source.append(('\\n').join([sub_question, sub_text]))
            target.append(sub_title)

with open("./input/train.source", 'w') as f:
    for item in source:
        f.write(item + '\n')
with open("./input/train.target", 'w') as f:
    for item in target:
        f.write(item + '\n')
