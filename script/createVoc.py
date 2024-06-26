import os
import json
import nltk
import random
import argparse
import pdb
from tqdm import tqdm


def catDoc(textlist):
    res = []
    for tlist in textlist:
        res.extend(tlist)
    return res

def PrintInformation(keys, allcnt):
    # Vocab > 10
    cnt = 0
    first = 0.0
    for key, val in keys:
        if val >= 10:
            cnt += 1
            first += val
    print("appearance > 10 cnt: %d, percent: %f" % (cnt, first / allcnt))  # 416,303

    # first 30,000, last freq 31
    if len(keys) > 30000:
        first = 0.0
        for k, v in keys[:30000]:
            first += v
        print("First 30,000 percent: %f, last freq %d" % (first / allcnt, keys[30000][1]))

    # first 50,000, last freq 383
    if len(keys) > 50000:
        first = 0.0
        for k, v in keys[:50000]:
            first += v
        print("First 50,000 percent: %f, last freq %d" % (first / allcnt, keys[50000][1]))

    # first 100,000, last freq 107
    if len(keys) > 100000:
        first = 0.0
        for k, v in keys[:100000]:
            first += v
        print("First 100,000 percent: %f, last freq %d" % (first / allcnt, keys[100000][1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='../dataset', help='File to deal with')
    
    args = parser.parse_args()

    path_file= os.path.join(args.data_path, 'train.json')
    saveFile = os.path.join(args.data_path, 'vocab')
    print("Save vocab of dataset  to %s" % saveFile)

    text = []
    summary = []
    allword = []
    cnt = 0
    with open(path_file, encoding='utf8') as f:
        for line in tqdm(f):
            e = json.loads(line)

            text = []
            references = e['reference']
            for key in references:
                ref = references[key]['abstract']
                text.append(ref)

            text.append(e['target_paper'])
            text.append(e['summary'])

            text = ' '.join(text)
            allword.extend(nltk.word_tokenize(text.lower()))
            cnt+= 1
    print("Training set of dataset has %d example" % cnt)

    fdist1 = nltk.FreqDist(allword)

    fout = open(saveFile, "w")
    keys = fdist1.most_common()
    for key, val in keys:
        try:
            fout.write("%s\t%d\n" % (key, val))
        except UnicodeEncodeError as e:
            # print(repr(e))
            # print(key, val)
            continue

    fout.close()

    allcnt = fdist1.N() # 788,159,121
    allset = fdist1.B() # 5,153,669
    print("All appearance %d, unique word %d" % (allcnt, allset))

    PrintInformation(keys, allcnt)
