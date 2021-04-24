import argparse
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab')
    parser.add_argument('--vocab2')
    parser.add_argument('--out')
    args = parser.parse_args()

    vocabf  = args.vocab
    vocab2f  = args.vocab2
    outfile = args.out
    vocab = []
    vocab2 = []

    with open(vocabf,encoding="utf-8") as f:
        vocab =  set([word.strip() for word in f.readlines()])
    with open(vocab2f,encoding="utf-8") as f:
        vocab2 =  set([word.strip() for word in f.readlines()])


    oov = []
    for word in vocab2:
        if word not in vocab:
            oov.append(word)
    
    
        
    with open(outfile,"w",encoding="utf-8") as f:
        f.write(r"oov count = {}/{}\n".format(len(oov),len(vocab2)))
        for word in oov:
            f.write(word+"\n")