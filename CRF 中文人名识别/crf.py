#coding=utf-8
import re
import jieba.posseg as pseg
import codecs
import numpy as np
from sklearn.model_selection import KFold
import os
import sys

file_read = codecs.open('data_nr.txt','r', 'utf-8')

raw_data = file_read.readlines()
sentences = np.array(raw_data)
file_read.close()

def create_train_data(words, fwrite):
    for word in words:
        word = word.strip(' ')
        if len(word) > 1:
            if '/nr' in word:
                word = word.rstrip('/nr')
                for i in range(len(word)):
                    if i == 0:
                        fwrite.write(word[i] + '\t' + 'B' + '\t' + 'nr' + '\t' + 'B' + '\n')
                    elif i == len(word) - 1:
                        fwrite.write(word[i] + '\t' + 'E' + '\t' + 'nr' + '\t' + 'I' + '\n')
                    else:
                        fwrite.write(word[i] + '\t' + 'M' + '\t' + 'nr' + '\t' + 'I' + '\n')
            else:
                wf = pseg.cut(word)
                w, flag = next(wf)
                for i in range(len(word)):
                    if i == 0:
                        fwrite.write(word[i] + '\t' + 'B' + '\t' + flag + '\t' + 'O' + '\n')
                    elif i == len(word) - 1:
                        fwrite.write(word[i] + '\t' + 'E' + '\t' + flag + '\t' + 'O' + '\n')
                    else:
                        fwrite.write(word[i] + '\t' + 'M' + '\t' + flag + '\t' + 'O' + '\n')
        elif len(word) == 1:
            wf = pseg.cut(word)
            w, flag = next(wf)
            fwrite.write(word + '\t' + 'S' + '\t' + flag + '\t' + 'O' + '\n')

def create_test_data(words, fwrite):
    for word in words:
        word = word.strip(' ')
        if len(word) > 1:
            if '/nr' in word:
                word = word.rstrip('/nr')
                for i in range(len(word)):
                    if i == 0:
                        fwrite.write(word[i] + '\t' + 'B' + '\t' + 'nr' + '\n')
                    elif i == len(word) - 1:
                        fwrite.write(word[i] + '\t' + 'E' + '\t' + 'nr' + '\n')
                    else:
                        fwrite.write(word[i] + '\t' + 'M' + '\t' + 'nr' + '\n')
            else:
                wf = pseg.cut(word)
                w, flag = next(wf)
                for i in range(len(word)):
                    if i == 0:
                        fwrite.write(word[i] + '\t' + 'B' + '\t' + flag + '\n')
                    elif i == len(word) - 1:
                        fwrite.write(word[i] + '\t' + 'E' + '\t' + flag + '\n')
                    else:
                        fwrite.write(word[i] + '\t' + 'M' + '\t' + flag + '\n')
        elif len(word) == 1:
            wf = pseg.cut(word)
            w, flag = next(wf)
            fwrite.write(word + '\t' + 'S' + '\t' + flag + '\n')

print('train sentences: ', int(len(sentences)*0.8))
print('test sentences: ' , int(len(sentences)*0.2))

kf = KFold(n_splits=5)
epoch = 1
for train_index, test_index in kf.split(sentences):
    print('epoch : ', epoch)

    file_train_write = codecs.open('train.txt', 'w', 'utf-8')
    file_test_write = codecs.open('test.txt', 'w', 'utf-8')
    file_expect = codecs.open('expect.txt', 'w', 'utf-8')

    raw_train_data = sentences[train_index]
    raw_test_data = sentences[test_index]

    for line in raw_train_data:
        words = line.strip('\n').split(' ')
        create_train_data(words, file_train_write)
        file_train_write.write('\n')

    for line in raw_test_data:
        words = line.strip('\n').split(' ')
        create_test_data(words, file_test_write)
        create_train_data(words, file_expect)
        file_test_write.write('\n')
        file_expect.write('\n')

    file_train_write.close()
    file_test_write.close()
    file_expect.close()

    os.system('crf_learn -f 4 -c 1.5 temp.txt train.txt crf_model')
    os.system('crf_test -m crf_model test.txt > predict.txt')

    epoch += 1

