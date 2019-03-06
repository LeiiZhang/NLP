#encoding=utf-8

#==============================================================================
# words_list=[]
# with open('source.txt', 'r+',encoding='utf-8') as fr:
#     with open('my_source.txt', 'w+',encoding='utf-8') as fw:
#         sentence_list=[]
#         for line in fr.readlines():
#             sentence_list = line.strip().split()
#             for word in sentence_list:
#                 fw.write(word)
#                 fw.write('\n')
#             fw.write('\n')
# with open('source_target.txt', 'r+',encoding='utf-8') as fr:
#     with open('my_source_target.txt', 'w+',encoding='utf-8') as fw:
#         sentence_list=[]
#         for line in fr.readlines():
#             sentence_list = line.strip().split()
#             for word in sentence_list:
#                 fw.write(word)
#                 fw.write('\n')
#             fw.write('\n')
#==============================================================================


words_list=[]
targets_list=[]
with open('test.txt', 'r+',encoding='utf-8') as fr:
    sentence_list=[]
    for line in fr.readlines():
        sentence_list = line.strip().split()
        words_list.append(sentence_list)

with open('test_target.txt', 'r+',encoding='utf-8') as fr:
    sentence_list=[]
    for line in fr.readlines():
        sentence_list = line.strip().split()
        targets_list.append(sentence_list)

with open('test_data', 'w+',encoding='utf-8') as fw:
    for i in range(len(targets_list)):
        for j in range(len(words_list[i])):
            fw.writelines([words_list[i][j],'\t',targets_list[i][j],'\n'])
        fw.write('\n')

 

           