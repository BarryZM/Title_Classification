# -*- coding: utf-8 -*-
# author: Scandium
# work_location: CSM Peking 
# project : TC
# time: 2020/01/22 11:13

import copy
import csv
import json
import os
import re
import traceback

# 观察分类正交
topic_dic = a2.train_data
write_list = []
for topic in topic_dic:
    for topic2 in topic_dic:
        print(topic, topic2)
        vector_build = a2.pre_vector_build(topic_dic[topic], topic_dic[topic2])
        write_list.append([topic, topic2, a2.cosine_similarity(vector_build[0], vector_build[1])])
        print(a2.cosine_similarity(vector_build[0], vector_build[1]))
ld_to_csv(write_list, r'F:\TC', 'vector_orthogonality')

all_word_list = []
for topic in topic_dic:
    all_word_list += list(topic_dic[topic].keys())
    all_word_list = list(set(all_word_list))
    print(len(all_word_list))

dic_word = {}
tfidf_list = {}
for word in all_word_list:
    for topic in topic_dic:
        try:
            tfidf = topic_dic[topic][word]
            tfidf_list[topic] = tfidf
        except:
            continue
    out_dic = dic_order_by_value(tfidf_list)
    dic_word[word] = tfidf_list
    print(tfidf_list)
    tfidf_list = {}

word_common_distributon = {}
for num in range(2, 16):
    for key in dic_word:
        if len(dic_word[key]) == num:
            word_common_distributon[key] = dic_word[key]
    print(len(word_common_distributon))
    word_common_distributon = {}

word_common_distributon = {}

for key in dic_word:
    if len(dic_word[key]) >= 3:
        word_common_distributon[key] = dic_word[key]
print(len(word_common_distributon))

for key in word_common_distributon:
    word_common_distributon[key] = dic_order_by_value(word_common_distributon[key])

dic_op = topic_dic.copy()


def oprator_basic_vector(input_vector, input_oder_dic):
    for word in input_oder_dic:
        list_del = list(input_oder_dic[word].keys())[2:]
        for num in list_del:
            del input_vector[num][word]


oprator_basic_vector(dic_op, word_common_distributon)

len(dic_word)

for tp in dic_op:
    print(tp, a2.topic_dic[tp], len(topic_dic[tp]))

topic_dic = a2.topic_dic()

a2.calculate_topic()


def ld_to_csv(input_dic, csv_directory, csv_name):
    with open(r'{dic_rectory}\{name}.csv'.format(dic_rectory=csv_directory, name=csv_name), 'w', newline='',
              encoding='gb18030') as csv_w:
        file = csv.writer(csv_w)
        if type(input_dic).__name__ == 'dict':
            for key in input_dic.keys():
                list_write = []
                if type(input_dic[key]).__name__ == 'list':
                    for write_value in input_dic[key]:
                        list_write.append(write_value)
                else:
                    list_write = [key, input_dic[key]]
                    file.writerow(list_write)
        elif type(input_dic).__name__ == 'list':
            for key in input_dic:
                list_write = []
                for write_value in key:
                    list_write.append(write_value)
                file.writerow(list_write)


def file_to_list(file_name, header=False):
    if header:
        start_num = 1
    else:
        start_num = 0
    if file_name.endswith('csv'):
        try:
            with open(file_name, "r", encoding='gb18030') as csv_file:
                # print('gb_csv')
                list_out = []
                csv_r = csv.reader((line.replace('\0', '') for line in csv_file))
                for row in csv_r:
                    list_out.append(row)
                return list_out[start_num:]
        except:
            with open(file_name, "r", encoding='utf-8') as csv_file:
                list_out = []
                csv_r = csv.reader((line.replace('\0', '') for line in csv_file))
                for row in csv_r:
                    list_out.append(row)
                return list_out[start_num:]
    else:
        try:
            list_out = []
            with open(file_name, "r", encoding='gb18030') as file1:
                for row in file1.readlines():
                    list_out.append(row)
            return list_out[start_num:]
        except:
            try:
                list_out = []
                with open(file_name, "r", encoding='utf-8') as file1:
                    for row in file1.readlines():
                        list_out.append(row)
                return list_out[start_num:]
            except:
                print('Can not open', file_name)
                return []


def screen_data_from_dir(input_dir, out_dir, input_value_list, input_key='', maxnum='all'):
    list_file = []
    value_list = input_value_list.copy()

    dic_value_count = {}
    for value in input_value_list:
        dic_value_count[value] = 0

    for root, dirs, files in os.walk(input_dir):
        for name in files:
            list_file.append(os.path.join(root, name))
    if str(input_key).isdigit():
        file_class_index = input_key
    else:
        head_line = file_to_list(list_file[0])[0]
        file_class_index = head_line.index(input_key)
    list_out_file = []

    for file_name in list_file:
        if value_list:
            if str(input_key).isdigit():
                file_list = file_to_list(file_name)
            else:
                file_list = file_to_list(file_name, header=True)

            for data_line in file_list:
                if data_line[file_class_index] in value_list:
                    print(data_line)
                    list_out_file.append(data_line)
                    dic_value_count[data_line[file_class_index]] += 1
                    if maxnum != 'all':
                        if dic_value_count[data_line[file_class_index]] >= int(maxnum):
                            value_list.remove(data_line[file_class_index])
    # list_out_file += list(filter(lambda x: x[file_class_index] in input_value_list, file_list))
    out_file_name = 'test_{input_key}_{maxnum}'.format(input_key=input_key, maxnum=maxnum)
    ld_to_csv(list_out_file, out_dir, out_file_name)


if __name__ == '__main__':
    list_value = ['社会', '法制', '军事', '财经', '旅游', '宠物', '亲子', '科技', '汽车', '美食', '吃货', '情感', '体育', '游戏', '新闻', '娱乐']
    #len(list_value)
    write_list = []
    for topic in dic_op:
        for topic2 in topic_dic:
            print(topic, topic2)
            vector_build = a2.pre_vector_build(topic_dic[topic], topic_dic[topic2])
            write_list.append([topic, topic2, a2.cosine_similarity(vector_build[0], vector_build[1])])
            print(a2.cosine_similarity(vector_build[0], vector_build[1]))
    ld_to_csv(write_list, r'F:\TC', 'vector_orthogonality')
    screen_data_from_dir('F:\PyCharm_project\short_Video_title_classify\es_test_data_10',
                     'F:\PyCharm_project\short_Video_title_classify', list_value, input_key=2, maxnum=1000)
