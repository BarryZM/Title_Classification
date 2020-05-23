# -*- coding: utf-8 -*-
# author: Scandium
# work_location: CSM Peking 
# project : Title_Classify
# time: 2020/01/02 17:04

import csv
# import jieba.posseg as jp
import os
import pkuseg


def file_to_list(file_name):
    if file_name.endswith('csv'):
        try:
            with open(file_name, "r", encoding='utf-8') as csv_file:
                list_out = []
                csv_r = csv.reader((line.replace('\0', '') for line in csv_file))
                for row in csv_r:
                    list_out.append(row)
                return list_out
        except:
            # print('gb_csv')
            with open(file_name, "r", encoding='gb18030') as csv_file:
                list_out = []
                csv_r = csv.reader((line.replace('\0', '') for line in csv_file))
                for row in csv_r:
                    list_out.append(row)
                return list_out
    else:
        try:
            list_out = []
            with open(file_name, "r", encoding='utf-8') as file1:
                for row in file1.readlines():
                    list_out.append(row)
            return list_out
        except:
            try:
                list_out = []
                with open(file_name, "r", encoding='utf-8') as file1:
                    for row in file1.readlines():
                        list_out.append(row)
                return list_out
            except:
                print('Can not open', file_name)
                return []


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


def del_lastN(input_list):
    out_list = []
    for i in range(len(input_list)):
        out_list.append(input_list[i].strip('\n'))
    return out_list


def stop_word_build():
    stop_words = del_lastN(file_to_list('stop_words'))
    stop_words_add = del_lastN(file_to_list('stop_word_add'))
    stop_words_recover = del_lastN(file_to_list('stop_word_recover'))
    stop_word = set(stop_words + stop_words_add)
    out_stop_word = [word for word in stop_word if word not in stop_words_recover]
    return out_stop_word


def list_seg(list_input):
    out_list = []
    for str_line in list_input:
        uni_str = []
        row_list = seg.cut(str_line)
        for word_cp in row_list:
            if word_cp[0] not in stop_words:
                uni_str.append(word_cp)
        out_list.extend(uni_str)
    return out_list


def file_to_sum_dic(input_list):
    dic_vector = {}
    for line in input_list:
        for word in line:
            if word not in dic_vector.keys():
                dic_vector[word] = 1
            else:
                dic_vector[word] += 1
    return dic_vector


def title_judge(input_title):
    try:
        if type(input_title).__name__ == 'NoneType':
            return False
    except:
        print('no_type_title')
    if len(input_title) > 6:
        include_chinese = 0
        for ch in input_title:
            if u'\u4e00' <= ch <= u'\u9fff':
                include_chinese += 1
        if include_chinese >= 6:
            return True
        else:
            return False
    else:
        return False


def dic_order_by_value(input_dic):
    list_tuple = sorted(input_dic.items(), key=lambda input_dic: input_dic[1], reverse=True)
    return dict(list_tuple)


# 此行之下皆为测试
if __name__ == '__main__':
    stop_words = stop_word_build()

    seg = pkuseg.pkuseg(postag=False)

    es_dir = r'F:\PyCharm_project\short_Video_title_classify\es_test_data_11'

    list_topic_convert = file_to_list('topic_convert.csv')

    dic_topic_convert = dict(zip([i[0] for i in list_topic_convert], [i[1] for i in list_topic_convert]))

    topic_type = "22_topic"
    train_dic = {}
    test_dic = {}

    for topic_file_name in os.listdir(es_dir):
        file_open = file_to_list(os.path.join(es_dir, topic_file_name))
        print(topic_file_name)
        for line in file_open:
            if title_judge(line[1]):
                channel = line[2]
                if channel in dic_topic_convert.keys():
                    topic = dic_topic_convert[channel]
                else:
                    topic = '0'
                word_sentence = list_seg([line[1]])  # 修改发布者还是标题 [:2]合辑 [line[1]]标题  [line[0]]发布者
                word_sentence.insert(0, line[0])
                with open(
                        "F:\PyCharm_project\short_Video_title_classify\pre_tfidf\{topic_num}_.csv".format(topic_num=topic),
                        'a+', newline='', encoding='gb18030') as csv_file:
                    csv_w = csv.writer(csv_file)
                    csv_w.writerow(word_sentence)
                csv_file.close()

    for topic in train_dic.keys():
        name = 'topic_{t}_total_{type}'.format(t=topic, type=topic_type)
        ld_to_csv(dic_order_by_value(train_dic[topic]),
                  r'F:\PyCharm_project\short_Video_title_classify\init_number_count_dic_11', name)
