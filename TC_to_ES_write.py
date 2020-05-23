# -*- coding: utf-8 -*-
# author: Scandium
# work_location: CSM Peking 
# project : TC
# time: 2020/01/15 18:01
import csv
import hashlib
import numpy as np
import os
import pkuseg
import redis
import time
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch.helpers import scan


# rds_title = redis.StrictRedis(host='127.0.0.1', port=6379, db=4, decode_responses=True)

def md5_code(input_text):
    return hashlib.md5(input_text.encode(encoding='UTF-8')).hexdigest()


def ld_to_csv(input_dic, csv_directory, csv_name):  # 将字典或者列表写入csv
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


def dic_order_by_value(input_dic):
    list_tuple = sorted(input_dic.items(), key=lambda input_dic: float(input_dic[1]), reverse=True)
    return dict(list_tuple)


class train_vector:  # 训练集数据设置
    def __init__(self, dir_train):
        self.dir_train = dir_train

    def dic_name(self):
        return os.path.split(self.dir_train)[-1]

    def topic_list(self):
        return os.listdir(self.dir_train)

    def screen(self, dic_train_input):
        dic_out = {}
        for key_topic in dic_train_input:
            dic_all = dic_order_by_value(dic_train_input[key_topic])
            # dic_screened_key =  dic_all.keys()[:10000]
            dic_screened_key = list(dic_all.keys())[:int(0.5 * len(dic_all.keys()))]
            dic_this_key = {}
            for word in dic_this_key:
                dic_this_key[word] = dic_all[word]
            dic_out[key_topic] = dic_all

        return dic_out

    def train_dic_build(self):
        train_dic_path = os.listdir(self.dir_train)
        dic_train_out = {}
        for file in train_dic_path:
            topic = file.split('.')[0]
            path_topic = os.path.join(self.dir_train, file)
            file = file_to_list(path_topic)
            dic_build = dict(zip([i[0] for i in file], [i[1] for i in file]))
            dic_train_out[topic] = dic_build
        dic_asg = self.screen(dic_train_out)
        return dic_asg


class Title_parse():  # 处理标题,发布者,和channel

    def __init__(self):
        self.seg = pkuseg.pkuseg(postag=False)
        self.stop_words = stop_word_build()
        self.topic_dic = {
            '1': '新闻',
            '2': '娱乐',
            '3': '游戏',
            '4': '体育',
            '5': '情感',
            '6': '美食',
            '7': '汽车',
            '8': '科技',
            '9': '家庭',
            '10': '宠物',
            '11': '旅游',
            '12': '财经',
            '13': '军事',
            '14': '法制',
            '15': '社会'
        }
        topic = file_to_list(r'F:\TC\topic_convert.csv')
        topic_out = {}
        for topic_line in topic:
            topic_out[topic_line[0]] = topic_line[1]
        self.topic_covert = topic_out

    def word_divid(self, input_word):
        list_word = self.seg.cut(input_word)
        list_new = [word for word in list_word if word not in self.stop_words]
        return list_new

    def vector_build(self, title_input, releaser_input=''):
        if releaser_input:
            line_words = self.word_divid(releaser_input) + self.word_divid(title_input)
        else:
            line_words = self.word_divid(title_input)
        vector_dic = {}
        for word_uni in set(line_words):
            vector_dic[word_uni] = line_words.count(word_uni)
        return vector_dic

    def channel_judge(self, input_channel):
        if input_channel in self.topic_covert:
            return self.topic_covert[input_channel]
        else:
            return None

    def title_judge(self, input_title):
        try:
            if type(input_title).__name__ == 'NoneType':
                return False
        except:
            print('no_type_title')
        if len(input_title) > 10:
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

    def parse_title_releaser_channel(self, input_title, inpurt_releaser='', input_channel=''):
        channel = self.channel_judge(input_channel)
        if channel:
            if channel != '0':
                return self.topic_dic[channel]
            elif channel == '0':
                return self.vector_build(input_title, inpurt_releaser)
        else:
            if self.title_judge(input_title):
                return self.vector_build(input_title, inpurt_releaser)
            else:
                return None


class Title_classifier():
    def __init__(self, train_dir, topic_good):  # test_dir,
        self.topic_good = topic_good
        # self.test_dir =  test_dir
        self.trans_file = train_dir
        self.train_data = train_vector(train_dir).train_dic_build()
        self.topic_dic = {
            '1': '新闻',
            '2': '娱乐',
            '3': '游戏',
            '4': '体育',
            '5': '情感',
            '6': '美食',
            '7': '汽车',
            '8': '科技',
            '9': '家庭',
            '10': '宠物',
            '11': '旅游',
            '12': '财经',
            '13': '军事',
            '14': '法制',
            '15': '社会'
        }

    def bit_product_sum(self, x, y):
        return sum([item[0] * item[1] for item in zip(x, y)])

    def topic_good(self, topic_num):
        if int(topic_num) in self.topic_good:
            return True
        else:
            return False

    def cosine_similarity(self, x, y, norm=False):  # """ 计算两个向量x和y的余弦相似度 """
        assert len(x) == len(y), "len(x) != len(y)"
        zero_list = [0] * len(x)
        if x == zero_list or y == zero_list:
            return float(1) if x == y else float(0)
        cos = self.bit_product_sum(x, y) / (np.sqrt(self.bit_product_sum(x, x)) * np.sqrt(self.bit_product_sum(y, y)))
        return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内

    def vector_nor1(self, vector_input):
        factor = np.sqrt(sum(float(val) * 2 for val in vector_input))
        if factor != 0:
            vector = [float(value) / factor for value in vector_input]
        else:
            vector = [0 for value in vector_input]
        return vector

    def pre_vector_build(self, dic_test, dic_train):  # 输入两个字典产生相同长度向量
        list_k_topic = list(dic_train.keys())  # [:int(len(dic_train.keys())*0.8)]
        list_c_topic = [float(int_i) for int_i in list(dic_train.values())]
        list_c_test = [0] * len(list_c_topic)
        for word_key in list(dic_test.keys()):
            if word_key in list_k_topic:
                word_index = list_k_topic.index(word_key)
                word_count = dic_test[word_key]
                list_c_test[word_index] = float(word_count)
        return self.vector_nor1(list_c_test), list_c_topic

    def list_to_vector_dic(self, input_list):
        dic_vector = {}
        for word in input_list:
            if word not in dic_vector.keys():
                dic_vector[word] = 1
            else:
                dic_vector[word] += 1
        return dic_vector

    # def ld_to_csv(self,input_dic,csv_directory,csv_name):
    # 	with open( r'{dic_rectory}\{name}.csv'.format(dic_rectory=csv_directory ,name=csv_name),'w',newline='', encoding='gb18030') as csv_w:
    # 		file =  csv.writer(csv_w)
    # 		if type(input_dic).__name__ == 'dict':
    # 			for key in input_dic.keys():
    # 				list_write = [key]
    # 				if type(input_dic[key]).__name__ == 'list':
    # 					for write_value in input_dic[key]:
    # 						list_write.append(write_value)
    # 				else:
    # 					list_write = [key,input_dic[key]]
    # 				file.writerow(list_write)
    # 		elif type(input_dic).__name__ == 'list':
    # 			for key in input_dic:
    # 				list_write = []
    # 				for write_value in key:
    # 					list_write.append(write_value)
    # 				file.writerow(list_write)

    def dic_order_by_value(self, input_dic):
        list_tuple = sorted(input_dic.items(), key=lambda input_dic: input_dic[1], reverse=True)
        return dict(list_tuple)

    def calculate_best_num_topic(self, test_data,
                                 topic_num):  # 找到最合适的topic train_data为训练集产生的主题字典例如:{topic1:{word:199},topic2:{word:209}}
        test_w_dic = test_data
        best_topic_dic = {}
        for train_topic in self.train_data.keys():
            topic_w_dic = self.train_data[train_topic]
            list_trp = self.pre_vector_build(test_w_dic, topic_w_dic)
            distance_out = self.cosine_similarity(list_trp[0], list_trp[1])
            best_topic_dic[train_topic] = distance_out
        best_topic_dic = self.dic_order_by_value(best_topic_dic)
        return list(best_topic_dic.keys())[:topic_num], list(best_topic_dic.values())[:topic_num]

    def calculate_topic(self, input_sentence):  # 输入向
        if input_sentence:
            topic_list = self.calculate_best_num_topic(input_sentence, 5)
            out_topic_list = []
            cos_value_list = topic_list[1]
            # print(cos_value_list)
            for cos_num in range(len(cos_value_list) - 1):
                if cos_value_list[cos_num] >= 0.05:
                    out_topic_list.append(self.topic_dic[topic_list[0][cos_num]])
                    # print(cos_value_list[cos_num])
                    if float(float(cos_value_list[cos_num]) / float(
                            cos_value_list[cos_num + 1] + 0.0000000000000001)) > 1.2:
                        break
        else:
            out_topic_list = []
        # if float(float(cos_value_list[cos_num])/float(cos_value_list[cos_num+1])) > 1.2 and cos_value_list[cos_num] >= 0.01:#标题筛选设置,从cos值大于0.01,主分类比后分类foldchange> 1.5
        # 	print(float(float(cos_value_list[cos_num]) / float(cos_value_list[cos_num + 1])))
        # 	out_topic_list.append(self.topic_dic[topic_list[0][cos_num]])
        return out_topic_list


class Es_operator():
    def __init__(self):
        es_option = {
            'host_reader': '192.168.17.11',
            'host_writer': '192.168.6.34',
            'port_reader': 80,
            'port_writer': 9200,
            'user': 'liukang',
            'passwd_reader': 'xSEHhTRGE6AX',
            'passwd_writer': 'xSEHhTRGE6AX'
        }
        # self.http_auth = (es_option['user'], es_option['passwd'])
        self.es_reader = Elasticsearch(hosts=es_option['host_reader'], port=es_option['port_reader'],
                                       http_auth=(es_option['user'], es_option['passwd_reader']))
        self.reader_option = {"query": {"term": {"_type": "all-time-url"}}}
        self.es_writer = Elasticsearch(hosts=es_option['host_writer'], port=es_option['port_writer'],
                                       http_auth=(es_option['user'], es_option['passwd_writer']))
        self.writer_option = {"title": "", "tags": "", "timestamp": 0}
        self.title_cal_topic = Title_classifier(train_dir, tpoic_good)
        self.parse_data = Title_parse()

    def scan_build_result(self, index):
        es_result = scan(
            client=self.es_reader,
            query=self.reader_option,
            scroll='50m',
            index=index,
            timeout="2m",
            raise_on_error=False)
        return es_result

    def title_classify(self, data_input):
        print(data_input)
        if data_input:
            # data为 input_title, inpurt_releaser='', input_channel=''
            title = data_input[0]
            try:
                releaser = data_input[1]
            except:
                releaser = ''
            try:
                channel = data_input[2]
            except:
                channel = ''
            title_cal_topic = self.title_cal_topic
            parse_data = self.parse_data
            data_parsed = parse_data.parse_title_releaser_channel(title, releaser, channel)
            print(data_parsed)
            if not data_parsed:
                return None
            if isinstance(data_parsed, str):
                return [title, data_parsed]
            if isinstance(data_parsed, dict):
                out_tags = title_cal_topic.calculate_topic(data_parsed)
                tgs_out = ','.join(out_tags)
                return [title, tgs_out]
        else:
            print('3')
            return None

    def write_data_es(self, data_list):  # 输入列表[title,tags] or [[title1,tags1],[title2,tags2],.....]
        timestamp = int(time.time())
        if isinstance(data_list[0], list):
            actions = []
            for data_line in data_list:
                action = {
                    "_index": "title_classification",
                    "_type": "classified_tags",
                    "_id": md5_code(data_line[0]),
                    "_score": 1,
                    "_source": {
                        "title": data_line[0],
                        "tags": data_line[1],
                        "timestamp": timestamp
                    }
                }
                actions.append(action)
        else:
            action = {
                "_index": "title_classification",
                "_type": "classified_tags",
                "_id": md5_code(data_list[0]),
                "_score": 1,
                "_source": {
                    "title": data_list[0],
                    "tags": data_list[1],
                    "timestamp": timestamp
                }
            }
            actions = [action]
        print(actions)
        bulk(self.es_writer, actions)

    def es_title_fetch(self, num='all'):  # 获取一定数目的ES数据
        es_scan = self.scan_build_result('short-video-all-time-url-v2')
        final_result_list = []
        count_cead = 0
        write_num = 0
        for item in es_scan:
            print('mark_here')
            count_cead += 1
            try:
                es_channel = item['_source']["channel"]
            except Exception as e:
                print('Error:', e)
                es_channel = 'None'
            finally:
                final_result_list.append([item['_source']["title"], item['_source']["releaser"], es_channel])
                print(item['_source']["title"])
            if len(final_result_list) >= 10:
                write_num += 1
                print(write_num)
                title_classfied = [self.title_classify(uni_data) for uni_data in final_result_list]
                for data in title_classfied:
                    parse_data = self.title_classify(data)

                    if parse_data:
                        if parse_data[1]:
                            self.write_data_es(parse_data)
                final_result_list = []
                count_cead = 0
                if num == 'all':
                    pass
                elif write_num * 10 >= int(num):
                    break

    # title_classfied = [self.title_classify(uni_data) for uni_data in final_result_list]
    # for data in title_classfied:
    # 	parse_data = self.title_classify(data)
    # 	self.write_data_es(parse_data)


train_dir = r'F:\TC\tfidf_retrain'

tpoic_good = list(range(1, 16))

e1 = Es_operator()

e1.es_title_fetch('200')

import time

start = time.process_time()
e1.es_title_fetch('20')
end = time.process_time()
print(end - start)

tags = e1.title_classify(data_input)

a1 = Title_parse()

a2 = Title_classifier(train_dir, tpoic_good)

#
# e1.write_data_es(['诸暨市广播电视台视听诸暨', '军事'])
#
# data_input = ["这种癌症不痛不痒，一发现就是晚期！身体出现这5个信号是警报！","",""]
#
# tags = e1.title_classify(data_input)
#
# print(tags)
#
# es_option = {
# 	'host_reader': '192.168.17.11',
# 	'host_writer': '192.168.6.34',
# 	'port_reader': 80,
# 	'port_writer': 9200,
# 	'user': 'liukang',
# 	'passwd_reader': 'xSEHhTRGE6AX',
# 	'passwd_writer': 'xSEHhTRGE6AX'
# }
# # self.http_auth = (es_option['user'], es_option['passwd'])
# es_reader = Elasticsearch(hosts=es_option['host_reader'], port=es_option['port_reader'],
# 							   http_auth=(es_option['user'], es_option['passwd_reader']))
# reader_option = {"query":{"term": {"_type": "all-time-url"}}}
#
# es_writer = Elasticsearch(hosts=es_option['host_writer'], port=es_option['port_writer'],
# 							   http_auth=(es_option['user'], es_option['passwd_writer']))
# writer_option = {"title": "", "tags": "", "timestamp": 0}
# title_cal_topic = Title_classifier(train_dir, tpoic_good)
# parse_data = Title_parse()
#
#
# def scan_build_result(index):
# 	es_result = scan(
# 		client=es_reader,
# 		query=reader_option,
# 		scroll='50m',
# 		index=index,
# 		timeout="2m",
# 		)
# 	return es_result
#
# es_scan = scan_build_result('short-video-all-time-url-v2')
#
#
# count_cead = 0
# write_num = 0
# for item in es_scan:
# 	print('mark_here')
# 	count_cead += 1
#
# print(isinstance([1,2], list))
#
# import elasticsear
# #此行之下皆为测试
#
# es_option = {
# 	'host_reader': '192.168.17.11',
# 	'host_writer': '192.168.6.34',
# 	'port': 9200,
# 	'user': 'liukang',
# 	'passwd_reader': 'xSEHhTRGE6AX',
# 	'passwd_writer': 'xSEHhTRGE6AX'
# }
#
# es_writer = Elasticsearch(hosts = es_option['host_writer'], port= es_option['port'], http_auth= (es_option['user'], es_option['passwd_writer']))
#
# writer_option = {"title" :"","tags" :"","timestamp":0}
#
# title = "老公外遇，回到家直接被老婆关门外，晚上不要进去睡觉了"
#
# action = {
#         "_index": "title_classification",
#         "_type": "classified_tags",
#         "_id": md5_code(title),
#         "_score": 1,
#         "_source": {
#           "title": title,
#           "tags": "体育",
#           "timestamp": int(time.time())
# 					}
# 		}
#
# action = [{'_index': 'title_classification', '_type': 'classified_tags', '_id': '37567f29ff75eb5e77a3a45aafc467ff', '_score': 1, '_source': {'title': '诸暨市广播电视台视听诸暨', 'tags': '军事', 'timestamp': 1579520799}}]
#
# bulk(es_writer, action)
#
# # 分类范畴
# topic_sort = list(range(0, 10))
# # 训练集目录
# train_dir = r"F:\TC\tfidf_retrain"
# # 测试集数据
# test_dic =  test_vector_build(r'F:\TCtest_file', r'F:\TC\test_word_list')
#
# dic_test = test_dic.dic_word_list_include_releaser()
#
# test_dic.write_list()
#
# test_dic_omi = test_dic.test_dic_build()
#
# topic_good = list(range(1, 16))
# # 分类器
# tc = Title_classifier(train_dir, topic_good)
#
# print(tc.calculate_best_num_topic(test_dic_omi['4'][256], 2))
# print(test_dic_omi['4'][256])
# s2 = test_dic_omi['4'][256]
# a1 = {'冯小刚':1, '病情':1, '恶化':1, '钱':1, '治':1, '徐帆':1, '坦言':1, '依旧':1, '抽烟':1, '喝酒':1, '放弃':1, '治疗':1}
# a2 = [1] * 12
# tc.calculate_best_num_topic(a1, 5)
# tc.vector_nor1(a2)
#
# start = time.process_time()
# for i in test_dic_omi['2'][500:600]:
# 	a = tc.calculate_best_num_topic(i, 2)
# 	if a[0][0] != '2_TFIDF':
# 		print(i,a[0][0])
# end = time.process_time()
# print(end - start)
#
#
# dic_t = test_dic_omi['4'][:100]
# dic_t1 = {}
# dic_t1['4'] = dic_t
#
# tc = Title_classifier(train_dir, topic_good)
#
# get_precison(dic_t1)
#
# get_precison_cos(dic_t1)
#
#
# import time
# start = time.process_time()
# for i in range(10):
# 	a = tc.calculate_best_num_topic(s2, 1)
# 	print(a)
# end = time.process_time()
# print(end-start)
