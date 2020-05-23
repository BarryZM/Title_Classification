# -*- coding: utf-8 -*-
# author: Scandium
# work_location: CSM Peking 
# project : Title_Classify
# time: 2020/01/02 16:52

import os, re, csv, json, traceback
import csv
import os
import numpy as np


def topic_good(topic_num):
	if topic_num in topic_sort:
		return True
	else:
		return False


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


def file_to_sum_dic(input_list):
	dic_vector = {}
	for line in input_list:
		for word in line:
			if word not in dic_vector.keys():
				dic_vector[word] = 1
			else:
				dic_vector[word] += 1
	return dic_vector


def count_if_init(word, wordcount):
	return sum(1 for i in wordcount if word in i)


def total_dic_from_dir(input_dir):
	dic_total_out = {}
	file_list = os.listdir(input_dir)
	for file_name in file_list:
		top_num = file_name.split('.csv')[0]
		if topic_good(int(top_num)):
			with open(os.path.join(input_dir, file_name), "r", encoding='gb18030') as csv_file:
				list_out = []
				csv_r = csv.reader(csv_file)
				for row in csv_r:
					list_out.append(row)  # [1:])#修改为直接加入发布者信息
				dic_total_out[top_num] = list_out
	return dic_total_out


def all_word_list_build(dic_total_input):
	all_word_list_out = []
	for key in dic_total_input:
		print(key)
		list_this_key = []
		for list_word in dic_total_input[key]:
			list_this_key = list(set(list_this_key).union(set(list_word)))
		all_word_list_out = list(set(all_word_list_out).union(set(list_this_key)))
	return all_word_list_out


def all_length_build(dic_total_input):
	dic_out_of_length_by_topic = {}
	for key in dic_total_input:
		dic_out_of_length_by_topic[key] = len(dic_total_input[key])
	return dic_out_of_length_by_topic


def dic_order_by_value(input_dic):
	list_tuple = sorted(input_dic.items(), key=lambda input_dic: input_dic[1], reverse=True)
	return dict(list_tuple)


def tdidf_cal_and_write(dic_total_input, all_word_list_input):
	lenth_dic = all_length_build(dic_total_input)
	all_length = sum(lenth_dic.values())
	all_word_list = all_word_list_input
	for word in all_word_list:
		topic_w_list = []
		tfidf_w_list = []
		dic_the_word = {}
		for topic_key in dic_total_input:
			word_num_no = count_if_init(word, dic_total_input[topic_key])
			dic_the_word[topic_key] = word_num_no
		all_this_word_num_in_topics = sum(dic_the_word.values())
		for wt_topic in dic_the_word:
			# tfidf 计算方式 能想到的有两种,idf计算为log(语料库的文档总数 / (包含词w的文档数 + 1))  或者 log(语料库的文档总数- 该类中所有的词条数目 / (除了这一类包含词w的文档数 + 1)) 目前使用方式1
			word_tfidf = (dic_the_word[wt_topic] / lenth_dic[wt_topic]) * (
						np.log(all_length) / all_this_word_num_in_topics + 1)
			if word_tfidf != 0:
				with open("{dir_tfidf_vector}\{topic_num}_TFIDF.csv".format(dir_tfidf_vector=dir_tfidf_vector,
																			topic_num=wt_topic),
						  'a+', newline='', encoding='gb18030') as csv_file:
					csv_w = csv.writer(csv_file)
					word_wtl = [word, word_tfidf]
					csv_w.writerow(word_wtl)
				csv_file.close()


if __name__ == '__main__':
	topic_sort = list(range(1, 16))  # + list(range(100, 123))
	dir_pre_tfidf = r'F:\TC\div_releaser'
	dir_tfidf_vector = r'F:\TC\tfidf_retrain'
	total_dic = total_dic_from_dir(dir_pre_tfidf)
	all_word_list = all_word_list_build(total_dic)
	tdidf_cal_and_write(total_dic, all_word_list[1:])

if __name__ == '__main__':
	pass

#此行之下皆为测试

