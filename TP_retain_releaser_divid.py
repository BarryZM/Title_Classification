# -*- coding: utf-8 -*-
# author: Scandium
# work_location: CSM Peking 
# project : TC
# time: 2020/01/02 21:05

import pkuseg
import csv
# import jieba.posseg as jp
import os


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


def dic_order_by_value(input_dic):
	list_tuple = sorted(input_dic.items(), key=lambda input_dic: input_dic[1], reverse=True)
	return dict(list_tuple)


# 此行之下皆为测试

stop_words = stop_word_build()

seg = pkuseg.pkuseg(postag=False)

es_dir = r'F:\TC\retrain_2_topic'

for topic_file_name in os.listdir(es_dir):
	file_open = file_to_list(os.path.join(es_dir, topic_file_name))
	print(topic_file_name)
	for line in file_open:
		releaser = line[0]
		releaser_div = seg.cut(releaser)
		new_line = releaser_div + line[1:]
		write_line = [word for word in new_line if word not in stop_words]
		with open("F:\TC\div_releaser\{topic_num}".format(topic_num=topic_file_name),
				  'a+', newline='', encoding='gb18030') as csv_file:
			csv_w = csv.writer(csv_file)
			csv_w.writerow(write_line)
		csv_file.close()


# 此行之下皆为测试
