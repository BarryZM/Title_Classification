# -*- coding: utf-8 -*-
# author: Scandium
# work_location: CSM Peking 
# project : Title_Classify
# time: 2019/12/31 17:13
import os, re, csv, json, traceback
import numpy as np
import os, csv
import pkuseg
import copy


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
	list_tuple = sorted(input_dic.items(), key=lambda input_dic: input_dic[1], reverse=True)
	return dict(list_tuple)


class train_vector:  # 训练集数据设置
	def __init__(self, dir_train):
		self.dir_train = dir_train

	def dic_name(self):
		return os.path.split(self.dir_train)[-1]

	def topic_list(self):
		return os.listdir(self.dir_train)

	def screen(self,dic_train_input):
		dic_out = {}
		for key_topic in dic_train_input:
			dic_all = dic_order_by_value(dic_train_input[key_topic])
			#dic_screened_key =  dic_all.keys()[:10000]
			dic_screened_key = list(dic_all.keys())[:int(0.5*len(dic_all.keys()))]
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


class Title_classifier():
	def __init__(self, train_dir, topic_good):  # test_dir,
		self.topic_good = topic_good
		# self.test_dir =  test_dir
		self.trans_file = train_dir
		self.train_data = train_vector(train_dir).train_dic_build()

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

	def pre_vector_build(self, dic_1, dic_2):  # 输入两个字典产生相同长度向量
		list_k_topic = list(dic_2.keys())
		list_c_topic = [float(int_i) for int_i in list(dic_2.values())]
		list_c_test = [0] * len(list_c_topic)
		for word_key in list(dic_1.keys()):
			if word_key in list_k_topic:
				word_index = list_k_topic.index(word_key)
				word_count = dic_1[word_key]
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

	def calculate_topic(self,input_sentence):
		topic_list = self.calculate_best_num_topic(input_sentence, 5)
		out_topic_list = []
		for num in range(len(topic_list[0])):
			if topic_list[1][num] > 0.01:
				out_topic_list.append(topic_list[0][num])
		return out_topic_list


class test_vector_build():  # 测试集数据设置

	def __init__(self, dir_test, dir_test_wordlist):
		self.dir_test = dir_test
		self.seg = pkuseg.pkuseg(postag=False)
		self.filelist = os.listdir(dir_test)
		self.stop_words = stop_word_build()
		self.dir_test_wordlist = dir_test_wordlist

	def topic_list(self):
		return os.listdir(self.dir_test)

	def word_divid(self, input_word):
		list_word = self.seg.cut(input_word)
		list_new = [word for word in list_word if word not in self.stop_words]
		return list_new

	def dic_word_list_include_releaser(self):
		f_list = self.filelist
		dic_word_list = {}
		for file in f_list:
			line_list = file_to_list(os.path.join(self.dir_test, file))
			for line in line_list:
				releaser_seg = self.word_divid(line[0])
				title = self.word_divid(line[1])
				line_word = releaser_seg + title
				channel = line[2]
				if channel not in dic_word_list:
					dic_word_list[channel] = [line_word]
				else:
					dic_word_list[channel].append(line_word)
		return dic_word_list

	def topic_covert(self):
		topic = file_to_list(r'F:\TC\topic_convert.csv')
		dic_out = {}
		for topic_line in topic:
			dic_out[topic_line[0]] = topic_line[1]
		return dic_out

	def write_list(self):
		dic_word_list = self.dic_word_list_include_releaser()
		dic_topic_convert = self.topic_covert()
		for key, value in dic_word_list.items():
			topic = dic_topic_convert[key]
			ld_to_csv(value, self.dir_test_wordlist, topic)

	def list_to_word_dic(self, input_list):
		dic_out = {}
		for word in input_list:
			if word not in dic_out:
				dic_out[word] = input_list.count(word)
		return dic_out

	def test_dic_build(self):
		test_dic_path = os.listdir(self.dir_test_wordlist)
		dic_test_out = {}
		for file in test_dic_path:
			topic = file.split('.csv')[0]
			path_topic = os.path.join(self.dir_test_wordlist, file)
			file = file_to_list(path_topic)
			list_this_topic = []
			for line in file:
				dic_out = self.list_to_word_dic(line)
				list_this_topic.append(dic_out)
			dic_test_out[topic] = list_this_topic
		return dic_test_out

class title_parse():  # 处理标题,发布者,和channel

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
			'9': '亲子',
			'10': '宠物',
			'11': '旅游',
			'12': '财经',
			'13':' 军事',
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

	def vector_build(self,title_input,releaser_input=''):
		if releaser_input:
			line_words = self.word_divid(releaser_input) + self.word_divid(title_input)
		else:
			line_words = self.word_divid(title_input)
		vector_dic = {}
		for word_uni in set(line_words):
			vector_dic[word_uni] = line_words.count(word_uni)
		return vector_dic

	def channel_judge(self,input_channel):
		if input_channel in self.topic_covert:
			return self.topic_covert[input_channel]
		else:
			return None

	def parse_title_releaser_channel(self, input_title, inpurt_releaser='', input_channel=''):
		if self.channel_judge(input_channel):
			return self.topic_dic[self.channel_judge(input_channel)]
		else:
			return self.vector_build(input_title,inpurt_releaser)

if __name__ == '__main__':
	# 分类范畴
	topic_sort = list(range(1, 16))
	# 训练集目录
	train_dir = r"F:\TC\tfidf_retrain"
	# 测试集数据
	test_dic = test_vector_build(r'F:\TC\test_file', r'F:\TC\test_word_list')

	test_dic.write_list()

	test_dic_omi = test_dic.test_dic_build()

	topic_good = list(range(1, 16))
	# 分类器
	tc = Title_classifier(train_dir, topic_good)



#tc.calculate_best_num_topic(test_dic_omi['3'][16], 2)

# 此行之下皆为测试

def get_precison(test_dic_input):
	for key in test_dic_input:
		key_list = []
		key_distribution = {}
		key_right_num = 0
		this_key_list = test_dic_input[key]
		for line in this_key_list:
			list_topic_cos = tc.calculate_best_num_topic(line, 1)
			predict_title = list_topic_cos[0][0].split('_')[0]
			if predict_title not in key_distribution:
				key_distribution[predict_title] = 1
			else:
				key_distribution[predict_title] += 1

			if predict_title == key:
				key_right_num += 1
		precison = key_right_num / len(this_key_list)
		print(key,precison,dic_order_by_value(key_distribution))

def get_precison_cos(test_dic_input):
	for key in test_dic_input:
		key_list = []
		key_distribution = {}
		key_right_num = 0
		key_totol_num = 0
		this_key_list = test_dic_input[key]
		for line in this_key_list:
			list_topic_cos = tc.calculate_best_num_topic(line, 1)
			predict_title = list_topic_cos[0][0].split('_')[0]
			cos_simimarity = list_topic_cos[1][0]
			if cos_simimarity >= 0.05:
				key_totol_num += 1
				if predict_title not in key_distribution:
					key_distribution[predict_title] = 1
				else:
					key_distribution[predict_title] += 1

				if predict_title == key:
					key_right_num += 1
		precison = key_right_num / key_totol_num
		print(key,precison,dic_order_by_value(key_distribution))

get_precison(test_dic_omi)

get_precison_cos(test_dic_omi)

# 这里对输出topic做去除_TFIDF处理
# 测试对象准确