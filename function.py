import numpy as np
import os
import json

def pos_embed(x,maxlen):
	return max(0, min(x + maxlen, maxlen + maxlen + 1))

def find_index(x,y):
	for index, item in enumerate(y):
		if x == item:
			return index
	return -1

def init_word(data_path,export_path,word2id, word_siz,word_vec):
	# reading word embedding data...
	#global word2id, word_size
	res = []
	ff = open(export_path + "/entity2id.txt", "w")
	f = open(data_path + "kg/train.txt", "r")
	while True:
		content = f.readline()
		if content == "":
			break
		h, t, r = content.strip().split("\t")
		if not h in word2id:
			word2id[h] = len(word2id)
			ff.write("%s\t%d\n"%(h, word2id[h]))
		if not t in word2id:
			word2id[t] = len(word2id)
			ff.write("%s\t%d\n"%(t, word2id[t]))
	f.close()
	f = open(data_path + "text/train.txt", "r")
	while True:
		content = f.readline()
		if content == "":
			break
		h,t = content.strip().split("\t")[:2]
		if not h in word2id:
			word2id[h] = len(word2id)
			ff.write("%s\t%d\n"%(h, word2id[h]))
		if not t in word2id:
			word2id[t] = len(word2id)
			ff.write("%s\t%d\n"%(t, word2id[t]))
	f.close()
	f = open(data_path + "text/test.txt", "r")
	while True:
		content = f.readline()
		if content == "":
			break
		h,t = content.strip().split("\t")[:2]
		if not h in word2id:
			word2id[h] = len(word2id)
			ff.write("%s\t%d\n"%(h, word2id[h]))
		if not t in word2id:
			word2id[t] = len(word2id)
			ff.write("%s\t%d\n"%(t, word2id[t]))
	f.close()
	res.append(len(word2id))
	ff.close()

	print 'reading word embedding data...'
	f = open(data_path + 'text/vec.txt', "r")
	total, size = f.readline().strip().split()[:2]
	total = (int)(total)
	word_size = (int)(size)
	vec = np.ones((total + res[0], word_size), dtype = np.float32)
	for i in range(total):
		content = f.readline().strip().split()
		word2id[content[0]] = len(word2id)
		for j in range(word_size):
			vec[i + res[0]][j] = (float)(content[j+1])
	f.close()
	word2id['UNK'] = len(word2id)
	word2id['BLANK'] = len(word2id)
	#global word_vec
	word_vec = vec
	res.append(len(word2id))
	res.append(word_vec)
	return res

def init_relation(data_path,export_path,relation2id):
	# reading relation ids...
	#global relation2id
	print 'reading relation ids...'	
	res = []
	ff = open(export_path + "/relation2id.txt", "w")

	voc=[]
	with open(data_path + "text/relation2id.txt","r") as f:
		for i in f.readlines()[1:]:
			voc.append([a.strip()  for a in i.split(' ')[::-1]])
	voc=dict(voc)
	import pickle
	with open ('jointE/voc.pkl','wb')as f:
		pickle.dump(voc,f)
	del voc 
	f = open(data_path + "text/relation2id.txt","r")
	total = (int)(f.readline().strip())
	for i in range(total):
		content = f.readline().strip().split()
		if not content[0] in relation2id:
			relation2id[content[0]] = len(relation2id)
			ff.write("%s\t%d\n"%(content[0], relation2id[content[0]]))
	f.close()


	res.append(len(relation2id))
	f = open(data_path + "kg/train.txt", "r")
	for i in f.readlines():
		h, t, r = i.strip().split("\t")
		if not r in relation2id:
			relation2id[r] = len(relation2id)
			ff.write("%s\t%d\n"%(r, relation2id[r]))
	f.close()
	ff.close()
	res.append(len(relation2id))
	return res

def sort_files(data_path,name,relation2id, limit):
	hash = {}
	f = open(data_path + "text/" + name + '.txt','r')
	s = 0
	while True:
		content = f.readline()
		if content == '':
			break
		s = s + 1
		origin_data = content
		content = content.strip().split()
		en1_id = content[0]
		en2_id = content[1]
		rel_name = content[4]
		if (rel_name in relation2id) and ((int)(relation2id[rel_name]) < limit[0]):
			relation = relation2id[rel_name]
		else:
			relation = relation2id['NA']
		id1 = str(en1_id)+"#"+str(en2_id)
		id2 = str(relation)
		if not id1 in hash:
			hash[id1] = {}
		if not id2 in hash[id1]:
			hash[id1][id2] = []
		hash[id1][id2].append(origin_data)
	f.close()
	f = open(data_path + name + "_sort.txt", "w")
	f.write("%d\n"%(s))
	for i in hash:
		for j in hash[i]:
			for k in hash[i][j]:
				f.write(k)
	f.close()

def init_train_files(data_path,name, limit,maxlen,fixlen,relation2id,word2id):
	print 'reading ' + name +' data...'
	f = open(data_path + name + '.txt','r')
	total = (int)(f.readline().strip())
	sen_word = np.zeros((total, fixlen), dtype = np.int32)
	sen_pos1 = np.zeros((total, fixlen), dtype = np.int32)
	sen_pos2 = np.zeros((total, fixlen), dtype = np.int32)
	sen_mask = np.zeros((total, fixlen), dtype = np.int32)
	sen_len = np.zeros((total), dtype = np.int32)
	sen_label = np.zeros((total), dtype = np.int32)
	sen_head = np.zeros((total), dtype = np.int32)
	sen_tail = np.zeros((total), dtype = np.int32)
	instance_scope = []
	instance_triple = []

	tex=[]
	for s in range(total):
		content = f.readline().strip().split()
		sentence = content[5:-1]
		en1_id = content[0]
		en2_id = content[1]
		en1_name = content[2]
		en2_name = content[3]
		rel_name = content[4]
		tex.append([en1_name,en2_name,' '.join(sentence),rel_name])
		if rel_name in relation2id and ((int)(relation2id[rel_name]) < limit[0]):
			relation = relation2id[rel_name]
		else:
			relation = relation2id['NA']
		en1pos = 0
		en2pos = 0
		for i in range(len(sentence)):
			if sentence[i] == en1_name:
				sentence[i] = en1_id
				en1pos = i
				sen_head[s] = word2id[en1_id]
			if sentence[i] == en2_name:
				sentence[i] = en2_id
				en2pos = i
				sen_tail[s] = word2id[en2_id]
		en_first = min(en1pos,en2pos)
		en_second = en1pos + en2pos - en_first
		for i in range(fixlen):
			sen_word[s][i] = word2id['BLANK']
			sen_pos1[s][i] = pos_embed(i - en1pos,maxlen)
			sen_pos2[s][i] = pos_embed(i - en2pos,maxlen)
			if i >= len(sentence):
				sen_mask[s][i] = 0
			elif i - en_first<=0:
				sen_mask[s][i] = 1
			elif i - en_second<=0:
				sen_mask[s][i] = 2
			else:
				sen_mask[s][i] = 3
		for i, word in enumerate(sentence):
			if i >= fixlen:
				break
			elif not word in word2id:
				sen_word[s][i] = word2id['UNK']
			else:
				sen_word[s][i] = word2id[word]
		sen_len[s] = min(fixlen, len(sentence))
		sen_label[s] = relation
		#put the same entity pair sentences into a dict
		tup = (en1_id,en2_id,relation)
		if instance_triple == [] or instance_triple[len(instance_triple) - 1] != tup:
			instance_triple.append(tup)
			instance_scope.append([s,s])
		instance_scope[len(instance_triple) - 1][1] = s
		if (s+1) % 100 == 0:
			print s
	import pandas as pd 
	pd.DataFrame(tex).drop_duplicates([0,1]).to_csv('jointE/test.csv')
	return np.array(instance_triple), np.array(instance_scope), sen_len, sen_label, sen_word, sen_pos1, sen_pos2, sen_mask, sen_head, sen_tail

def init_kg(data_path,export_path,word2id,relation2id):
	ff = open(export_path + "/triple2id.txt", "w")
	f = open(data_path + "kg/train.txt", "r")
	content = f.readlines()
	ff.write("%d\n"%(len(content)))
	for i in content:
		h,t,r = i.strip().split("\t")
		ff.write("%d\t%d\t%d\n"%(word2id[h], word2id[t], relation2id[r]))
	f.close()
	ff.close()

	f = open(export_path + "/entity2id.txt", "r")
	content = f.readlines()
	f.close()
	f = open(export_path + "/entity2id.txt", "w")
	f.write("%d\n"%(len(content)))
	for i in content:
		f.write(i.strip()+"\n")
	f.close()

	f = open(export_path + "/relation2id.txt", "r")
	content = f.readlines()
	f.close()
	f = open(export_path + "/relation2id.txt", "w")
	f.write("%d\n"%(len(content)))
	for i in content:
		f.write(i.strip()+"\n")
	f.close()

def make_shape(array,last_dim):
	output = []
	for i in array:
		for j in i:
			output.append(j)
	output = np.array(output)
	if np.shape(output)[-1]==last_dim:
		return output

	else:
		print 'Make Shape Error!'