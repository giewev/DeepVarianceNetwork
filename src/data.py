import csv
import os
import collections
import numpy as np
from tqdm import tqdm
import random

MAX_COMMENT_LENGTH = 200 #Change this. Ideally 10,000 but taking too much memory
MIN_COMMENT_LENGTH = 10
COMMENTS_PER_USER = 100


# all_redditors{
# 	"user_id":{
# 		'name':"",
# 		'on_bipolar':1/0,
# 		'comment_karma':9999,
# 		'link_karma':9999,
# 		'acct_created_at':'2018-01-01 11:11:11',
# 		"comments":[{
# 			"text":"",
# 			"timestamp":""
# 		},.....
# 		],
# 		"comment_embeddings":[] #shape num_comments x MAX_COMMENT_LENGTH x emb_dim
# }




# Takes a copy of the model from the disk and instantiates it
def load_pretrained_model(model_dir):
	raise NotImplemented

# loads the already processed and vectorized data from the disk
def load_processed_data(data_dir):
	raise NotImplemented

# loads the text comments, processes them, and saves them to the disk
def load_raw_data(data_file_path):

	maxed_users = set()
	all_redditors = {}
	full_redditors = {}
	
	with open(data_file_path,'r',newline='', encoding = 'utf-8') as csv_file:
		csv_reader = csv.reader(csv_file,delimiter=',',skipinitialspace=True,lineterminator = '\n')
		

		for row in csv_reader:
			
			if len(row)!= 9:
				#data was corrupted
				#TODO Arjun : Check this
				continue
			
			if len(row[2]) not in [1,0]:
				#just a check to see if row is valid
				#TODO Arjun:remove this condition
				continue

			user_id = row[1]
			while user_id in maxed_users:
				user_id = user_id + "1"

			if user_id not in all_redditors:
				all_redditors[user_id] = {
					'name':row[0],
					'on_bipolar':row[2],
					'comment_karma':row[6],
					'link_karma':row[7],
					'acct_created_at':row[8],
					"comments":[],
					"comment_embeddings":[]
				}
			
			if len(row[3].split(' ')) < MIN_COMMENT_LENGTH:
				continue

			all_redditors[user_id]['comments'].append({"text":row[3],"timestamp":row[4]})
			if len(all_redditors[user_id]['comments']) >= COMMENTS_PER_USER:
				maxed_users.add(user_id)
				full_redditors[user_id] = all_redditors[user_id]

			# if len(all_redditors) == 50:
			# 	break

	return full_redditors


# loads the mappings from words to GloVe embeddings from the disk
# Downloads the embeddings if they are not already there
def load_embeddings(embeddings_txt_file: str,embedding_dim: int) -> np.ndarray:
    """
    Given a vocabulary (mapping from index to token), this function builds
    an embedding matrix of vocabulary size in which ith row vector is an
    entry from pretrained embeddings (loaded from embeddings_txt_file).
    """
    #tokens_to_keep = set(vocab_id_to_token.values())
    #vocab_size = len(vocab_id_to_token)

    embeddings = {}
    print("\nReading pretrained embedding file.")
    with open(embeddings_txt_file, encoding = 'utf-8') as file:
        for line in tqdm(file):
            line = str(line).strip()
            token = line.split(' ', 1)[0]
            token = token.lower()
            # if not token in tokens_to_keep:
            #     continue
            fields = line.rstrip().split(' ')
            if len(fields) - 1 != embedding_dim:
                raise Exception(f"Pretrained embedding vector and expected "
                                f"embedding_dim do not match for {token}.")
                continue
            vector = np.asarray(fields[1:], dtype='float32')
            embeddings[token] = vector


    #Create for UNK token and Padding Token
    # Estimate mean and std variation in embeddings and initialize it random normally with it
    all_embeddings = np.asarray(list(embeddings.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_std = float(np.std(all_embeddings))

    unk_emb = np.random.normal(embeddings_mean, embeddings_std,(1, embedding_dim))
    pad_emb = np.zeros(shape=(1, embedding_dim))

    embeddings['unk_token'] = unk_emb
    embeddings['pad_token'] = pad_emb

    return embeddings

# Converts each user comment into its corresponding list of vectors, and corresponding class
def vectorize_data( all_redditors,all_embeddings):
	for user_id in all_redditors:
		redditor_obj = all_redditors[user_id]
		redditor_obj['comment_embeddings'] = np.zeros(shape=(len(redditor_obj['comments']),MAX_COMMENT_LENGTH,50))
		for ci in range(len( redditor_obj['comments'])):
			comment_obj = redditor_obj['comments'][ci]
			comment_text = comment_obj['text']
			words_list = comment_text.strip().split(' ')
			embedding_matrix = np.zeros(shape=(MAX_COMMENT_LENGTH,50))
			for wi in range(MAX_COMMENT_LENGTH):
				if wi >= len(words_list):
					word_emb = all_embeddings['pad_token']
				else:
					word = words_list[wi].lower()
					if word != '':
						if word in all_redditors:
							word_emb = all_embeddings[word]
						else:
							word_emb = all_embeddings['unk_token']
					else:
						#print("empty word encountered.Fix this")
						#TODO Arjun:
						pass

				embedding_matrix[wi] = word_emb
			
			redditor_obj['comment_embeddings'][ci] = embedding_matrix

	
	return all_redditors

def alpha_only(word):
	return ''.join(x for x in word if x in "qwertyuiopasdfghjklzxcvbnm")

def vectorize_user(user, all_embeddings):
	vectors = np.zeros(shape=(len(user['comments']),MAX_COMMENT_LENGTH,50))
	for ci in range(len( user['comments'])):
		comment_obj = user['comments'][ci]
		comment_text = comment_obj['text']
		words_list = comment_text.strip().split(' ')
		embedding_matrix = np.zeros(shape=(MAX_COMMENT_LENGTH,50))
		for wi in range(MAX_COMMENT_LENGTH):
			if wi >= len(words_list):
				word_emb = all_embeddings['pad_token']
			else:
				word = words_list[wi].lower()
				word = alpha_only(word)
				# if word != '':
				# 	if word in all_embeddings:
				# 		word_emb = all_embeddings[word]
				# 	else:
				# 		word_emb = all_embeddings['unk_token']
				# else:
				# 	#print("empty word encountered.Fix this")
				# 	#TODO Arjun:
				# 	continue
				# 	pass
				if word in all_embeddings:
					word_emb = all_embeddings[word]
				else:
					word_emb = all_embeddings['unk_token']

			embedding_matrix[wi] = word_emb
		
		vectors[ci] = embedding_matrix
	return vectors


# Normalizes the data to have a mean of 0 and a standard deviation of 1
# Might also want to correct for any class disparity here, else we can do it in the training routine
def normalize_data(vectorized_data):
	raise NotImplemented


class DataCorpus(object):
	def __init__(self):
		self.positive_samples = []
		self.negative_samples = []

	def seperate_test(self, ratio):
		random.shuffle(self.positive_samples)
		random.shuffle(self.negative_samples)
		self.positive_test = self.positive_samples[: int(len(self.positive_samples) * ratio)]
		self.positive_training = self.positive_samples[int(len(self.positive_samples) * ratio) :]
		self.negative_test = self.negative_samples[: int(len(self.negative_samples) * ratio)]
		self.negative_training = self.negative_samples[int(len(self.negative_samples) * ratio) :]

	def shuffle(self):
		random.shuffle(self.positive_test)
		random.shuffle(self.positive_training)
		random.shuffle(self.negative_test)
		random.shuffle(self.negative_training)

	def batch_count(self, batch_size):
		return int(min(len(self.positive_training), len(self.negative_training)) / batch_size)

	def test_batch_count(self, batch_size):
		return int(min(len(self.positive_test), len(self.negative_test)) / batch_size)

	def get_batch(self, batch_index, batch_size):
		batch_start = batch_index * batch_size
		positive = self.positive_training[batch_start : batch_start + batch_size]
		negative = self.negative_training[batch_start : batch_start + batch_size]

		return positive + negative

	def get_test(self):
		return self.positive_test + self.negative_test

	def get_test_batch(self, batch_index, batch_size):
		batch_start = batch_index * batch_size
		positive = self.positive_test[batch_start : batch_start + batch_size]
		negative = self.negative_test[batch_start : batch_start + batch_size]

		return positive + negative