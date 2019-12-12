import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
import tensorflow as tf
from tensorflow import optimizers
import model
import data
import numpy as np

def average_padded_sequence(seq):
	count = 0
	total = None
	for item in seq:
		if item[0] == 0:
			continue
		if total is None:
			total = item
			count = 1
		else:
			total += item
			count += 1
	if count == 0:
		return seq[0]
	return total / count

def variance_padded_sequence(seq):
	for ind, item in enumerate(seq):
		if item[0] == 0:
			last = max(1, ind - 1)
			break
	else:
		last = len(seq)

	unpadded = np.array(seq[:last])
	return np.clip(np.var(unpadded, axis = 0), 0, 10)

def generate_features(users):
	comment_averaged = []
	for user in users:
		comment_averaged.append([])
		for comment in user:
			comment_averaged[-1].append(average_padded_sequence(comment))

	features = {
		'mean' : tf.convert_to_tensor([average_padded_sequence(x) for x in comment_averaged], dtype=tf.float32),
		'variance' : tf.convert_to_tensor([variance_padded_sequence(x) for x in comment_averaged], dtype=tf.float32)
	}

	return features

def user_truth(user):
	truth = np.array([0, 0, 0, 0], dtype = np.float32)
	if user['on_bipolar'] == '1':
		truth[1] = 1
	if user['on_bpd'] == '1':
		truth[2] = 1
	if np.sum(truth) == 0:
		truth[0] = 1
	elif np.sum(truth) == 2:
		truth = np.array([0, .1, .1, .8], dtype = np.float32)

	return truth / np.sum(truth)

def test_accuracy(classifier, embeddings, data_corpus):
	confusion_matrix = np.zeros([classifier.num_classes, classifier.num_classes], dtype = np.int32)
	losses = []
	batch_size = 128
	for batch_ind in range(data_corpus.test_batch_count(batch_size)):
		batch_users = data_corpus.get_test_batch(batch_ind, batch_size)
		inputs = [data.vectorize_user(x, embeddings) for x in batch_users]
		truth = [user_truth(x) for x in batch_users]
		features = generate_features(inputs)

		outputs, loss = classifier(features, truth = truth)
		losses.append(loss)
		
		for t, o in zip(truth, outputs):
			confusion_matrix[np.argmax(t), np.argmax(o)] += 1

	loss = np.sum(losses) / len(losses)
	print("Test loss: " + str(tf.reduce_mean(loss).numpy()))
	print(confusion_matrix)
	with open('test_loss.txt', 'a') as f:
		f.write(str(float(tf.reduce_mean(loss).numpy())))
		f.write('\n')
	with open('test_acc.txt', 'a') as f:
		f.write(str(confusion_matrix))
		f.write('\n')

def train(classifier, data_corpus, num_batches, batch_size):
	optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003)
	embeddings = data.load_embeddings(embeddings_file_path, 50)
	for batch_ind in range(1, num_batches):
		# print(batch_ind, epoch)
		batch_users = data_corpus.get_batch(batch_size)

		inputs = [data.vectorize_user(x, embeddings) for x in batch_users]
		truth = [user_truth(x) for x in batch_users]
		features = generate_features(inputs)

		with tf.GradientTape() as tape:
			outputs, loss = classifier(features, truth = truth)
			print(batch_ind, tf.reduce_mean(loss).numpy())
			grads = tape.gradient(loss, classifier.trainable_variables)
			optimizer.apply_gradients(zip(grads, classifier.trainable_variables))
			with open('train_loss.txt', 'a') as f:
				f.write(str(float(tf.reduce_mean(loss).numpy())))
				f.write('\n')
		if batch_ind % 10 == 0:
			test_accuracy(classifier, embeddings, data_corpus)

if __name__ == '__main__':
	root_dir = os.path.normpath(r"C:/repos/DeepVarianceNetwork")
	device_name = tf.test.gpu_device_name()
	print(f"GPU found: {device_name == '/device:GPU:0'}")
	print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

	data_file_path = os.path.join(root_dir, "data/reddit_data.csv")
	all_redditors = data.load_raw_data(data_file_path)
	embeddings_file_path = os.path.join(root_dir,"data/glove.6B.50d.txt")

	data_corpus = data.DataCorpus()
	for user_id in all_redditors:
		data_corpus.add_sample(np.argmax(user_truth(all_redditors[user_id])), all_redditors[user_id])
	for x in data_corpus.classes:
		print(len(data_corpus.classes[x]))

	data_corpus.seperate_test(0.10)

	classifier = model.MentalHealthClassifier(embedding_size = 50, layer_sizes = [500, 400, 300, 200, 100], signals = ['mean'], num_classes = 4)
	# classifier = model.SoftmaxRegression(embedding_size = 50, signals = ['mean', 'variance'], num_classes = 4)
	
	with open('train_loss.txt', 'w') as f:
		pass
	with open('test_loss.txt', 'w') as f:
		pass
	with open('test_acc.txt', 'w') as f:
		pass

	train(classifier, data_corpus, 9999, 32)