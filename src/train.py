import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
import tensorflow as tf
from tensorflow import optimizers
import model
import data
import numpy as np

def average_padded_sequence(seq):
	# print(seq)
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
		# print(item[0])
		if item[0] == 0:
			last = max(1, ind - 1)
			break
	else:
		last = len(seq)

	unpadded = np.array(seq[:last])
	# print(unpadded.shape)
	return np.clip(np.var(unpadded, axis = 0), 0, 10)

def generate_features(users):
	comment_averaged = []
	for user in users:
		comment_averaged.append([])
		for comment in user:
			comment_averaged[-1].append(average_padded_sequence(comment))

	# print([variance_padded_sequence(x) for x in comment_averaged])
	features = {
		'mean' : tf.convert_to_tensor([average_padded_sequence(x) for x in comment_averaged], dtype=tf.float32),
		'variance' : tf.convert_to_tensor([variance_padded_sequence(x) for x in comment_averaged], dtype=tf.float32)
	}
	# print(features['variance'].shape)
	# print(features['variance'])

	return features

def test_accuracy(classifier, embeddings, data_corpus):
	true_positive = 0
	false_positive = 0
	true_negative = 0
	false_negative = 0
	losses = []
	for batch_ind in range(data_corpus.test_batch_count(32)):
		batch_users = data_corpus.get_test_batch(batch_ind, 32)
		inputs = [data.vectorize_user(x, embeddings) for x in batch_users]
		truth = [[int(x['on_bipolar']) * 2 - 1] for x in batch_users]
		features = generate_features(inputs)

		outputs, loss = classifier(features, truth = truth)
		losses.append(loss)
		
		for t, o in zip(truth, outputs):
			# print(t,o)
			if t[0] == 1:
				if o > 0:
					true_positive += 1
				else:
					false_negative += 1
			else:
				if o > 0:
					false_positive += 1
				else:
					true_negative += 1

	loss = np.sum(losses) / len(losses)
	print("Test loss: " + str(tf.reduce_mean(loss).numpy()))
	print(true_positive, true_negative, false_positive, false_negative)
	with open('test_loss.txt', 'a') as f:
		f.write(str(float(tf.reduce_mean(loss).numpy())))
		f.write('\n')
	with open('test_acc.txt', 'a') as f:
		f.write(','.join(str(x) for x in [true_positive, true_negative, false_positive, false_negative]))
		f.write('\n')

def train(classifier, data_corpus, num_epochs, batch_size):
	optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003)
	embeddings = data.load_embeddings(embeddings_file_path, 50)
	for epoch in range(num_epochs):
		data_corpus.shuffle()
		for batch_ind in range(data_corpus.batch_count(batch_size)):
			# print(batch_ind, epoch)
			batch_users = data_corpus.get_batch(batch_ind, batch_size)

			# print("About to vectorize")
			inputs = [data.vectorize_user(x, embeddings) for x in batch_users]
			truth = [[int(x['on_bipolar']) * 2 - 1] for x in batch_users]
			features = generate_features(inputs)
			# print("Features generated")

			with tf.GradientTape() as tape:
				outputs, loss = classifier(features, truth = truth)
				# print(outputs.numpy(), tf.reduce_mean(loss))
				print(batch_ind, epoch, tf.reduce_mean(loss).numpy())
				# print(classifier.trainable_variables)
				grads = tape.gradient(loss, classifier.trainable_variables)
				optimizer.apply_gradients(zip(grads, classifier.trainable_variables))
				with open('train_loss.txt', 'a') as f:
					f.write(str(float(tf.reduce_mean(loss).numpy())))
					f.write('\n')
			if batch_ind % 10 == 0:
				test_accuracy(classifier, embeddings, data_corpus)

if __name__ == '__main__':
	root_dir = os.path.normpath(r"C:/repos/DeepVarianceNetwork")
	# Read arguments or use defaults
	device_name = tf.test.gpu_device_name()
	print(f"GPU found: {device_name == '/device:GPU:0'}")
	print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

	data_file_path = os.path.join(root_dir, "data/reddit_bipolar.csv")
	embeddings_file_path = os.path.join(root_dir,"data/glove.6B.50d.txt")
	all_redditors = data.load_raw_data(data_file_path)
	# embeddings = data.load_embeddings(embeddings_file_path,50)
	# all_redditors = data.vectorize_data(all_redditors, embeddings)

	data_corpus = data.DataCorpus()
	for user_id in all_redditors:
		if all_redditors[user_id]['on_bipolar'] == '1':
			data_corpus.positive_samples.append(all_redditors[user_id])
		else:
			data_corpus.negative_samples.append(all_redditors[user_id])
	print(len(data_corpus.positive_samples))
	print(len(data_corpus.negative_samples))
	data_corpus.seperate_test(0.10)

	# Load existing model if it exists and "Restart" parameter is false
	# Otherwise initialize a new model
	classifier = model.BipolarClassifier(embedding_size = 50, layer_sizes = [500, 400, 300, 200, 100], signals = ['mean', 'variance'])
	with open('train_loss.txt', 'w') as f:
		pass
	with open('test_loss.txt', 'w') as f:
		pass
	with open('test_acc.txt', 'w') as f:
		pass
	# Train the models
	train(classifier, data_corpus, 9999, 32)

	# Save the model