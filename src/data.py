# Takes a copy of the model from the disk and instantiates it
def load_pretrained_model(model_dir):
	raise NotImplemented

# loads the already processed and vectorized data from the disk
def load_processed_data(data_dir):
	raise NotImplemented

# loads the text comments, processes them, and saves them to the disk
def load_raw_data(data_dir):
	raise NotImplemented

# loads the mappings from words to GloVe embeddings from the disk
# Downloads the embeddings if they are not already there
def load_embeddings(data_dir):
	raise NotImplemented

# Converts each user comment into its corresponding list of vectors, and corresponding class
def vectorize_data(raw_data):
	raise NotImplemented

# Normalizes the data to have a mean of 0 and a standard deviation of 1
# Might also want to correct for any class disparity here, else we can do it in the training routine
def normalize_data(vectorized_data):
	raise NotImplemented