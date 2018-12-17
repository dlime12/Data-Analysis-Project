import pandas as pd



# Text Lemmatisation
import string
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

# Token Vecotriser
from sklearn.feature_extraction.text import TfidfVectorizer

# Clustering
from sklearn.cluster import KMeans

# Optimal K finding and visualisation
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Zip f law
from sklearn.feature_extraction.text import CountVectorizer

# Zip f visualisation
from scipy.spatial.distance import euclidean
from math import sqrt

# Time
import time

# Singular Value Decomposition (Matrix dimensionality reduction)
from sklearn.decomposition import TruncatedSVD


"""
Task Definition:
BBC, start an "online personalised news story service"
Determine "clusters" of stories based on similar topics
"""



# load the dataset
df = pd.read_json('dataset/bbc.json')
# random state
rs = 42


def explore_dataset():
	# as usual, explore the dataset
	print("\n", df.info())

	# print out the first 200 characters of the first row of text column
	print("\n", df.get_value(index=0, col='text')[:200])

	# average length of text column
	print("\n", df['text'].apply(lambda x: len(x)).mean())



# Data preprocessing

# initialise WordNet lemmatizer and punctuation filter
lemmatizer = WordNetLemmatizer()
punct = set(string.punctuation)

# load the provided stopwords
df_stop = pd.read_json('dataset/bbc.json')

# join provided stopwords with the default NLTK English stopwords
stopwords = set(sw.words('english'))

def lemmatize(token, tag):
	tag = {
		'N': wn.NOUN,
		'V': wn.VERB,
		'R': wn.ADV,
		'J': wn.ADJ
	}.get(tag[0], wn.NOUN)

	return lemmatizer.lemmatize(token, tag)


# Pre processing, take in a document string, splits then preprocess it
def cab_tokenizer(document):
	# initialize token list
	tokens = []

	# split the document into sentences
	for sent in sent_tokenize(document):
		# split the document into tokens and then create part of speech tag for each token
		for token, tag in pos_tag(wordpunct_tokenize(sent)):
			# preprocess and remove unnecessary characters
			token = token.lower()
			token = token.strip()
			token = token.strip('_')
			token = token.strip('*')

			# If stopword, ignore token and continue
			if token in stopwords:
				continue

			# If punctuation, ignore token and continue
			if all(char in punct for char in token):
				continue
			# Lemmatize the token and add back to the tokens list
			lemma = lemmatize(token, tag)
			tokens.append(lemma)

	return tokens



# Once we have the tokens, we need to vectorise into matrices
# 'ngram_range' of (1,2) to produce both unigram and bigram (two word) tokens
# Bigram allows us to cpature information from two consecutive words (phrases)

# tf idf vectoriser
tfidf_vec = TfidfVectorizer(tokenizer=cab_tokenizer, ngram_range=(1,2))
X = tfidf_vec.fit_transform(df['text'])

# see the number of unique tokens produced by the vectorizer.
print(len(tfidf_vec.get_feature_names()))






# Document analysis Task 2

# K means clustering using the term vector
kmeans = KMeans(n_clusters=8, random_state=rs).fit(X)



# Analyse each cluster's topic
# function to visualise text cluster. Useful for the assignment too :)
def visualise_text_cluster(n_clusters, cluster_centers, terms, num_word = 5):
	# -- Params --
	# cluster_centers: cluster centers of fitted/trained KMeans/other centroid-based clustering
	# terms: terms used for clustering
	# num_word: number of terms to show per cluster. Change as you please.

	# find features/terms closest to centroids
	ordered_centroids = cluster_centers.argsort()[:, ::-1]

	for cluster in range(n_clusters):
		print("Top terms for cluster {}:".format(cluster), end=" ")
		for term_idx in ordered_centroids[cluster, :5]:
			print(terms[term_idx], end=', ')
		print()

# call it - prints each clusters with terms closest to its centroid
visualise_text_cluster(kmeans.n_clusters, kmeans.cluster_centers_, tfidf_vec.get_feature_names())



# Determine the optimal K, Didn't work out to well
def optimalK_elbow():
	# list to save the clusters and cost
	clusters = []
	inertia_vals = []

	# this whole process should take a while
	for k in range(1, 25, 1):
		# train clustering with the specified K
		model = KMeans(n_clusters=k, random_state=rs, n_jobs=10)
		model.fit(X)

		# append model to cluster list
		clusters.append(model)
		inertia_vals.append(model.inertia_)

	# plot the inertia vs K values
	plt.plot(range(1,25,1), inertia_vals, marker='*')
	plt.show()

	for k in range(1, 24, 1):
		#print("\n", clusters[k], "\n")
		print("Silhouette score for k=", k+1, silhouette_score(X, clusters[k].predict(X)))


 ## Uncomment for testing
#if __name__ == '__main__':
	#optimalK_elbow()







# Feature Selection and Transformation

# creating tf-idf terms - a bit slow, do it occasionaly
def calculate_tf_idf_terms(document_col):
# Param - document_col: collection of raw document text that you want to analyse

	# use count vectorizer to find TF and DF of each term
	count_vec = CountVectorizer(tokenizer=cab_tokenizer, ngram_range=(1,2))
	X_count = count_vec.fit_transform(df['text'])

	# create list of terms and their tf and df
	terms = [{'term': t, 'idx': count_vec.vocabulary_[t],
	'tf': X_count[:, count_vec.vocabulary_[t]].sum(),
	'df': X_count[:, count_vec.vocabulary_[t]].count_nonzero()}
	for t in count_vec.vocabulary_]

	return terms

terms = calculate_tf_idf_terms(df['text'])





# visualisation of ZIPF law
def visualise_zipf(terms, itr_step = 40):
	# --- Param ---
	# terms: collection of terms dictionary from calculate_tf_idf_terms function
	# itr_step: used to control how many terms that you want to plot. Num of terms to plot = N terms / itr_step

	# sort terms by its frequency
	terms.sort(key=lambda x: (x['tf'], x['df']), reverse=True)


	# select a few of the terms for plotting purpose
	sel_terms = [terms[i] for i in range(0, len(terms), itr_step)]
	labels = [term['term'] for term in sel_terms]

	# plot term frequency ranking vs its DF
	plt.plot(range(len(sel_terms)), [x['df'] for x in sel_terms])
	plt.xlabel('Term frequency ranking')
	plt.ylabel('Document frequency')

	max_x = len(sel_terms)
	max_y = max([x['df'] for x in sel_terms])

	# annotate the points
	prev_x, prev_y = 0, 0
	for label, x, y in zip(labels,range(len(sel_terms)), [x['df'] for x in sel_terms]):
		# calculate the relative distance between labels to increase visibility
		x_dist = (abs(x - prev_x) / float(max_x)) ** 2
		y_dist = (abs(y - prev_y) / float(max_y)) ** 2
		scaled_dist = sqrt(x_dist + y_dist)

		if (scaled_dist > 0.1):
			plt.text(x+2, y+2, label, {'ha': 'left', 'va': 'bottom'}, rotation=30)
			prev_x, prev_y = x, y

	plt.show()

visualise_zipf(terms)




# another tf idf vectoriser
# limit the terms produced to terms that occured in min of 2 documents and max 80% of all documents
filter_vec = TfidfVectorizer(tokenizer=cab_tokenizer, ngram_range=(1,2), min_df=2, max_df=0.8)
X_filter = filter_vec.fit_transform(df['text'])

# see the number of unique tokens produced by the vectorizer. Reduced!
print("\n", len(filter_vec.get_feature_names()),"\n")



start = time.time()
# K means clustering using the new term vector, time it for comparison to SVD
kmeans_fil = KMeans(n_clusters=8, random_state=rs).fit(X_filter)
end = time.time()
print((end - start)*10, "seconds")


# visualisation
visualise_text_cluster(kmeans_fil.n_clusters, kmeans_fil.cluster_centers_, filter_vec.get_feature_names())



svd = TruncatedSVD(n_components=100, random_state=42)
X_trans = svd.fit_transform(X_filter)




# sort the components by largest weighted word
sorted_comp = svd.components_.argsort()[:, ::-1]
terms = filter_vec.get_feature_names()

print("\n")

# visualise word - concept/component relationships
for comp_num in range(8):
	print("Top terms in component #{}".format(comp_num), end=" ")
	for i in sorted_comp[comp_num, :5]:
		print(terms[i], end=", ")
	print()



print("\n")
start = time.time()

# K-means clustering using LSA-transformed X
svd_kmeans = KMeans(n_clusters=8, random_state=rs).fit(X_trans)

end = time.time()
print((end - start)*10, "seconds")


# transform cluster centers back to original feature space for visualisation
original_space_centroids = svd.inverse_transform(svd_kmeans.cluster_centers_)

# visualisation
visualise_text_cluster(svd_kmeans.n_clusters, original_space_centroids, filter_vec.get_feature_names())

