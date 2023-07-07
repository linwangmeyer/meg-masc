import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.models import TfidfModel
import gensim
from sklearn.cluster import KMeans
from collections import Counter
from wordcloud import WordCloud
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


stop_words = stopwords.words('english')
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if not word.isdigit()]
    tokens = [re.sub(r'[^\x00-\x7F]+', '', word) for word in tokens]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    processed_text = ' '.join(tokens)
    return processed_text

#####################################################
## read data
fname = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/LDA/ALL_EXPERIMENTAL_STIMULI/InfoTopic_stim_toyData_gpt4_output.csv'
df = pd.read_csv(fname)
df_sel = df.filter(like='gpt4')
for column in df_sel.columns:
    df_sel[column] = df_sel[column].apply(preprocess_text)

## pre-process data
item1 = df_sel.iloc[1].to_list()
processed_item = []
for sentence in item1:
    sentence_list = sentence.split()
    processed_item.append(sentence_list)

## vectorize data
dictionary = corpora.Dictionary(processed_item)
corpus = [dictionary.doc2bow(text) for text in processed_item]
tfidf_model = TfidfModel(corpus)
tfidf_corpus = tfidf_model[corpus]
matrix = gensim.matutils.corpus2dense(tfidf_corpus, len(dictionary)) # words x documents sparse matrix

## Elbow: identify optional cluster
wcss = []
k_values = range(1,4)
for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(matrix.T)
    wcss.append(kmeans.inertia_)  # Inertia is the WCSS value

# Plot the WCSS values
plt.plot(k_values, wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.savefig('kmeans-elbow.png')
plt.show()

## Run k-means
num_clusters = 3  # Specify the number of identified K clusters
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(matrix.T)
cluster_labels = kmeans.labels_

df_label = pd.Series(cluster_labels, name='cluster', index=['gpt4_p1','gpt4_p2','gpt4_p3','gpt4_p4'])
df_para = pd.Series(item1, name = 'text', index=['gpt4_p1','gpt4_p2','gpt4_p3','gpt4_p4'])
df_kmean = pd.concat([df_para,df_label], axis=1)
df_kmean.to_csv(f'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/LDA/ALL_EXPERIMENTAL_STIMULI/kmeans.cvs',index=False)


## Plot to visualize content of the cluster
cluster_labels = df_kmean['cluster'].unique()
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
axes = axes.flatten()

for i, cluster_label in enumerate(cluster_labels):
    df_text = df_kmean[df_kmean['cluster'] == cluster_label]['text']
    combined_text = ' '.join(df_text.astype(str))
    word_frequencies = Counter(combined_text.split())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_frequencies)
    ax = axes[i]
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(f'Cluster {cluster_label}')
    ax.axis('off')
if len(cluster_labels) < len(axes):
    fig.delaxes(axes[len(cluster_labels)])

plt.tight_layout()
plt.savefig('/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/LDA/plots/kmeans-wordcloud.png')
plt.show()