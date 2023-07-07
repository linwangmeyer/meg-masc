'''Get top distribution for each context'''
import json
import mne
import pandas as pd
import numpy as np
from bertopic import BERTopic
from wordcloud import WordCloud
import matplotlib.pyplot as plt

topic_model = BERTopic.load("MaartenGr/BERTopic_Wikipedia")
#topic_model = BERTopic.load("davanstrien/chat_topics")
# for a list of pre-trained topics, see: https://huggingface.co/models?library=bertopic&sort=downloads

def create_wordcloud(model, topic):
    text = {word: value for word, value in model.get_topic(topic)}
    wc = WordCloud(background_color="white", max_words=1000)
    wc.generate_from_frequencies(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()




## Get data
fname = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/LDA/ALL_EXPERIMENTAL_STIMULI/infocon_stims.csv'
df = pd.read_csv(fname)

df_topic = pd.DataFrame()
for column in df.columns:
    topic_index, prob = topic_model.transform(df[column])
    column_result = pd.DataFrame({'topic_index': topic_index, 'prob': prob})
    column_result['condition'] = column
    df_topic = pd.concat([df_topic, column_result])
df_topic = df_topic.reset_index(drop=True)

df_topic.groupby('condition').mean()
df_topic.groupby('condition').std()


# Show wordcloud
create_wordcloud(topic_model, topic=999)

####### Infopos stimuli
fname = r'/Users/linwang/Dropbox (Partners HealthCare)/OngoingProjects/LDA/ALL_EXPERIMENTAL_STIMULI/InfoTopic_stim_toyData_gpt4_output.csv'
df = pd.read_csv(fname)

df_sel = df.iloc[:,-8:-1]

# Create empty lists to store the results
topic_indices = []
probs = []

# Iterate over each element in the DataFrame
for index, row in df_sel.iterrows():
    # Apply topic_model.transform to the element
    topic_index, prob = topic_model.transform(row[2:])
    # Append the topic_index and prob values to the corresponding lists
    topic_indices.append(topic_index)
    probs.append(prob)
result_df = pd.DataFrame({'topic_index': topic_indices, 'prob': probs})
check_df = pd.concat([df_sel['constraint'],result_df],axis=1)
