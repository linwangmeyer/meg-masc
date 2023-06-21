from bertopic import BERTopic

# Initialize and train BERTopic on the full dataset with the optimal number of topics
model = BERTopic(num_topics=optimal_num_topics)
topics, _ = model.fit_transform(full_dataset)

# Get the topic-word distribution for the optimal number of topics
topic_word_distribution = model.get_topic()

# Print the top words for each topic
for topic_id, topic_words in topic_word_distribution.items():
    print(f"Topic {topic_id}: {', '.join(topic_words[:5])}")  # Print the top 5 words for each topic




# Initialize and train BERTopic on the full dataset with the optimal number of topics
model = BERTopic(num_topics=optimal_num_topics)
topics, _ = model.fit_transform(full_dataset)

# Get the topic-word distribution for the optimal number of topics
topic_word_distribution = model.get_topic()

# Print the top words for each topic
for topic_id, topic_words in topic_word_distribution.items():
    print(f"Topic {topic_id}: {', '.join(topic_words[:5])}")  # Print the top 5 words for each topic
