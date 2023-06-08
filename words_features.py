## Get features related to words

#################################################
data = epochs._get_data() #trial*chan*time
words = epochs.metadata['word']

# get clean epochs for each subject
subject = '01'
epochs = _get_epochs(subject)
