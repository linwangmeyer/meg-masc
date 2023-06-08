## Test evoked activity


###########################################################
# Check evoked activity modulated by word length
epochs.metadata['NumberOfLetters'] = epochs.metadata['word'].apply(lambda x: len(x))
evokeds = dict()
query = "NumberOfLetters == {}"
for n_letters in epochs.metadata["NumberOfLetters"].unique():
    evokeds[str(n_letters)] = epochs[query.format(n_letters)].average()
mne.viz.plot_compare_evokeds(evokeds, cmap=("word length", "viridis"), picks="MEG 011")

