import os

class DataSet(object):
    '''
    Args:
        data_aug: False for valid/testing.
        shuffle: true for training, False for valid/test.
    '''
    def __init__(self, dataset, batch_size, data_aug=False, shuffle=True):
        np.random.seed(0)
        self.data_path = dataset
        self.batch_size = batch_size
        self.data_aug = data_aug
        self.shuffle = shuffle
        self.x1s, self.x2s self.ys = self.load_data()
        self._num_examples = len(self.ys)
        self.init_epoch()

    def load_data(self):
        '''Fetch all data into a list.'''


    def word2vec(self, word):
        '''Translate a word to a vector.'''

        return vec

    def next_batch(self):
        '''Fetch the next batch of sentences and labels.
           x1_batch is a batch of the first sentences,
           x2_batch is a batch of the first sentences,
           y_batch is a batch of labels.
        '''
        if not self.has_next_batch():
            return None
        # print(self.cur_index)
        return x1_batch, x2_batch, y_batch

    def has_next_batch(self):
        '''Call this function before fetching the next batch.
        If no batch left, a training epoch is over.'''
        start = self.cur_index
        end = self.batch_size + start
        if end > self._num_examples: 
            return False
        else: 
            return True

    def init_epoch(self):
        '''Make sure you would shuffle the training set before the next epoch.
        e.g. if not train_set.has_next_batch(): train_set.init_epoch()'''
