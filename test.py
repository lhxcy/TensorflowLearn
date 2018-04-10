# # from six.moves import urllib
# # import os
# # import sys
# # import tensorflow as tf
# # import tarfile
# # FLAGS = tf.app.flags.FLAGS#提取系统参数作用的变量
# # tf.app.flags.DEFINE_string('dir','D:/download_html','directory of html')#将下载目录保存到变量dir中,通过FLAGS.dir提取
# # directory = FLAGS.dir#从FLAGS中提取dir变量
# # url = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
# # filename = url.split('/')[-1]#-1表示分割后的最后一个元素
# # filepath = os.path.join(directory,filename)
# # if not os.path.exists(directory):
# #     os.makedirs(directory)
# # if not os.path.exists(filepath):
# #     def _recall_func(num,block_size,total_size):
# #         sys.stdout.write('\r>> downloading %s %.1f%%' % (filename,float(num*block_size)/float(total_size)*100.0))
# #         sys.stdout.flush()
# #     urllib.request.urlretrieve(url,filepath,_recall_func)
# #     print()
# #     file_info = os.stat(filepath)
# #     print('Successfully download',filename,file_info.st_size,'bytes')
# # tar = tarfile.open(filepath,'r:gz')#指定解压路径和解压方式为解压gzip
# # tar.extractall(directory)#全部解压
#
#
#
#
#
#
# import gzip
# import os
# import numpy
# import sys
# import six.moves.cPickle as pickle
# from six.moves import urllib
#
# def get_dataset_file(path, default_dataset, origin):
#     '''Look for it as if it was a full path, if not, try local file,
# 	if not try in the data directory.
# 	Download dataset if it is not present
# 	'''
#     data_dir, data_file = os.path.split(path)
#     print(data_dir)
#     print(data_file)
#     # filename = origin.split('/')[-1]  # -1表示分割后的最后一个元素
#     # if not os.path.exists(path):
#     #     urllib.request.urlretrieve(origin, path)
#     if (not os.path.isfile(path)) and data_file == default_dataset:
#         print('Downloading data from %s' % origin)
#         urllib.request.urlretrieve(origin, path)
#     # 	from six.moves import urllib
#     # 	urllib.request.urlretrieve(origin,'',_recall_func)
#
#
#     return path
#
#
# def load_data(path="imdb.pkl", n_words=100000, valid_portion=0.1,
#               maxlen=None,
#               sort_by_len=True):
#     '''Loads the dataset
#     :type path: String
#     :param path: The path to the dataset (here IMDB)
#     :type n_words: int
#     :param n_words: The number of word to keep in the vocabulary.
#         All extra words are set to unknow (1).
#     :type valid_portion: float
#     :param valid_portion: The proportion of the full train set used for
#         the validation set.
#     :type maxlen: None or positive int
#     :param maxlen: the max sequence length we use in the train/valid set.
#     :type sort_by_len: bool
#     :name sort_by_len: Sort by the sequence lenght for the train,
#         valid and test set. This allow faster execution as it cause
#         less padding per minibatch. Another mechanism must be used to
#         shuffle the train set at each epoch.
#     '''
#
#     #############
#     # LOAD DATA #
#     #############
#
#     # Load the dataset
#     path = get_dataset_file(
#         path, "imdb.pkl",
#         "http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl")
#
#     print(path)
#     if path.endswith(".gz"):
#         f = gzip.open(path, 'rb')
#     else:
#         print("open")
#         f = open(path, 'rb')
#
#
#     train_set = pickle.load(f)
#     print(type(train_set))
#     print(len(train_set))
#     print(train_set[0][0])
#     print(train_set[1][0])
#     test_set = pickle.load(f)
#     print(type(test_set))
#     f.close()
#     if maxlen:
#         new_train_set_x = []
#         new_train_set_y = []
#         for x, y in zip(train_set[0], train_set[1]):
#             if len(x) < maxlen:
#                 new_train_set_x.append(x)
#                 new_train_set_y.append(y)
#         train_set = (new_train_set_x, new_train_set_y)
#         del new_train_set_x, new_train_set_y
#
#     # split training set into validation set
#     train_set_x, train_set_y = train_set
#     n_samples = len(train_set_x)
#     print(n_samples)
#     sidx = numpy.random.permutation(n_samples)
#     '''
#     如果传给permutation一个矩阵，它会返回一个洗牌后的矩阵副本；
#     而shuffle只是对一个矩阵进行洗牌，无返回值。 如果传入一个整数，它会返回一个洗牌后的arange
#     '''
#     n_train = int(numpy.round(n_samples * (1. - valid_portion)))
#     valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
#     valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
#     train_set_x = [train_set_x[s] for s in sidx[:n_train]]
#     train_set_y = [train_set_y[s] for s in sidx[:n_train]]
#
#     train_set = (train_set_x, train_set_y)
#     valid_set = (valid_set_x, valid_set_y)
#
#     def remove_unk(x):
#         return [[1 if w >= n_words else w for w in sen] for sen in x]
#
#     test_set_x, test_set_y = test_set
#     valid_set_x, valid_set_y = valid_set
#     train_set_x, train_set_y = train_set
#
#     train_set_x = remove_unk(train_set_x)
#     valid_set_x = remove_unk(valid_set_x)
#     test_set_x = remove_unk(test_set_x)
#
#     def len_argsort(seq):
#         return sorted(range(len(seq)), key=lambda x: len(seq[x]))
#
#     if sort_by_len:
#         sorted_index = len_argsort(test_set_x)
#         test_set_x = [test_set_x[i] for i in sorted_index]
#         test_set_y = [test_set_y[i] for i in sorted_index]
#
#         sorted_index = len_argsort(valid_set_x)
#         valid_set_x = [valid_set_x[i] for i in sorted_index]
#         valid_set_y = [valid_set_y[i] for i in sorted_index]
#
#         sorted_index = len_argsort(train_set_x)
#         train_set_x = [train_set_x[i] for i in sorted_index]
#         train_set_y = [train_set_y[i] for i in sorted_index]
#
#     train = (train_set_x, train_set_y)
#     valid = (valid_set_x, valid_set_y)
#     test = (test_set_x, test_set_y)
#
#     return train, valid, test
#
#
# train, valid, test = load_data(path='data/imdb.pk1', n_words=10000,
# 							 valid_portion=0.1)


lis = []
for i in range(0,100,5):
	lis.append(i)
print(lis)

import numpy as np
x = np.zeros((4,3,2))
print(x)









