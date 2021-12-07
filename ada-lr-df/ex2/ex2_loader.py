# Pickle data from data folder
import _pickle as cPickle
import gzip

def load_data_wrapper(FILEPATH):
    f1=gzip.open(FILEPATH,'rb')
    training_data, validation_data, test_data, data_mean, data_std, label_mean, label_std=cPickle.load(f1, encoding='latin1')
    f1.close()
    return (training_data, validation_data, test_data, data_mean, data_std, label_mean, label_std)

def load_new_data(FILEPATH):
    f2=gzip.open(FILEPATH,'rb')
    datas, labels, x_new=cPickle.load(f2, encoding='latin1')
    f2.close()
    return (datas, labels, x_new)