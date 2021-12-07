# Pickle data from data folder
import _pickle as cPickle
import gzip

#PATH_TRAIN='../SmallScale/Data/SmallGoodTrain.pklz'
#PATH_NEW='../SmallScale/Data/SmallGoodNew.pklz'

def load_data_wrapper(PATH_TRAIN):
    f1=gzip.open(PATH_TRAIN,'rb')
    training_data, validation_data, test_data, data_mean, data_std, label_mean, label_std=cPickle.load(f1, encoding='latin1')
    f1.close()
    return (training_data, validation_data, test_data, data_mean, data_std, label_mean, label_std)

def load_new_data(PATH_NEW):
    f2=gzip.open(PATH_NEW,'rb')
    datas, labels, x_new=cPickle.load(f2, encoding='latin1')
    f2.close()
    return (datas, labels, x_new)