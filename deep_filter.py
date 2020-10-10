"""
deep_filter.py
Input: datas,labels,x_hats,x_bars,x_raws,y_raws
Output: model, data_mean, data_std, label_mean, label_std
"""

#----------------------------------
# Deep Filtering Function
#----------------------------------

def deep_filtering(datas,labels,x_hats,x_bars,x_raws,y_raws):
    """datas,labels,x_hats,x_bars,x_raws,y_raws"""
    
    # Data Preprocessing Procedure
    datas=datas.reshape(((N-n0+2)*N_sample,dimY*n0))
    # convert numpy array into pandas dataframe
    datas=pd.DataFrame(datas)
    labels=pd.DataFrame(labels)
    x_hats=pd.DataFrame(x_hats)
    x_bars=pd.DataFrame(x_bars)
    
    #from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    seed=3
    np.random.seed(seed)
    training_data, test_data, training_label, test_label=train_test_split(datas,labels, test_size=0.2, random_state=seed)

    # Input normalization
    data_mean=training_data.mean(axis=0)
    data_std=training_data.std(axis=0)
    training_data=(training_data-data_mean)/data_std
    test_data=(test_data-data_mean)/data_std

    # Output normalization
    label_mean=training_label.mean(axis=0)
    label_std=training_label.std(axis=0)
    training_label=(training_label-label_mean)/label_std
    test_label=(test_label-label_mean)/label_std

    #-------------------------------
    # Model building
    #-------------------------------

    from keras import models
    from keras import layers
    from keras import optimizers

    def build_model():
        model=models.Sequential()
        model.add(layers.Dense(5,activation='relu',input_shape=(dimY*n0,)))
        model.add(layers.Dense(5,activation='relu'))
        model.add(layers.Dense(5,activation='relu'))
        model.add(layers.Dense(5,activation='relu'))
        model.add(layers.Dense(5,activation='relu'))
        model.add(layers.Dense(dimX))

        model.compile(optimizer=optimizers.SGD(lr=0.001), 
                      loss='mean_squared_error', 
                      metrics=[tf.keras.metrics.MeanSquaredError()])
        return model

    model=build_model()
    mymodel=model.fit(training_data,training_label, epochs=10, batch_size=16)

    #-------------------------------
    # Evaluation Performance
    #-------------------------------

    from sklearn.metrics import mean_squared_error

    test_mse_score, test_mae_score=model.evaluate(test_data,test_label)

    index=test_label.index.tolist()

    # Need to do same normalization with deep filtering to compare.
    x_bars=(x_bars-label_mean)/label_std
    kf_mse_err=mean_squared_error(x_bars.iloc[index],test_label)

    cpu_end=time.perf_counter()

    #print("The mse of deep filtering is {:.3%}".format(test_mse_score))
    #print("The mse of Kalman Filtering is {:.3%}".format(kf_mse_err))
    #print("The CPU consuming time is {:.5}".format(cpu_end-cpu_start))

    #history_dict=mymodel.history

    #loss_value=history_dict['loss']
    #val_loss_value=history_dict['val_loss']
    #epochs=range(1,10+1)
    #import matplotlib.pyplot as plt
    #plt.plot(epochs, loss_value, 'bo',label='Training Loss')
    #plt.plot(epochs, val_loss_value,'b',label='Validation Loss')
    #plt.legend()
    #plt.show()
    
    return model, data_mean, data_std, label_mean, label_std