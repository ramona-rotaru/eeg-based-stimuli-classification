from utils import *

path = 'D:\\eeg_pasion_classification\\data'

for subdir, dirs, files in os.walk(path):
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and '_2.mat' in i:
            filepath = subdir + os.sep + i
            print(filepath)
            annots = loadmat(filepath)
            print(annots.keys())
            for key, value in annots.items():
                if key == 'data_pers':
                    data_pers = value
                    data_pers1 = data_pers[:, 500:2000, :]
                    data_pers2 = swap_axes(data_pers1)
                    print(data_pers2.shape)
                if key == 'data_fractals':
                    data_fractals = value
                    data_fractals1 = data_fractals[:, 500:2000, :]
                    data_fractals2 = swap_axes(data_fractals1)
                    print(data_fractals2.shape)
                if key == 'data_uni':
                    data_uni = value
                    data_uni1 = data_uni[:, 500:2000, :]
                    data_uni2 = swap_axes(data_uni1)
                    print(data_uni2.shape)
data = np.concatenate((data_pers2, data_fractals2, data_uni2), axis=0)

position = np.where(np.isnan(data[:, 0, 0]))[0]
labels = [0] * 32 + [1] * 32 + [2] * 32
if len(position) != 0:
    index_positions = position
    print(index_positions)
    contor = 0
    for index in index_positions:
        index = index - contor
        data = np.delete(data, index, axis=0)
        del labels[index]
        contor += 1

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data, labels, test_size=0.4,
                                                                            train_size=0.6, random_state=33,
                                                                          shuffle=True, stratify=labels)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

nb_classes = 3

model = EEGNet(nb_classes, Chans=33, Samples=1500, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25,
               dropoutType='Dropout')
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
print(model.summary())
save_model_path = 'Model_Sub_2_60-40.h5'
model.fit(x_train, y_train, batch_size=16, epochs=100, verbose=2, callbacks=[get_tensorboard_callback(), tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=300, verbose=0, mode="auto", baseline=None, restore_best_weights=False), ModelCheckpoint(save_model_path, save_best_only=True, verbose=1)], validation_data=(x_test, y_test))

# load_model_path = 'Model_Sub_2_60-40.h5'
model = tf.keras.models.load_model(save_model_path)
print(model.evaluate(x_test, y_test), 'prediction result')
y_score = model.predict(x_test)
y_predict = np.argmax(y_score, axis=1)
y_test_arr = np.asarray(y_test)
cfm = sklearn.metrics.confusion_matrix(y_test_arr, y_predict, labels=None, sample_weight=None, normalize=None)
print(cfm)

