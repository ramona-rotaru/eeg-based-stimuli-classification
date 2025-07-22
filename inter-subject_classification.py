from keras.utils import to_categorical

from utils import *
import os
import math

path = 'D:\\eeg_pasion_classification'

xtrain_concat = np.empty((0, 33, 1500))
ytrain_concat = np.empty((0, 3))
xtest_concat = np.empty((0, 33, 1500))
ytest_concat = np.empty((0, 3))


for i in os.listdir(path):
    labels = 32 * [0] + 32 * [1] + 32 * [2]
    if os.path.isfile(os.path.join(path, i)) and 'data_s' in i:
        data, labels = extract_nans(i, labels)
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data, labels, test_size=0.9,
                                                                                    train_size=0.1, random_state=33,
                                                                                    shuffle=True, stratify=labels)

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        xtrain_concat = np.append(xtrain_concat, x_train, axis=0)
        ytrain_concat = np.append(ytrain_concat, y_train, axis=0)
        xtest_concat = np.append(xtest_concat, x_test, axis=0)
        ytest_concat = np.append(ytest_concat, y_test, axis=0)

nb_classes = 3

from EEGModels import EEGNet

model = EEGNet(nb_classes, Chans=33, Samples=1500, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25,
               dropoutType='Dropout')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
save_model_path = 'Model_all_sub_10-90.h5'
model.fit(xtrain_concat, ytrain_concat, batch_size=16, epochs=200, verbose=2, callbacks=[get_tensorboard_callback(), tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=300, verbose=0, mode="auto", baseline=None, restore_best_weights=False), ModelCheckpoint(save_model_path, save_best_only=True, verbose=1)], validation_data=(xtest_concat, ytest_concat))

# model = EEGNet(nb_classes, Chans=33, Samples=1500, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout')
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
load_model_path = 'ceva.h5'

model = tf.keras.models.load_model(load_model_path)
print(model.evaluate(xtest_concat, ytest_concat), 'prediction results')