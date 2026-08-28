# Import third-party libraries
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential, Input
from tensorflow.keras.optimizers import Adam

def model_maker(model_label, csv_paths: list, download_csv: bool = False, csv_label:str = ''):

    # read each csv file into data frames according to the cvs_paths list from
    # the parameter and add them to an empty list
    data_frames = []
    for path in csv_paths:
        path = f'csv_folder/{path}.csv'
        df = pd.read_csv(path)
        data_frames.append(df)

    # joining all data frames into one data frame and randomize them with new index
    data_frame = pd.concat(data_frames, ignore_index= True)
    data_frame = data_frame.sample(frac = 1, random_state = 42).reset_index(drop= True)

    # separate the data into x, y and make sure to fill the empty data
    # and maintain the order of the original labeling before turning 
    # non-numeric data into numeric for the AI to read
    x = data_frame.drop('label', axis= 1)
    data_frame['label'] = data_frame['label'].fillna('None')
    data_frame['label'] = pd.Categorical(data_frame['label'], categories=csv_paths, ordered= True)
    y = pd.get_dummies(data_frame['label']).astype(int)

    # get the number of output classes
    outputs_num = y.shape[1]

    # if the parameter download_csv is True then save the data frame into a local csv file
    if download_csv:
        data_frame.to_csv(f'{csv_label}.csv', index= False)

    # EarlyStopping object to prevent overfitting
    call_back = EarlyStopping(
        monitor= 'val_loss',
        patience= 30,
        verbose= 1,
        restore_best_weights= True
    )

    # the model layers
    model = Sequential([
        Input(shape= (63,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(outputs_num, 'softmax')
    ])

    # compile the model with appropriate metrics and optimizer 
    model.compile(
        loss = 'categorical_crossentropy',
        metrics = ['accuracy'],
        optimizer = Adam()
    )

    # training the model 
    model.fit(
        x = x,
        y = y,
        epochs = 15000,
        callbacks = [call_back],
        validation_split = 0.3,
        verbose= 2
    )
    model.summary()

    # convert and save the model as a TFLite model instead of keras format since
    # TFLite is lightweight in comparison to prevent any lag or frame drops
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(f'{model_label}.tflite', 'wb') as f:
        f.write(tflite_model)

    print(f'the model {model_label} is ready')