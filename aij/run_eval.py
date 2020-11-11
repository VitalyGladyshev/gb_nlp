import os

import cv2
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Conv2D, Dense, GRU, Input, Lambda, MaxPool2D, LSTM
from tensorflow.keras.models import Model

letters = [' ', ')', '+', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[', ']', 'i', 'k', 'l', '|', '×', 'ǂ',
           'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х',
           'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'і', 'ѣ', '–', '…', '⊕', '⊗']


def process_image_g(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
#     print(f"w: {w} h: {h}")
    if h > (w*2.5):
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)   # ROTATE_90_CLOCKWISE
        h, w = img.shape
    
    new_h = 128
    new_w = int(w * (new_h / h))
    img = cv2.resize(img, (new_w, new_h))
    h, w = img.shape
    
    img = img.astype('float32')
    
    if h < 128:
        add_zeros = np.full((128-h, w), 255)
        img = np.concatenate((img, add_zeros))
        h, w = img.shape
    
    if w < 1024:
        add_zeros = np.full((h, 1024-w), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        h, w = img.shape
        
    if w > 1024 or h > 128:
        dim = (1024,128)
        img = cv2.resize(img, dim)
    
    img = cv2.subtract(255, img)

    img = img / 255
    
    return img

def create_model():
    inputs = Input(shape=(128, 1024, 1))

    conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    pool_1 = MaxPool2D(pool_size=(4, 2), strides=2)(conv_1)

    conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(4, 2), strides=2)(conv_2)

    conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)

    conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_3)

    pool_4 = MaxPool2D(pool_size=(4, 1), padding='same')(conv_4)

    conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)

    batch_norm_5 = BatchNormalization()(conv_5)

    conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(4, 1), padding='same')(batch_norm_6)

    conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_6)

    squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

    blstm_1 = Bidirectional(LSTM(1024, return_sequences=True, dropout=0.2))(squeezed)
    blstm_2 = Bidirectional(LSTM(1024, return_sequences=True, dropout=0.2))(blstm_1)

    outputs = Dense(len(letters) + 1, activation='softmax')(blstm_2)
    act_model = Model(inputs=inputs, outputs=outputs)

    return act_model


def get_prediction(act_model, test_images):
    prediction = act_model.predict(test_images)

    decoded = K.ctc_decode(prediction,
                           input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                           greedy=True)[0][0]

    out = K.get_value(decoded)

    prediction = []
    for i, x in enumerate(out):
        pred = ''
        for p in x:
            if int(p) != -1:
                pred += letters[int(p)]

        prediction.append(pred)
    return prediction


def write_prediction(names_test, prediction, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for num, (name, line) in enumerate(zip(names_test, prediction)):
        with open(os.path.join(output_dir, name.replace('.jpg', '.txt')), 'w') as file:
            file.write(line)


def load_test_images(test_image_dir):
    test_images = []
    names_test = []
    for name in os.listdir(test_image_dir):
        img = cv2.imread(test_image_dir + '/' + name, cv2.IMREAD_GRAYSCALE)
        img = process_image_g(img)
        test_images.append(img)
        names_test.append(name)
    test_images = np.asarray(test_images)
    test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 1)
    return names_test, test_images


def main():
    test_image_dir = '/data'
    filepath = 'checkpoint/model_lstm_1024_20.hdf5'
    pred_path = '/output'

    print('Creating model...', end=' ')
    act_model = create_model()
    print('Success')

    print(f'Loading weights from {filepath}...', end=' ')
    act_model.load_weights(filepath)
    print('Success')

    print(f'Loading test images from {test_image_dir}...', end=' ')
    names_test, test_images = load_test_images(test_image_dir)
    print('Success')

    print('Running inference...')
    prediction = get_prediction(act_model, test_images)

    print('Writing predictions...')
    write_prediction(names_test, prediction, pred_path)


if __name__ == '__main__':
    main()
