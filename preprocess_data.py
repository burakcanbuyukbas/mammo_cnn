import numpy as np
import os
import cv2
import pydicom as dicom
from skimage import exposure
from skimage.transform import resize
import pandas as pd
from pydicom.pixel_data_handlers.util import apply_color_lut
import imageio



image_size = 256


def clahe(bgr_image, gridsize=2):
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    hsv_planes = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize))
    hsv_planes[2] = clahe.apply(hsv_planes[2])
    hsv = cv2.merge(hsv_planes)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def get_data(save_data=False, method=0):

    if(method==0):
        X_Train_arr = []
        X_Test_arr = []

        Y_Train, Y_Test = np.empty([0]), np.empty([0])
        Y_train, Y_test = organize_labels(method=0)

        path = r'D:\Users\Burak\mammograpgy\jpg'
        for dir in os.listdir(path):
            try:
                cropdir = dir[8:]
                if (cropdir[-1].isdigit()):
                    print(cropdir)
                    abnormality_id = cropdir[-1]
                    p_index = cropdir.find('P')
                    patient_id = cropdir[p_index + 2: p_index + 7]
                    breast = 'LEFT' if 'LEFT' in cropdir else 'RIGHT'
                    view = 'MLO' if 'MLO' in cropdir else 'CC'
                    abnormality = 'calcification' if cropdir[0:4] == 'Calc' else 'mass'
                    is_train = True if 'Training' in cropdir else False
                    id = 'P_' + patient_id + '_' + breast + '_' + view + '_' + abnormality_id + '_' + abnormality

                    if len(os.listdir(path + '\\' + dir)) == 0:
                        continue
                    dir_ = str(os.listdir(path + '\\' + dir)[0])

                    if len(os.listdir(path + '\\' + dir + '\\' + dir_)) == 0:
                        continue
                    subdir = str(os.listdir(path + '\\' + dir + '\\' + dir_)[0])
                    print('     ' + subdir)

                    if len(os.listdir(
                            path + '\\' + dir + '\\' + str(os.listdir(path + '\\' + dir)[0]) + '\\' + subdir)) == 0:
                        continue
                    subsubdir = str(
                        os.listdir(path + '\\' + dir + '\\' + str(os.listdir(path + '\\' + dir)[0]) + '\\' + subdir)[0])
                    print('             ' + subsubdir)

                    imagew = cv2.imread(path + '\\' + dir + '\\' + dir_ + '\\' + subdir + '\\' + subsubdir)
                    image = clahe(imagew)
                    # img_yuv = cv2.cvtColor(imagew, cv2.COLOR_BGR2YUV)
                    #
                    # # equalize the histogram of the Y channel
                    # img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                    #
                    # # convert the YUV image back to RGB format
                    # image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

                    image = cv2.resize(image, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA)

                    #image = clahe(image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    #cv2.imshow(id, image)
                    #cv2.waitKey(2000)

                    # image = dicom.dcmread(path + '\\' + dir + '\\' + dir_ + '\\' + subdir + '\\' + subsubdir)
                    # # Convert to float to avoid overflow or underflow losses.
                    # image_2d = image.pixel_array.astype(float)
                    #
                    # image_2d = image_histogram_equalization(image_2d)
                    #
                    # # Rescaling grey scale between 0-255
                    # image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
                    #
                    # # Convert to uint
                    # image_2d_scaled = np.uint8(image_2d_scaled)
                    #
                    # resized_img = resize(image_2d_scaled, (image_size, image_size), anti_aliasing=True)
                    #
                    #cv2.imshow(id, image)
                    #cv2.waitKey(500)

                    if is_train:
                        if(Y_train.loc[(Y_train['id'] == id).idxmax(), 'label'].iloc[0] == 1 or
                                Y_train.loc[(Y_train['id'] == id).idxmax(), 'label'].iloc[0] == 0):
                            X_Train_arr.append(image)
                            Y_Train = np.append(Y_Train, Y_train.loc[(Y_train['id'] == id).idxmax(), 'label'].iloc[0])

                    else:
                        if(Y_test.loc[(Y_test['id'] == id).idxmax(), 'label'].iloc[0] == 1 or
                                Y_test.loc[(Y_test['id'] == id).idxmax(), 'label'].iloc[0] == 0):
                            X_Test_arr.append(image)
                            Y_Test = np.append(Y_Test, Y_test.loc[(Y_test['id'] == id).idxmax(), 'label'].iloc[0])

            except Exception as err:
                print("**********************  Error!  *****************************")
                print(err)

        X_Train = np.array(X_Train_arr)
        X_Test = np.array(X_Test_arr)

        print(X_Train.shape)
        print(X_Test.shape)

        if save_data:
            np.save('X_train_' + str(image_size) + '_GRY.npy', X_Train)
            np.save('X_test_' + str(image_size) + '_GRY.npy', X_Test)
            np.save('Y_train.npy', Y_Train)
            np.save('Y_test.npy', Y_Test)

    else:

        X_Train_Calc_arr = []
        X_Train_Mass_arr = []
        X_Test_Calc_arr = []
        X_Test_Mass_arr = []

        Y_Train_Calc, Y_Train_Mass, Y_Test_Calc, Y_Test_Mass = np.empty([0]), np.empty([0]), np.empty([0]), np.empty([0])
        Y_Calc_train, Y_Calc_test, Y_Mass_train, Y_Mass_test = organize_labels(method=1)

        path = r'D:\Users\Burak\mammograpgy\jpg'
        for dir in os.listdir(path):
            try:
                cropdir = dir[8:]
                if (cropdir[-1].isdigit()):
                    print(cropdir)
                    abnormality_id = cropdir[-1]
                    p_index = cropdir.find('P')
                    patient_id = cropdir[p_index + 2: p_index + 7]
                    breast = 'LEFT' if 'LEFT' in cropdir else 'RIGHT'
                    view = 'MLO' if 'MLO' in cropdir else 'CC'
                    abnormality = 'Calc' if cropdir[0:4] == 'Calc' else 'mass'
                    is_train = True if 'Training' in cropdir else False
                    id = 'P_' + patient_id + '_' + breast + '_' + view + '_' + abnormality_id

                    if len(os.listdir(path + '\\' + dir)) == 0:
                        continue
                    dir_ = str(os.listdir(path + '\\' + dir)[0])


                    if len(os.listdir(path + '\\' + dir + '\\' + dir_)) == 0:
                        continue
                    subdir = str(os.listdir(path + '\\' + dir + '\\' + dir_)[0])
                    print('     ' + subdir)

                    if len(os.listdir(path + '\\' + dir + '\\' + str(os.listdir(path + '\\' + dir)[0]) + '\\' + subdir)) == 0:
                        continue
                    subsubdir = str(os.listdir(path + '\\' + dir + '\\' + str(os.listdir(path + '\\' + dir)[0]) + '\\' + subdir)[0])
                    print('             ' + subsubdir)


                    imagew = cv2.imread(path + '\\' + dir + '\\' + dir_ + '\\' + subdir + '\\' + subsubdir)
                    image = clahe(imagew)
                    image = cv2.resize(image, dsize=(image_size, image_size), interpolation=cv2.INTER_AREA)

                    #image = clahe(image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    if is_train:
                        if abnormality == 'Calc':

                                X_Train_Calc_arr.append(image)
                                Y_Train_Calc = np.append(Y_Train_Calc,
                                                    Y_Calc_train.loc[(Y_Calc_train['id'] == id).idxmax(), 'label'])
                        else:

                                X_Train_Mass_arr.append(image)
                                Y_Train_Mass = np.append(Y_Train_Mass,
                                                    Y_Mass_train.loc[(Y_Mass_train['id'] == id).idxmax(), 'label'])
                    else:
                        if abnormality == 'Calc':

                                X_Test_Calc_arr.append(image)
                                Y_Test_Calc = np.append(Y_Test_Calc,
                                                    Y_Calc_test.loc[(Y_Calc_test['id'] == id).idxmax(), 'label'])
                        else:

                                X_Test_Mass_arr.append(image)
                                Y_Test_Mass = np.append(Y_Test_Mass,
                                                    Y_Mass_test.loc[(Y_Mass_test['id'] == id).idxmax(), 'label'])
            except Exception as err:
                print("**********************  Error!  *****************************")
                print(err)

        X_Train_Mass = np.array(X_Train_Mass_arr)
        X_Train_Calc = np.array(X_Train_Calc_arr)
        X_Test_Calc = np.array(X_Test_Calc_arr)
        X_Test_Mass = np.array(X_Test_Mass_arr)

        print("X_Train_Mass: " + str(X_Train_Mass.shape))   # (64, 64, 1318)
        print("X_Train_Calc: " + str(X_Train_Calc.shape))   # (64, 64, 1546)
        print("X_Test_Calc: " + str(X_Test_Calc.shape))    # (64, 64, 326)
        print("X_Test_Mass: " + str(X_Test_Mass.shape))    # (64, 64, 378)

        if save_data:
            np.save('data/X_mass_train_' + str(image_size) + 'GRY.npy', X_Train_Mass)
            np.save('data/X_calc_train_' + str(image_size) + 'GRY.npy', X_Train_Calc)
            np.save('data/X_calc_test_' + str(image_size) + 'GRY.npy', X_Test_Calc)
            np.save('data/X_mass_test_' + str(image_size) + 'GRY.npy', X_Test_Mass)
            np.save('data/Y_mass_train.npy', Y_Train_Mass)
            np.save('data/Y_calc_train.npy', Y_Train_Calc)
            np.save('data/Y_calc_test.npy', Y_Test_Calc)
            np.save('data/Y_mass_test.npy', Y_Test_Mass)

def load_data(method=0):
    if(method==0):
        X_Train = np.load('X_train' + str(image_size) + '_1.npy')
        X_Test = np.load('X_test' + str(image_size) + '_1.npy')
        Y_Train = np.load('Y_train.npy')
        Y_Test = np.load('Y_test.npy')

        return X_Train, X_Test, Y_Train, Y_Test

    else:
        X_Train_Mass = np.load('X_mass_train' + str(image_size) + '.npy')
        X_Train_Calc = np.load('X_calc_train' + str(image_size) + '.npy')
        X_Test_Calc = np.load('X_calc_test' + str(image_size) + '.npy')
        X_Test_Mass = np.load('X_mass_test' + str(image_size) + '.npy')
        Y_Train_Mass = np.load('Y_mass_train.npy')
        Y_Train_Calc = np.load('Y_calc_train.npy')
        Y_Test_Calc = np.load('Y_calc_test.npy')
        Y_Test_Mass = np.load('Y_mass_test.npy')
        print(X_Train_Mass.shape)  # (64, 64, 1318)
        print(X_Train_Calc.shape)  # (64, 64, 1546)
        print(X_Test_Calc.shape)  # (64, 64, 326)
        print(X_Test_Mass.shape)  # (64, 64, 378)
        print(Y_Train_Mass.shape)  # (64, 64, 1318)
        print(Y_Train_Calc.shape)  # (64, 64, 1546)
        print(Y_Test_Calc.shape)  # (64, 64, 326)
        print(Y_Test_Mass.shape)  # (64, 64, 378)
        return X_Train_Mass, X_Train_Calc, X_Test_Mass, X_Test_Calc, Y_Train_Mass, Y_Train_Calc, Y_Test_Mass, Y_Test_Calc

def organize_labels(method=0):
    if(method==0):
        Calc_test_labels = pd.read_csv('calc_case_description_test_set.csv')
        Calc_train_labels = pd.read_csv('calc_case_description_train_set.csv')
        Mass_test_labels = pd.read_csv('mass_case_description_test_set.csv')
        Mass_train_labels = pd.read_csv('mass_case_description_train_set.csv')

        Y_train = pd.DataFrame()
        Y_test = pd.DataFrame()

        train_labels = pd.concat([Calc_train_labels, Mass_train_labels])
        test_labels = pd.concat([Calc_test_labels, Mass_test_labels])
        class_mapping = {'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0, 'MALIGNANT': 1}

        Y_train['id'] = train_labels['patient_id'] + '_' + train_labels['left or right breast'] + '_' + train_labels['image view'] + \
                        '_' + train_labels['abnormality id'].map(str) + '_' + train_labels['abnormality type']
        Y_train['label'] = train_labels['pathology'].map(class_mapping)

        Y_test['id'] = test_labels['patient_id'] + '_' + test_labels['left or right breast'] + '_' + test_labels['image view'] + '_' +\
                       test_labels['abnormality id'].map(str) + '_' + test_labels['abnormality type']
        Y_test['label'] = test_labels['pathology'].map(class_mapping)



        return Y_train, Y_test

    else:
        Calc_test_labels = pd.read_csv('calc_case_description_test_set.csv')
        Calc_train_labels = pd.read_csv('calc_case_description_train_set.csv')
        Mass_test_labels = pd.read_csv('mass_case_description_test_set.csv')
        Mass_train_labels = pd.read_csv('mass_case_description_train_set.csv')
        Y_Calc_train = pd.DataFrame()
        Y_Calc_test = pd.DataFrame()
        Y_Mass_train = pd.DataFrame()
        Y_Mass_test = pd.DataFrame()

        class_mapping = {'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0, 'MALIGNANT': 1}

        Y_Calc_train['id'] = Calc_train_labels['patient_id'] + '_' + Calc_train_labels['left or right breast'] +\
                            '_' + Calc_train_labels['image view'] + '_' + Calc_train_labels['abnormality id'].map(str)
        Y_Calc_train['label'] = Calc_train_labels['pathology'].map(class_mapping)

        Y_Calc_test['id'] = Calc_test_labels['patient_id'] + '_' + Calc_test_labels['left or right breast'] +\
                            '_' + Calc_test_labels['image view'] + '_' + Calc_test_labels['abnormality id'].map(str)
        Y_Calc_test['label'] = Calc_test_labels['pathology'].map(class_mapping)

        Y_Mass_train['id'] = Mass_train_labels['patient_id'] + '_' + Mass_train_labels['left or right breast'] +\
                            '_' + Mass_train_labels['image view'] + '_' + Mass_train_labels['abnormality id'].map(str)
        Y_Mass_train['label'] = Mass_train_labels['pathology'].map(class_mapping)

        Y_Mass_test['id'] = Mass_test_labels['patient_id'] + '_' + Mass_test_labels['left or right breast'] +\
                            '_' + Mass_test_labels['image view'] + '_' + Mass_test_labels['abnormality id'].map(str)
        Y_Mass_test['label'] = Mass_test_labels['pathology'].map(class_mapping)

        return Y_Calc_train, Y_Calc_test, Y_Mass_train, Y_Mass_test



#get_data(save_data=True, method=1)