from tfidf_use import tfidf
import xgboost as xgb
import numpy as np
import time
import csv
import pickle
import data

DEBUG = True


def train_by_xgb(train_X_array, train_Y_array, test_X_array, test_Y_array, model_name, version):
    print('start training ')
    xg_train = xgb.DMatrix(train_X_array, label = train_Y_array)
    xg_test = xgb.DMatrix(test_X_array,label = test_Y_array)

    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softmax'
    # scale weight of positive examples
    param['eta'] = 0.1
    param['max_depth'] = 5
    param['silent'] = 1
    param['num_class'] = 6

    watchlist = [ (xg_train,'eval'), (xg_test, 'test') ]
    num_round = 5

    if version == 1:
        model = xgb.train(param, xg_train, num_round, watchlist)
        model.save_model('{}{}.model'.format(model_name, version))
    elif version == 2:
        model = xgb.train(param, xg_train, num_round, watchlist, xgb_model='{}{}.model'.format(model_name,version - 1))
        model.save_model('{}{}.model'.format(model_name,version))
    elif version == 3:
        model = xgb.train(param, xg_train, num_round, watchlist, xgb_model='{}{}.model'.format(model_name,version - 1))
        model.save_model('{}{}.model'.format(model_name,version))
    elif version == 4:
        model = xgb.train(param, xg_train, num_round, watchlist, xgb_model='{}{}.model'.format(model_name,version - 1))
        model.save_model('{}{}.model'.format(model_name,version))



def main(test_data, test_label):

    name_row = []
    training_label1 = []
    training_label2 = []
    training_label3 = []
    training_label4 = []

    training_data1 = []
    training_data2 = []
    training_data3 = []
    training_data4 = []
    start_time = time.perf_counter()

    with open('testset_with_label.csv', 'r') as db01:
        reader = csv.reader(db01)
        for i, row in enumerate(reader):
            if i == 1:
                name_row.append(row)
            elif i > 2 and i < 10003:
                # get tfidf
                temp = tfidf.get_instance_tfidf_vector(str(row[0]))
                temp.extend(row[11:])

                training_label1.append(int(row[10]))
                training_data1.append(temp)

                del temp
            elif i > 10003 and i < 20004:
                # get tfidf
                temp = tfidf.get_instance_tfidf_vector(str(row[0]))
                temp.extend(row[11:])

                training_label2.append(int(row[10]))
                training_data2.append(temp)
            elif i > 20005 and i < 30006:
                # get tfidf
                temp = tfidf.get_instance_tfidf_vector(str(row[0]))
                temp.extend(row[11:])

                training_label3.append(int(row[10]))
                training_data3.append(temp)

            elif i > 30006:
                # get tfidf
                temp = tfidf.get_instance_tfidf_vector(str(row[0]))
                temp.extend(row[11:])

                training_label4.append(int(row[10]))
                training_data4.append(temp)

    # Test data is last 4k data
    test_data_array = np.array(test_data, dtype=float, ndmin=2)
    test_label_array = np.array(test_label, dtype=int).reshape(4000, 1)

    end_time = time.perf_counter()
    print('Finish {} s'.format(end_time - start_time))
    # array
    data1 = np.array(training_data1, dtype=float, ndmin=2)
    label1 = np.array(training_label1, dtype=int).reshape(len(training_data1), 1)

    del training_data1
    del training_label1

    data2 = np.array(training_data2, dtype=float, ndmin=2)
    label2 = np.array(training_label2, dtype=int).reshape(len(training_data2), 1)

    del training_data2
    del training_label2

    data3 = np.array(training_data3, dtype=float, ndmin=2)
    label3 = np.array(training_label3, dtype=int).reshape(len(training_data3), 1)

    del training_data3
    del training_label3

    data4 = np.array(training_data4, dtype=float, ndmin=2)
    label4 = np.array(training_label4, dtype=int).reshape(len(training_data4), 1)

    del training_data4
    del training_label4

    test_X_array = test_data_array
    test_Y_array = test_label_array

    train_X_array = data1
    train_Y_array = label1
    train_by_xgb(train_X_array, train_Y_array, test_X_array, test_Y_array, 'model_v', 1)

    train_X_array = data2
    train_Y_array = label2
    train_by_xgb(train_X_array, train_Y_array, test_X_array, test_Y_array, 'model_v', 2)

    train_X_array = data3
    train_Y_array = label3
    train_by_xgb(train_X_array, train_Y_array, test_X_array, test_Y_array, 'model_v', 3)

    train_X_array = data4
    train_Y_array = label4
    train_by_xgb(train_X_array, train_Y_array, test_X_array, test_Y_array, 'model_v', 4)



if __name__ == '__main__':
    test_label = data.read_pickle('test_label.pickle')
    test_data = data.read_pickle('test_data.pickle')
    main(test_data, test_label)



