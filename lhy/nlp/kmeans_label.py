import csv
import data
import pickle
import matplotlib.pyplot as plt

dict_agency_label = {}
with open('行业等级label类字典.txt', 'r') as f:
    for row in f:
        temp = row.split(',')
        dict_agency_label[temp[0]] = temp[1].strip('\n')
# print(dict_agency_label)

data_lst = []
try:
    with open('testset1.csv', 'r') as db01:
        reader = csv.reader(db01)
        for row in reader:
            data_lst.append(row)
except csv.Error as e:
    print("Error at line %s :%s", reader.line_num, e)

row_name = data_lst[0]
row_name.append('Label标签')
row_name.append('tfidf_value')
row_name.append('行业等级一的tfidf_value')
row_name.append('行业等级二的tfidf_value')
row_name.append('行业等级三的tfidf_value')
row_name.append('行业等级四的tfidf_value')

print(row_name)
for i in data_lst[1:]:
    if i[9] in dict_agency_label.keys():
        i.append(dict_agency_label[i[9]])
    else:
        i.append('5')
print(data_lst[1])
#
tfidf_value_dict = data.read_pickle("tf-idf_value.pickle")

# print(tfidf_value_dict)
for i in data_lst[1:]:
    temp_tfidf_sum = 0
    temp_one = 0
    temp_two = 0
    temp_three = 0
    temp_four = 0
    if i[2] in tfidf_value_dict.keys():
        temp_one = temp_one + tfidf_value_dict[i[2]]
    else:
        temp_one = temp_one
    if i[3] in tfidf_value_dict.keys():
        temp_two = temp_two + tfidf_value_dict[i[3]]
    else:
        temp_two = temp_two
    if i[4] in tfidf_value_dict.keys():
        temp_three = temp_three + tfidf_value_dict[i[4]]
    else:
        temp_three = temp_three
    if i[5] in tfidf_value_dict.keys():
        temp_four = temp_four + tfidf_value_dict[i[5]]
    else:
        temp_four = temp_four
    temp_tfidf_sum = temp_one+temp_two+temp_three+temp_four
    i.append(temp_tfidf_sum)
    i.append(temp_one)
    i.append(temp_two)
    i.append(temp_three)
    i.append(temp_four)
print(data_lst[1])
# count_zero = 0
# count_one = 0
# count_two = 0
# count_three = 0
# count_four = 0
# count_five = 0
# #
# # for i in data_lst[1:]:
# #     temp = 0
# #     if i[10] == '0':
# #         count_zero += 1
# #     else:
# #         count_zero = count_zero
# #     if i[10] == '1':
# #         count_one += 1
# #     else:
# #         count_one = count_one
# #     if i[10] == '2':
# #         count_two += 1
# #     else:
# #         count_two = count_two
# #     if i[10] == '3':
# #         count_three +=1
# #     else:
# #         count_three = count_three
# #     if i[10] == '4':
# #         count_four +=1
# #     else:
# #         count_four = count_four
# #     if i[10] == '5':
# #         count_five += 1
# #     else:
# #         count_five = count_five
# # print('----SMOTE之前的聚类划分数量情况----')
# # print('第0类的有{}个'.format(count_zero))
# # print('第1类的有{}个'.format(count_one))
# # print('第2类的有{}个'.format(count_two))
# # print('第3类的有{}个'.format(count_three))
# # print('第4类的有{}个'.format(count_four))
# # print('第5类的有{}个'.format(count_five))
#
#
with open("testset_with_label.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    #先写入columns_name
    writer.writerow(row_name)
    #写入多行用writerows
    writer.writerows(data_lst)
#
# import numpy as np
#
# from imblearn.combine import SMOTEENN
# y_label = []
# x_tfidf = []
# for i in data_lst[1:]:
#     x_tfidf.append(i[11:])
#     y_label.append(int(i[10]))
#
# y_label = np.array(y_label)
# x_label = np.array(x_tfidf)
#
# print(x_label.shape)
# print(y_label.shape)
# plot_y0 = []
# plot_y1 = []
# plot_y2 = []
# plot_y3 = []
# for i in x_label:
#     plot_y0.append(i[0])
#     plot_y1.append(i[1])
#     plot_y2.append(i[2])
#     plot_y3.append(i[3])
#
# # plt.ylabel("Tf-Idf Count")
# # plt.xlabel("Tf-Idf Value")
# # plt.title("Tf-Idf Distribution")
# # plt.subplot(221)
# # plt.hist(plot_y0, range = (0,1))
# # plt.subplot(222)
# # plt.hist(plot_y1, range = (0,1))
# # plt.subplot(223)
# # plt.hist(plot_y2, range = (0,1))
# # plt.subplot(224)
# # plt.hist(plot_y3, range = (0,1))
# # plt.show()
#
# sm = SMOTEENN()
# X_resampled, y_resampled = sm.fit_sample(x_label, y_label)
# print(X_resampled.shape)
# print(y_resampled.shape)
#
# #storage sample
# X_storage = np.c_[X_resampled,y_resampled]
# with open('training_data_tfidf_with_label.pickle', 'wb') as f:
# 	pickle.dump(X_storage, f, pickle.HIGHEST_PROTOCOL)
#
#
#
#
#
# count_zero = 0
# count_one = 0
# count_two = 0
# count_three = 0
# count_four = 0
# count_five = 0
#
# # for i in y_resampled:
# #     temp = 0
# #     if i == 0:
# #         count_zero += 1
# #     else:
# #         count_zero = count_zero
# #     if i == 1:
# #         count_one += 1
# #     else:
# #         count_one = count_one
# #     if i == 2:
# #         count_two += 1
# #     else:
# #         count_two = count_two
# #     if i == 3:
# #         count_three +=1
# #     else:
# #         count_three = count_three
# #     if i == 4:
# #         count_four +=1
# #     else:
# #         count_four = count_four
# #     if i == 5:
# #         count_five += 1
# #     else:
# #         count_five = count_five
# # print('----SMOTE之后的聚类划分数量情况----')
# # print('第0类的有{}个'.format(count_zero))
# # print('第1类的有{}个'.format(count_one))
# # print('第2类的有{}个'.format(count_two))
# # print('第3类的有{}个'.format(count_three))
# # print('第4类的有{}个'.format(count_four))
# # print('第5类的有{}个'.format(count_five))
# # # 统计
#
# # plot_y0 = []
# # plot_y1 = []
# # plot_y2 = []
# # plot_y3 = []
# # for i in X_resampled :
# #     plot_y0.append(i[0])
# #     plot_y1.append(i[1])
# #     plot_y2.append(i[2])
# #     plot_y3.append(i[3])
#
# # plt.ylabel("Tf-Idf Count")
# # plt.xlabel("Tf-Idf Value")
# # plt.title("Tf-Idf Distribution")
# # plt.subplot(221)
# # plt.hist(plot_y0, range = (0,1))
# # plt.subplot(222)
# # plt.hist(plot_y1, range = (0,1))
# # plt.subplot(223)
# # plt.hist(plot_y2, range = (0,1))
# # plt.subplot(224)
# # plt.hist(plot_y3, range = (0,1))
# # plt.show()






