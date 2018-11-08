import jieba
import jieba.posseg as pseg
import csv
import re

valid_data = []
unvalid_data = []
other_place = []
count_valid = 0
count_unvalid = 0
count_other_place = 0
#取出有效处置单位
try:
    with open('top5000.csv', 'r', encoding= 'GB18030') as db01:
        reader = csv.reader(db01)
        for row in reader:
            # 处置单位为空白
            if row[12] == '':
                unvalid_data.append(row)
                count_unvalid += 1
            # 处置单位为其他单位
            elif row[12] == '其他单位':
                other_place.append(row)
                count_other_place +=1
            # 有效处置单位
            else:
                valid_data.append(row)
                count_valid += 1

except csv.Error as e:
    print("Error at line %s :%s", reader.line_num, e)

list_describ_seg=[]
for i in valid_data:
    temp = i[7]
    str = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[a-zA-Z0-9:+——！，。？、~@#￥%……&*（）：《》]+", "",temp)
    words = pseg.cut(str)
    dict_describ_seg = {
        'id': i[0],
        'describ_seg':"/".join('%s%s ' % (w.flag, w.word) for w in words),
        'describ':i[7]
    }
    list_describ_seg.append(dict_describ_seg)


print(list_describ_seg[3])

place_seg_obj = open('诉求内容词性标签.txt', 'w')
for i in list_describ_seg:
    for key,value in i.items():
        place_seg_obj.write(key+':'+ value)
        place_seg_obj.write('\n')
place_seg_obj.close()

# list_place_seg_set = []
# for i in range(len(list_place_seg)):
#     list_place_seg_set.append(list_place_seg[i]['place_seg'])
# list_place_seg_set = set(list_place_seg_set)
#
# for i in list_place_seg_set:
#     print(i)
#     place_seg_obj.write(i)
#     place_seg_obj.write('\n')
# place_seg_obj.close()
