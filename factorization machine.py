import os
import numpy as np
from scipy import sparse
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer

def collect_item(data,feature_index,delimiter):
    items = []
    for index , row in enumerate(data[1:]):
        print('Collecting item now:' , index)
        feature = line.split(delimiter)
        if feature[feature_index] not in items:
            items.append(feature[feature_index])
    return items

def dis_item_list(data,feature_index,delimiter):
    try:
        dis_item = open('discard_list.csv').readlines()[0].split(',')
    except:
        items = collect_item(data,feature_index,delimiter)
        item = [0 for j in range(len(items))]
        for index , line in enumerate(data[1:]):
            print('Sum item vector now:' , index)
            feature = line.split(delimiter)
            item[items.index(feature[feature_index])] += 1
        count = []
        for index , item in enumerate(item):
            if item < 5 :
                count.append(index)
        dis_item = []
        for i in count:
            dis_item.append(items[i])
    return dis_item

def create_csv(data, path):  #清理不要的資料欄位與重新給予user ID與item ID
    import csv
    with open(path,'w',newline='') as f:
        csv_write = csv.writer(f, delimiter = '|')
        csv_write.writerow(["User_id","Age","Gender","Level","Styles","Country","City","Item_id","Attributes","Tag"])
        ord_user_index = []
        ord_item_index = []
        for index , line in enumerate(data[1:]):
            print('Saving data now:' , index)
            feature = line.split('\t')
            if feature[2] in dis_list:
                continue
            if feature[14] not in ord_user_index:
                ord_user_index.append(feature[14])
                feature[14] = str(ord_user_index.index(feature[14]))
            else:
                feature[14] = str(ord_user_index.index(feature[14]))
            if feature[2] not in ord_item_index:
                ord_item_index.append(feature[2])
                feature[2] = str(ord_item_index.index(feature[2]))
            else:
                feature[2] = str(ord_item_index.index(feature[2]))
            styles = []
            for style in feature[19].strip('[').strip(']').split(sep = ", "):
                styles.append(style.strip("'"))
            attributes = []
            for attribute in feature[20].strip('[').strip(']').split(sep = ", "):
                attributes.append(attribute.strip("'"))
            tags = []
            for tag in feature[28].strip().strip('[').strip(']').split(sep = ", "):
                tags.append(tag.strip("'"))
            
            row = [feature[14],feature[9],feature[12],feature[15],styles,feature[11],feature[10],feature[2],attributes,tags]
            csv_write.writerow(row)

def get_max(data):
    max_user_id = -1
    max_item_id = -1
    for row in data[1:]:
        cur_user = int(row.split('|')[0])
        cur_item = int(row.split('|')[7])
        if cur_user > max_user_id:
            max_user_id = cur_user
        if cur_item > max_item_id:
            max_item_id = cur_item
    return max_user_id+1 , max_item_id+1

def interaction_matrix(data):
    matrix = np.zeros((get_max(data)),dtype=int)
    for index , row in enumerate(data[1:]):
        print('building matrix now:' , index)
        user_id , item_id = int(row.split('|')[0]) , int(row.split('|')[7])
        matrix[user_id][item_id] = 1
    return matrix

def collect_style_attribute(data , tag_index):
    bag = []
    for index , row in enumerate(data[1:]):
        print('Collecting Style or Attribute Now!:' , index)
        feature = row.split('|')
        tag = feature[tag_index].strip('[').strip(']').split(', ')
        for i in tag:
            i = i.strip("'")
            if i not in bag:
                bag.append(i)
    return bag

def collect_tag(data):
    tags = []
    for index , row in enumerate(data[1:]):
        print('Collecting Tags Now!:' , index)
        feature = row.split('|')
        tag = feature[9].strip().strip('[').strip(']').split(', ')
        for i in tag:
            i = i.strip("'")
            if i not in tags:
                tags.append(i)
    return tags

def create_vector(item,item_list):
    item_vector = [0 for i in range(len(item_list))]
    for i in item:
        i = i.strip("'")
        try:
            item_vector[item_list.index(i)] = 1
        except:
            continue
    return item_vector

def Load_data(data):
    user_data = []
    item_data = []
    user_style =  sparse.csr_matrix((1,len(style_list)))
    item_attribute = sparse.csr_matrix((1,len(attribute_list)))
    item_tag = sparse.csr_matrix((1,len(tag_list)))
    for index , row in enumerate(data[1:]):
        print('Loading data now:', index)
        feature = row.split('|')
        style = feature[4].strip('[').strip(']').split(', ')
        style_vector = create_vector(style,style_list)
        
        user_style = sparse.vstack((user_style, sparse.csr_matrix(style_vector)))
        
        attribute = feature[8].strip('[').strip(']').split(', ')
        attribute_vector = create_vector(attribute,attribute_list)
        
        item_attribute = sparse.vstack((item_attribute, sparse.csr_matrix(attribute_vector)))
        
        tag = feature[9].strip().strip('[').strip(']').split(', ')
        tag_vector = create_vector(tag,tag_list)
        
        item_tag = sparse.vstack((item_tag, sparse.csr_matrix(tag_vector)))
        
        user_data.append({ "User_id": feature[0], "Age": feature[1], "Gender": feature[2], "Level":feature[3],
                          "Country": feature[5], "City": feature[6] })
        item_data.append( {"Item_id": feature[7] })
    return (user_data, item_data, user_style[1:], item_attribute[1:], item_tag[1:])

def user_item_vector_dic(data,user_data,item_data):
    user_dic = {}
    item_dic = {}
    for index, row in enumerate(data[1:]):
        feature = row.split('|')
        user_dic[int(feature[0])] = user_data[index]
        item_dic[int(feature[7])] = item_data[index]
        print('Creating user and item vector Now:', index)
    return user_dic, item_dic

def create_x_data(matrix, user_dic, item_dic):

    x_data = sparse.csr_matrix((1,(user_data.shape[1]+item_data.shape[1])))
    y_data = []
    for user_id, vector in enumerate(matrix):
        for item_id, element in enumerate(vector):
            x = sparse.hstack((user_dic[str(user_id)], item_dic[str(item_id)]))
            x_data = sparse.vstack((x_data, x))
            y_data.append(int(element))
            print('Creating X_data Now:', user_id , item_id)
    return x_data.tocsr(), np.array(y_data)

# =============================================================================
# def create_x_data(matrix, user_dic, item_dic):
#     x_data1 = sparse.csr_matrix((1,user_data.shape[1]))
#     x_data2 = sparse.csr_matrix((1,item_data.shape[1]))
#     y_data = []
#     for user_id, vector in enumerate(matrix):
#         for item_id, element in enumerate(vector):
#             x1 = user_dic[user_id]
#             x2 = item_dic[item_id]
#             x_data1 = sparse.vstack((x_data1, x1))
#             x_data2 = sparse.vstack((x_data2, x2))
#             y_data.append(element)
#             print('Creating X_data Now:', user_id , item_id)
#             
#     return x_data1[1:], x_data2[1:], np.array(y_data)
# 
# x_data1, x_data2, y_data = create_x_data(user_item_matrix, user_dic, item_dic)
# x_data = sparse.hstack((x_data1, x_data2)).tocsr()
# 
# =============================================================================

os.chdir('C:/Users/wcks1/Desktop/找教授/tree_enhanced_embedding_model-master/Data/Raw/London_Attractions_Complete_Review.csv')
raw_data = open('London_Attractions_Complete_Review.csv').readlines()
create_csv(raw_data, 'Data_Lon.csv')
data = open('Data_Lon.csv').readlines()

# Collect all styles, atttributes and tags
style_list = collect_style_attribute(data, 4).remove('')
attribute_list = collect_style_attribute(data, 8)
tag_list = collect_tag(data).remove('')

user_data, item_data, user_style, item_attribute, item_tag = Load_data(data)

v = DictVectorizer()
user_data = v.fit_transform(user_data)
item_data = v.fit_transform(item_data)

user_data = sparse.hstack((user_data, user_style)).tocsr()
item_data = sparse.hstack((item_data, item_attribute, item_tag)).tocsr()

user_dic, item_dic = user_item_vector_dic(data, user_data, item_data)
user_item_matrix = interaction_matrix(data)

# Create x_data and y_data
x_data , y_data = create_x_data(user_item_matrix, user_dic, item_dic)

# Save sparse matrix data
sparse.save_npz('user_data.npz', user_data)
sparse.save_npz('item_data.npz', item_data)
sparse.save_npz('x_data.npz', x_data)

# Set Factorizaion machine parameter
num_factors = 5
epoch = 200
learning_rate = 0.1

# Build model and Training
fm = pylibfm.FM(num_factors=num_factors, num_iter=epoch, verbose=True, task="classification", initial_learning_rate=learning_rate, learning_rate_schedule="optimal")
fm.fit(x_data,y_data)