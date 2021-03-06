import re
import numpy as np
from scipy import sparse
from pyfm import pylibfm
from sklearn.metrics import log_loss
from sklearn.feature_extraction import DictVectorizer

train_percentage = 0.8
num_factors = 2
epoch = 100
learning_rate = 0.01

raw_data = open('user_item_rating.csv').readlines()

# make English text clean
def clean_en_text(text):
    # keep English, digital and space
    comp = re.compile('[^A-Z^a-z^0-9^ ]')
    return comp.sub('', text)

def loadData(filename):
    data = []
    data_tag = []
    y = []
    for line in filename[1:]:
        feature = line.split('\t')
        label = feature[28][:-2].strip("[").split(sep = ", ")
        tag = [0 for j in range(len(tags))]
        for i in label:
            i = clean_en_text(i)
            try:
                tag[tags.index(i)] = 1
            except:
                continue
        data_tag.append(tag)
        data.append({ "user_id": str(feature[14]), "item_id": str(feature[2]),"Age": str(feature[9]), "Gender": str(feature[12]), "Level":str(feature[15]),
                      "Styles": str(feature[19]), "Country": str(feature[11]), "City": str(feature[10]),
                      "Attributes": str(feature[20])})
        y.append(float(feature[25]))

    return (data, np.array(y), np.array(data_tag))

def rating_to_binary(array):
    for i , rating in enumerate(array):
        if rating >= 4:
            array[i] = 1
        else:
            array[i] = 0
        
    return array

tags = []
for i in raw_data[1:]:
    text = i.split('\t')
    tag = text[28][:-2].strip("[").split(sep = ", ")
    for j in tag:
        j = clean_en_text(j)
        if j not in tags:
            tags.append(j)

(x, y, data_tag) = loadData(raw_data)

v = DictVectorizer()
x = v.fit_transform(x)

train_data = np.zeros((x.shape[0],x.shape[1]+data_tag.shape[1]))

for i , feature  in enumerate(x.A):
    train_data[i] = np.concatenate([feature,data_tag[i]])
x = sparse.csr_matrix(train_data)
y = rating_to_binary(y)

fm = pylibfm.FM(num_factors=num_factors, num_iter=epoch, verbose=True, task="classification", initial_learning_rate=learning_rate, learning_rate_schedule="optimal")
fm.fit(x[0:int(train_percentage*x.shape[0])],y[0:int(train_percentage*len(y))])

preds = fm.predict(x[int(train_percentage*x.shape[0]):x.shape[0]-1])

print("Test_data Log Loss: %.4f" % log_loss(y[int(train_percentage*x.shape[0]):x.shape[0]-1],preds))
