import torch
import torch.nn as nn
from torchtext import data, datasets
from torchtext.vocab import Vectors
import torch.nn.functional as F
import torch.optim as optim
from model import TextRNN
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


my_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
raw_train = pd.read_csv('train.csv', usecols=['id', 'text', 'target'])
test = pd.read_csv('test.csv', usecols=['id', 'text'])
train, cv = train_test_split(raw_train, test_size=0.25)

train.to_csv('train1.csv', index=False)
cv.to_csv('validation.csv', index=False)
test.to_csv('test2.csv', index=False)
'''
# spacy 英文分词
spacy_en = spacy.load('en')
# 引入停用词


# 分词
def tokenizer(text):
     return [x.text for x in spacy_en.tokenizer(text)]


# Field
PID = data.Field(sequential=False, use_vocab=False)
KEY = data.Field(sequential=False, use_vocab=False)
TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
LABEL = data.Field(sequential=False, use_vocab=False)

# 定义数据格式
train_x, validation = data.TabularDataset.splits(
    path='.', train='train1.csv', validation='validation.csv', format='csv', skip_header=True,
    fields=[('id', PID), ('text', TEXT), ('target', LABEL)]
)

test_x = data.TabularDataset(
    'test2.csv', format='csv', skip_header=True,
    fields=[('id', PID), ('text', TEXT)]
)
# 建立vocab
word_vec = Vectors('./.vector_cache/glove.6B.100d.txt', cache='./.vector_cache')
TEXT.build_vocab(train_x, vectors=word_vec)  # , max_size=3000
# 当 corpus 中有的 token 在 vectors 中不存在时 的初始化方式.
TEXT.vocab.vectors.unk_init = nn.init.xavier_uniform

# 构建迭代器 获取batch输入
train_iter = data.BucketIterator(train_x, batch_size=64, shuffle=True,
                                 sort_key=lambda x: len(x.text), device=my_dev)
val_iter = data.BucketIterator(validation, batch_size=64, shuffle=True,
                               sort_key=lambda x: len(x.text), device=my_dev)
# 在 test_iter , sort一定要设置成 False, 要不然会被 torchtext 搞乱样本顺序
test_iter = data.Iterator(dataset=test_x, batch_size=64, train=False,
                          sort=False, device=my_dev)
    

def my_entropy(target, out):
    return -target * torch.log(out) - (1-target) * torch.log(1-out)


len_vocab = len(TEXT.vocab)
model = TextRNN(len_vocab)
model.embedding.weight.data.copy_(TEXT.vocab.vectors)
model.to(my_dev)  # gpu

# 训练
optimizer = optim.Adam(model.parameters(), lr=0.001)

# epoch: 所有batch循环训练一遍为1 epoch
epochs = 1
best_val_acc = 0
for epoch in range(epochs):
    for index, batch in enumerate(train_iter):
        train_text = batch.text
        train_label = batch.target
        # 转换为ont-hot
        tmp = torch.sparse.torch.eye(2).index_select(dim=0, index=train_label.cpu().data)
        target = tmp.to(my_dev)
        train_texts = train_text.permute(1, 0)

        # 初始化梯度
        optimizer.zero_grad()
        out = model(train_texts)

        # entropy = nn.CrossEntropyLoss()
        # loss = entropy(out, target)
        loss = my_entropy(target, out)
        loss = loss.sum(-1).mean()
        loss.backward()
        optimizer.step()

        # 记录损失下降
        if (index + 1) % 20 == 0:
            _, y_pre = torch.max(out, -1)
            acc = torch.mean((torch.tensor(y_pre == batch.target, dtype=torch.float)))
            print('epoch: %d \t batch_idx : %d \t loss: %.4f \t train acc: %.4f' % (epoch, index, loss, acc))

    val_accs = []
    for batch_idx, batch in enumerate(val_iter):
        data = batch.text
        target = batch.target
        target = torch.sparse.torch.eye(2).index_select(dim=0, index=target.cpu().data)
        target = target.to(my_dev)
        data = data.permute(1, 0)
        out = model(data)

        _, y_pre = torch.max(out, -1)
        acc = torch.mean((torch.tensor(y_pre == batch.target, dtype=torch.float)))
        val_accs.append(acc)

    acc = np.array(val_accs).mean()
    if acc > best_val_acc:
        print('val acc : %.4f > %.4f saving model' % (acc, best_val_acc))
        # torch.save(lstm_model.state_dict(), 'params.pkl')  # 此为仅保存参数
        torch.save(model, 'is_real.pkl')
        best_val_acc = acc
    print('val acc: %.4f' % (acc))

print(model)

# 预测
# len_vocab = len(text.vocab)
# model = TextRNN(len_vocab)
model = torch.load('is_real.pkl')
df = pd.DataFrame(columns=['id', 'target'])
df.to_csv('Submission.csv', index=False)

# 预测
for batch in test_iter:
    batch.text = batch.text.permute(1, 0)
    preds = model(batch.text)
    # print(batch.text)
    # 我们想要求每一行最大的列标号，我们就要指定dim=1，表示我们不要列了，保留行的size就可以了
    preds = torch.argmax(preds, dim=1)
    pid = batch.id.cpu().numpy()
    target = preds.cpu().numpy()
    res = list(zip(pid, target))
    df = pd.DataFrame(res)
    # print(df)

    df.to_csv('Submission.csv', mode='a+', header=False, index=False)
