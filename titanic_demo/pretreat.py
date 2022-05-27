import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

'''
vscode中的jupyter插件 不知道为什么无法在jupyter文件中对pretreat进行import
所以本文件暂时作废. 
'''


def get_pretreated_training_data(loc ='./data/titanic/train.csv', get_y=False):
    # get Title feature
    training = pd.read_csv(loc)
    training['n_new']= training['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    training['Title'] = 0 
    training.loc[training["n_new"] == 'Mr', 'Title'] = 1
    training.loc[training["n_new"] == 'Miss', 'Title'] = 4
    training.loc[training["n_new"] == 'Mrs', 'Title'] = 5
    training.loc[training["n_new"] == 'Master', 'Title'] = 3
    training.loc[training["n_new"] == 'Dr', 'Title'] = 2

    # get whether has cabin info feature
    C = training.Cabin[training.Cabin.isna()]
    C_not = training.Cabin[training.Cabin.notna()]
    C.values[:] = 0
    C_not.values[:] = 1
    cabine_not = pd.concat([C, C_not]).sort_index()
    training['cabine_n'] = cabine_not
    training.cabine_n = training.cabine_n.astype(int)
    
    training['sp'] = training.SibSp + training.Parch

    training.drop(["n_new", "Name", "PassengerId","Ticket","Cabin"], inplace = True, axis = 1 )

    training['IsAlone'] = 1
    training['IsAlone'].loc[training['sp'] > 0] = 0
    training = pd.get_dummies(training)

    from sklearn.experimental import enable_iterative_imputer # do not remove
    from sklearn.impute import IterativeImputer
    imp = IterativeImputer(max_iter=10, random_state=0)
    x = imp.fit_transform(training)
    training = pd.DataFrame(x, columns = training.columns)
    save_col = [ 'Pclass', 'SibSp', 'Parch', 'Title', 'sp',
        'cabine_n', 'IsAlone', 'Sex_female', 'Sex_male', 
        'Embarked_C', 'Embarked_S', 'Embarked_Q']
    training[save_col] = training[save_col].astype(int)

    if get_y:
        y_train = training["Survived"]
        training = training.drop(['Survived'], axis = 1)
        return training, y_train

    return training


def get_pretreated_test_data(loc='./data/titanic/test.csv'):
    '''
    titanic seems do not have ture label.
    so x_test do not have column "Survived".
    return: x_test
    '''
    x_test = pd.read_csv(loc)
    x_test['n_new']= x_test['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    x_test['Title'] = 0
    x_test.loc[x_test["n_new"] == 'Mr', 'Title'] = 1
    x_test.loc[x_test["n_new"] == 'Miss', 'Title'] = 4
    x_test.loc[x_test["n_new"] == 'Mrs', 'Title'] = 5
    x_test.loc[x_test["n_new"] == 'Master', "Title"] = 3
    x_test.loc[x_test["n_new"] == 'Dr', 'Title'] = 2

    C = x_test.Cabin[x_test.Cabin.isna()]
    C_not = x_test.Cabin[x_test.Cabin.notna()]
    C.values[:] = 0
    C_not.values[:] = 1
    cabine_not = pd.concat([C, C_not]).sort_index()
    x_test['cabine_n'] = cabine_not

    x_test['sp'] = x_test.SibSp + x_test['Parch']
    x_test.cabine_n = x_test.cabine_n.astype(int)

    x_test['IsAlone'] = 1
    x_test['IsAlone'].loc[x_test['sp'] > 0] = 0

    x_test.drop(["Name", "Ticket", "n_new", "Cabin"], inplace = True, axis = 1 )
    x_test = pd.get_dummies(x_test)
    return x_test

def get_dataloader(x_train: pd.DataFrame, input_dict:list, bs:int=32, drop_last=False):
    '''
    input: x_train with "Survived"
    '''
    assert "Survived" in x_train.columns, "x_train must contain 'Survived'"
    X_train = x_train[input_dict].to_numpy()
    Y_train = x_train["Survived"].to_numpy()

    x_train=torch.from_numpy(X_train).float()
    y_train=torch.from_numpy(Y_train).long()
    dataset=TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(dataset,batch_size=bs, shuffle=True,drop_last=drop_last)
    return loader

class MLP(torch.nn.Module):
    layers: torch.nn.Module
    def __init__(self, input_dim: int, hidden_dims:int):
        super().__init__()
        layers = []
        i = input_dim
        for o in hidden_dims:
            layers += [
                torch.nn.Linear(i, o),
                torch.nn.ReLU(inplace=True),
            ]
            i = o
        layers += [
            torch.nn.Linear(i, 2),
            torch.nn.Softmax(),
        ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

def train(model, loader, comment, model_name, input_dic, epoches=2000, lr=0.001):
    loss_f = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    tb = SummaryWriter(comment=comment)
    loss_cur, acc_cur = [], []
    for epoch in range(epoches):
        sum_loss, sum_acc = 0, 0
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            outputs=model(x)
            optimizer.zero_grad()
            loss=loss_f(outputs, y)
            loss.backward()
            optimizer.step()
            _,id=torch.max(outputs.data,1)
            acc=torch.sum(id==y.data)
            sum_loss += loss.item()
            sum_acc += acc.item()    
        tb.add_scalar('loss', sum_loss, epoch)
        tb.add_scalar('acc', sum_acc, epoch)
        loss_cur.append(sum_loss)
        acc_cur.append(sum_acc)
        if epoch % 500 == 0:
            print("epoch:{}, loss:{}, acc:{}".format(epoch, sum_loss, sum_acc/891.))
    save_dir = tb.get_logdir()
    save_dic = {'input': input_dic, 'lr':lr, 'para':model.state_dict()}
    torch.save(save_dic, os.path.join(save_dir, model_name))
    np.savetxt(os.path.join(save_dir, 'acc_cur'), acc_cur)
    np.savetxt(os.path.join(save_dir, 'loss_cur'), loss_cur)
    return acc_cur, loss_cur
