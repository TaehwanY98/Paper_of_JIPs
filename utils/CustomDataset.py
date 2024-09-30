import pickle
from torch import Tensor, where
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
# from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
# import keras
# from sklearn.model_selection import train_test_split
# def f(x):
#     lis = [0,0,0,0,0,0,0,0]
#     lis[x]=1
#     return lis

class WESADDataset(object):
    def __init__(self, pkl_files:list, test_mode=False) -> None:
        self.files = []
        for file in pkl_files:
            with open(file, "rb") as fil:
                self.files.append(pickle.load(fil,encoding="latin1"))
        self.test_mode = test_mode
    def Normalization(self, df):
        standard_scaler = StandardScaler()
        return standard_scaler.fit_transform(df)
        
            
    def __getitem__(self, i):
        self.file = self.files[i]
        if self.test_mode:
            ACC=self.file['signal']['chest']["ACC"][:10, 0]
            label= self.file['label'][:10]
            EDA=self.file['signal']['chest']['EDA'][:10]
            Temp=self.file['signal']['chest']['Temp'][:10]
        else:
            ACC=self.file['signal']['chest']["ACC"][:, 0]
            label= self.file['label']
            EDA=self.file['signal']['chest']['EDA']
            Temp=self.file['signal']['chest']['Temp']
        
        X=self.Normalization([(float(acc),float(eda), float(temp)) for acc , eda, temp in zip(ACC, EDA, Temp)])
        X= Tensor(X)
        Y = where(Tensor(label)>2, 1.0, 0.0)
        # label = list(map(f, label))
        
        ret ={
            "x": X,
            "label": Y
        }
        return ret
    def __len__(self):
        return len(self.files)
    
class K_EMODataset(object):
    def __init__(self, data_dir, label_dir) -> None:
        self.clients = os.listdir(data_dir)
        self.EDA = [pd.read_csv(os.path.join(data_dir,path,"E4_EDA.csv")) for path in self.clients]
        self.ACC = [pd.read_csv(os.path.join(data_dir,path,"E4_ACC.csv")) for path in self.clients]
        self.Temp = [pd.read_csv(os.path.join(data_dir,path,"E4_TEMP.csv")) for path in self.clients]
        self.emo = [pd.read_csv(os.path.join(label_dir,'P'+path+".self.csv")) for path in self.clients]
    def Normalization(self, df):
        standard_scaler = StandardScaler()
        return standard_scaler.fit_transform(df)
    def __getitem__(self, i):
        emo = self.emo[i]
        EDA = self.EDA[i]
        ACC = self.ACC[i]
        Temp = self.Temp[i]
        emo = emo["arousal"].values
    
        label = where(Tensor(emo)>2, 1.0, 0.0)
        ACC = ACC["x"].values
        EDA = EDA["value"].values
        Temp = Temp["value"].values
        ACClength = ACC.shape[0]//emo.shape[0]
        EDAlength = EDA.shape[0]//emo.shape[0]
        Templength = Temp.shape[0]//emo.shape[0]
        ACC = [np.average(ACC[step*ACClength:(step+1)*ACClength]) for step in range(0, emo.shape[0])]
        EDA = [np.average(EDA[step*EDAlength:(step+1)*EDAlength]) for step in range(0, emo.shape[0])]
        Temp = [np.average(Temp[step*Templength:(step+1)*Templength]) for step in range(0, emo.shape[0])]
        X=self.Normalization(np.nan_to_num(np.array([(float(acc),float(eda), float(temp)) for acc , eda, temp in zip(ACC, EDA, Temp)])))
        X= Tensor(X)
        ret ={
            "x": X,
            "label": label
        }
        return ret
    def __len__(self):
        return len(self.clients)