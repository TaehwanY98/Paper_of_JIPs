import pickle
from torch import Tensor, where
from sklearn.preprocessing import StandardScaler
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
        
        # label = list(map(f, label))
        
        ret ={
            "x": Tensor(X),
            "label": where(Tensor(label)>2, 1.0, 0.0)
        }
        return ret
    def __len__(self):
        return len(self.files)
    