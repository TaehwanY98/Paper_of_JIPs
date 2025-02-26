from torch import nn,Tensor, stack, int64,float32, float64, argmax, save
from torch.utils.data import DataLoader
from torch.optim import SGD
import torch
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from utils import *
from Network import *
import numpy as np
import warnings
import random
import os
from tqdm import tqdm
def make_model_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        pass

def train(net, train_loader, valid_loader, epoch, lossf, optimizer, DEVICE, save_path):
    history = {'loss': [], 'acc': [], 'precision': [], 'f1score': [], "recall": []}
    for e in range(epoch):
        net.train()
        for sample in tqdm(train_loader, desc="train: "):
            X =  torch.stack([s["x"] for s in sample], dim=0)
            Y = torch.stack([s["label"] for s in sample], dim=0).unsqueeze(-1)
            out = net(X.type(float32).to(DEVICE))
            # print(out.size())
            loss = lossf(out.type(float32).to(DEVICE), Y.type(float32).to(DEVICE))
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        if valid_loader is not None:
            net.eval()
            print("valid start")
            with torch.no_grad():
                for key, value in valid(net, valid_loader, e, lossf, DEVICE).items():
                    history[key].append(value)
        if save_path is not None:            
            save(net.state_dict(), f"./Models/{save_path}/net.pt")
    if valid_loader is not None:                    
        return history
    else:
        return None
    
def valid(net, valid_loader, e, lossf, DEVICE):
    acc=0
    precision=0
    f1score=0
    recall=0
    length = len(valid_loader)
    loss=0
    for sample in tqdm(valid_loader, desc="validation:"):
        X =  torch.stack([s["x"] for s in sample])
        Y = torch.stack([s["label"] for s in sample]).unsqueeze(-1)
        out = net(X.type(float32).to(DEVICE))
        loss += lossf(out.type(float32).to(DEVICE), Y.type(float32).to(DEVICE))
        out = torch.where(out>0.6, 1.0, 0.0)
        acc+= accuracy_score(Y.cpu().squeeze().detach().numpy(), out.cpu().squeeze().detach().numpy())
        precision+= precision_score(Y.cpu().squeeze().detach().numpy(), out.cpu().squeeze().detach().numpy(), average='macro')
        f1score += f1_score(Y.cpu().squeeze().detach().numpy(), out.cpu().squeeze().detach().numpy(), average="macro")
        recall += recall_score(Y.cpu().squeeze().detach().numpy(), out.cpu().squeeze().detach().numpy(), average="macro")
    if e is not None:
        print(f"Result epoch {e+1}: loss:{loss.item()/length: .4f} acc:{acc/length: .4f} precision:{precision/length: .4f} f1score:{f1score/length: .4f} recall: {recall/length: .4f}")
        
    return {'loss': loss.item()/length, 'acc': acc/length, 'precision': precision/length, 'f1score': f1score/length, "recall": recall/length}

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    print("==== Centralized Learning by using WESAD Dataset ====")
    args = Centralparser()
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    make_model_folder(f"./Models/{args.version}")
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    
    lossf = nn.BCEWithLogitsLoss().to(DEVICE)
    net = K_emo_GRU(3, 4, 1)
    if args.pretrained is not None:
        net.load_state_dict(torch.load(args.pretrained, weights_only=True))
    net.to(DEVICE)
    optimizer = SGD(net.parameters(), lr=args.lr)
    
    print("==== LSTM 모델 ====")
    print(net)
    
    print("==== Loss ====")
    print(lossf.__class__.__name__)
    print("==== Args ====")
    print(f"seed value: {args.seed}")
    print(f"epoch number: {args.epoch}")
    print(f"K-emo Data dir: {args.wesad_path}")
    train_data = K_EMODataset("./data/e4_data/e4_data/train", "./data/emotion_annotations/emotion_annotations/self_annotations")
    test_data = K_EMODataset("./data/e4_data/e4_data/test", "./data/emotion_annotations/emotion_annotations/self_annotations")
    valid_data = K_EMODataset("./data/e4_data/e4_data/valid", "./data/emotion_annotations/emotion_annotations/self_annotations")

    print("==== Data Information ====")
    print("train:", train_data.clients)
    print("test:", test_data.clients)
    print("valid:", valid_data.clients)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn= lambda x:x)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, collate_fn= lambda x:x)
    print("==== Training ====")
    history = train(net, train_loader, valid_loader, args.epoch, lossf, optimizer, DEVICE, args.version)