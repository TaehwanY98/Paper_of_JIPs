from torch import nn
from torch.utils.data import DataLoader
import torch
from utils import *
from Network import *
import numpy as np
from Old_train import valid
import os
import warnings
def evaluate(net, testloader, lossf, DEVICE):
    net.eval()
    history = {'loss': [], 'acc': [], 'precision': [], 'f1score': [], "recall": []}
    with torch.no_grad():
        for key, value in valid(net, testloader, 0, lossf, DEVICE).items():
            history[key].append(value)
    return history

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    net = K_emo_GRU(3, 4, 1)
    args = Evaluateparaser()
    net.load_state_dict(torch.load(f"./Models/{args.version}/net.pt", weights_only=False))
    net.to(DEVICE)
    lossf = nn.BCEWithLogitsLoss()
    test_ids = [id for id  in os.listdir(os.path.join(args.wesad_path, "test")) if not "Cluster" in id]
    test_data = K_EMODataset("./data/e4_data/e4_data/test", "./data/emotion_annotations/emotion_annotations/self_annotations")
    test_loader=DataLoader(test_data, 1, shuffle=False, collate_fn=lambda x:x)
    
    history = evaluate(net, test_loader,lossf, DEVICE)