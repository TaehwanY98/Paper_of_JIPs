from torch import nn, stack, float16, float64, float32, int64
import torch
from torch.autograd import Variable

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GRU(nn.Module) :
   def __init__(self,input_dim,hidden_dim,output_dim,bias=True) :
       super(GRU, self).__init__()
       self.input_dim = input_dim
       self.hidden_dim = hidden_dim
       self.GRU = nn.GRUCell(input_dim, hidden_size=hidden_dim)
       self.fc = nn.Sequential(nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid())
       
       
   def forward(self, x) :
       out = self.GRU(x)
       out = self.fc(out)
       return out
