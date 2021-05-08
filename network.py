import torch 
from torch import nn 

class LinearNetwork_Saeed(nn.Module):
   def __init__(self):
      super(LinearNetwork_Saeed, self).__init__()
      self.linear1 = nn.Linear(120, 200)
      self.linear2 = nn.Linear(200, 400)
      self.linear3 = nn.Linear(400, 100)
      self.linear4 = nn.Linear(100, 4)
      self.activation = nn.ReLU()
      
   def forward(self, x):
      x = self.activation(self.linear1(x))
      x = self.activation(self.linear2(x))
      x = self.activation(self.linear3(x))
      x = self.linear4(x)

      return x 

   def init_weights(self, x):
      if type(x) == nn.Linear:
         nn.init.ones_(x.weight)

         if x.bias is not None:
            nn.init.zeros_(x.bias)


class LinearNetwork_Eric(nn.Module):
   def __init__(self):
      super(LinearNetwork_Eric, self).__init__()
      self.linear1 = nn.Linear(125, 400)
      self.linear2 = nn.Linear(400, 200)
      self.linear3 = nn.Linear(200, 100)
      self.linear4 = nn.Linear(100, 4)
      self.activation = nn.ReLU()

   def forward(self, x):
      x = self.activation(self.linear1(x))
      x = self.activation(self.linear2(x))
      x = self.activation(self.linear3(x))
      x = self.linear4(x)

      return x

   def init_weights(self, x):
      if type(x) == nn.Linear:
         nn.init.ones_(x.weight)

         if x.bias is not None:
            nn.init.zeros_(x.bias)


class LinearNetwork_Quoc(nn.Module):
   def __init__(self):
      super(LinearNetwork_Quoc, self).__init__()
      self.linear1 = nn.Linear(122, 200)
      self.linear2 = nn.Linear(200, 400)
      self.linear3 = nn.Linear(400, 100)
      self.linear4 = nn.Linear(100, 4)
      self.activation = nn.ReLU()

   def forward(self, x):
      x = self.activation(self.linear1(x))
      x = self.activation(self.linear2(x))
      x = self.activation(self.linear3(x))
      x = self.linear4(x)

      return x

   def init_weights(self, x):
      if type(x) == nn.Linear:
         nn.init.ones_(x.weight)

         if x.bias is not None:
            nn.init.zeros_(x.bias)
