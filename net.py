import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import pandas as pd

from torch.distributions import MultivariateNormal

import copy

NDAYS = 7

class TextCNN(nn.Module):
    
    def __init__(self,input_dim):
        
        super(TextCNN, self).__init__()
        
        self.input_dim = input_dim
        
        self.filter_size = [3,4,5]
        
        self.out_channels = [100,100,100]
        
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = input_dim, out_channels = o, kernel_size = k) for o,k in zip(self.out_channels,self.filter_size)])
    
    
    def forward(self,input):
        
        input = input.transpose(2,1)
      
        output = []
        
        for conv in self.convs:

            conv_input = conv(input)
                      
            pool = nn.MaxPool1d(kernel_size = conv_input.shape[-1])
            
            out = pool(conv_input)
            
            output.append(out.squeeze())
            
            
        output = torch.cat(output,dim = -1)
        
        
        return output
            
         
class LSTM(nn.Module):
    
    def __init__(self,hidden_input_dim,hidden_size):
        
        super(LSTM,self).__init__()     

        if torch.cuda.is_available():

           self.device = torch.device("cuda") 
        
        self.input_dim = hidden_input_dim
        
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size = self.input_dim, hidden_size = self.hidden_size ,batch_first = True).to(self.device)
        
    def forward(self,input):
        
        bs = input.shape[0]
        
        h0 = torch.zeros(1,bs,self.hidden_size).to(self.device)
        
        c0 = torch.zeros(1,bs,self.hidden_size).to(self.device)     
        
        output, (hn, cn) = self.lstm(input, (h0, c0))
        
        hn = hn.transpose(0,1)
        
        weight = torch.bmm(hn,output.transpose(1,2))
             
        softmax = torch.nn.Softmax(dim = 1)
        
        weight = softmax(weight)
        
        weight = weight.transpose(1,2)        
        
        out = weight * output
        
        out = torch.sum(out,dim=1)

        out = torch.cat([hn.squeeze(1),out], dim = 1)
        
               
        return out
        
class ActorPPO(nn.Module):
    
    def __init__(self,input_dim,state_dim,mid_dim,num_actions,hidden_size):

        action_std = 0.5
        
        super(ActorPPO,self).__init__()

        if torch.cuda.is_available():

           self.device = torch.device("cuda")
        
        self.combined_dim = 16 + 64 * num_actions
        
        self.action_dim = num_actions
        
        self.mid_dim = mid_dim       
        
        self.window_size = []
        
        self.hidden_input_dim = 300
        
        self.hidden_size = hidden_size
          
        self.TextEncoder = [TextCNN(input_dim).to(self.device) for i in range(num_actions)]
        
        self.LSTMEncoder = [LSTM(self.hidden_input_dim,hidden_size).to(self.device) for i in range(num_actions)]
        
        self.linearX = [nn.Linear(2 * hidden_size,hidden_size).to(self.device) for i in range(num_actions)]
        
        self.linearY = [nn.Linear(hidden_size,64).to(self.device) for i in range(num_actions)]
        
        self.drop = nn.Dropout(p = 0.1)
        
        self.relu = nn.ReLU()
        
        self.relu1 = nn.ReLU()
        
        self.linear1 = nn.Linear(state_dim,24)
                
        self.linear2 = nn.Linear(24,16)
        
        self.architecture = nn.Sequential(
           
             nn.Linear(self.combined_dim,self.mid_dim),
             nn.ReLU(),            
             nn.Linear(self.mid_dim,self.action_dim)
            
        )   
    
        self.action_var = nn.Parameter(torch.full((self.action_dim,), -action_std * action_std),requires_grad=True)

    
    def forward(self,stock_news,stock_feats):
        
        bs,nstocks,ndays,num_news,embed_dim = stock_news.shape
        
        stock_news = stock_news.permute(1,2,0,3,4)
        
        stock_text_out = torch.zeros(bs,nstocks,64).to(self.device)
        
        for s in range(nstocks):
            
            text_output = torch.zeros(ndays,bs,self.hidden_input_dim).to(self.device)
            
            for d in range(ndays):
                
                for b in range(bs):
                
                    news = stock_news[s,d,b].to(self.device)
                
                    news_embed = self.TextEncoder[s](news.unsqueeze(0))
                
                    text_output[d,b,:] = news_embed.squeeze()

                
            text_output = text_output.transpose(0,1)
            
            lstm_out = self.LSTMEncoder[s](text_output)
            
            lstm_out = self.drop(self.relu(self.linearX[s](lstm_out)))
            
            stock_text_out[:,s,:] = self.relu(self.linearY[s](lstm_out))
            
        
        stock_text_out = stock_text_out.contiguous().view(bs,-1)       

        stock_feats = stock_feats.to(self.device)
            
        stock_feats = self.relu1(self.linear1(stock_feats))    
        
        stock_feats = self.linear2(stock_feats)
            
        total_input = torch.cat((stock_text_out,stock_feats), dim = -1)
        
        mu = self.architecture(total_input).tanh()
        
        
        return mu,self.action_var.exp()

           
class Actor(nn.Module):
    
    def __init__(self,input_dim,state_dim,mid_dim,num_actions,hidden_size,device=torch.device("cuda")):
        
        super(Actor,self).__init__()

        if torch.cuda.is_available():

           self.device = torch.device("cuda")
        
        self.combined_dim = 16 + 64 * num_actions
        
        self.action_dim = num_actions
        
        self.mid_dim = mid_dim       
        
        self.window_size = []
        
        self.hidden_input_dim = 300
        
        self.hidden_size = hidden_size
        
        self.hidden_input_dim = 300
        
        self.TextEncoder = [TextCNN(input_dim).to(self.device) for i in range(num_actions)]
        
        self.LSTMEncoder = [LSTM(self.hidden_input_dim,hidden_size).to(self.device) for i in range(num_actions)]
        
        self.linearX = [nn.Linear(2 * hidden_size,hidden_size).to(self.device) for i in range(num_actions)]
        
        self.linearY = [nn.Linear(hidden_size,64).to(self.device) for i in range(num_actions)]
        
        self.drop = nn.Dropout(p = 0.1)
        
        self.relu = nn.ReLU()
        
        self.relu1 = nn.ReLU()
        
        self.linear1 = nn.Linear(state_dim,24)
                
        self.linear2 = nn.Linear(24,16)
        
        self.architecture = nn.Sequential(
           
             nn.Linear(self.combined_dim,self.mid_dim),
             nn.ReLU(),            
             nn.Linear(self.mid_dim,self.action_dim)
            
        )   
    
    
    
    def forward(self,stock_news,stock_feats):
        
        bs,nstocks,ndays,num_news,embed_dim = stock_news.shape

        stock_news = stock_news.permute(1,2,0,3,4)
        
        stock_text_out = torch.zeros(bs,nstocks,64).to(self.device)
        
        for s in range(nstocks):
            
            text_output = torch.zeros(ndays,bs,self.hidden_input_dim).to(self.device)
            
            for d in range(ndays):
                
                for b in range(bs):
                
                    news = stock_news[s,d,b].to(self.device)
                
                    news_embed = self.TextEncoder[s](news.unsqueeze(0))
                
                    text_output[d,b,:] = news_embed.squeeze()

                
            text_output = text_output.transpose(0,1)
            
            lstm_out = self.LSTMEncoder[s](text_output)
            
            lstm_out = self.drop(self.relu(self.linearX[s](lstm_out)))
            
            stock_text_out[:,s,:] = self.relu(self.linearY[s](lstm_out))
            
        

        
        stock_text_out = stock_text_out.contiguous().view(bs,-1)        

        stock_feats = stock_feats.to(self.device)
            
        stock_feats = self.relu1(self.linear1(stock_feats))    
        
        stock_feats = self.linear2(stock_feats)
 
        total_input = torch.cat((stock_text_out,stock_feats), dim = -1)
        
        output = self.architecture(total_input).tanh()
        
        return output
           
        
class Critic(nn.Module):
    
     def __init__(self,input_dim,state_dim,mid_dim,num_actions,hidden_size,device=torch.device("cuda")):
        
        super(Critic, self).__init__()

        if torch.cuda.is_available():

           self.device = torch.device("cuda")
        
        self.state_dim = state_dim
        
        self.action_dim = num_actions
        
        self.mid_dim = mid_dim
        
        self.window_size = []
        
        self.hidden_input_dim = 300
        
        self.hidden_size = hidden_size
        
        self.hidden_input_dim = 300
        
        self.TextEncoder = [TextCNN(input_dim).to(self.device) for i in range(num_actions)]
        
        self.LSTMEncoder = [LSTM(self.hidden_input_dim,hidden_size).to(self.device) for i in range(num_actions)]
        
        self.linearX = [nn.Linear(2 * hidden_size,hidden_size).to(self.device) for i in range(num_actions)]
        
        self.linearY = [nn.Linear(hidden_size,64).to(self.device) for i in range(num_actions)]
        
        self.drop = nn.Dropout(p = 0.1)
        
        self.relu = nn.ReLU()
        
        self.relu1 = nn.ReLU()
        
        self.relu2 = nn.ReLU()
        
        self.linear1 = nn.Linear(state_dim,24)
                
        self.linear2 = nn.Linear(24,16)
        
        self.combined_dim = 64 * num_actions  + 16
        
        self.linearc = nn.Linear(self.combined_dim,32)
        
        self.linear_ca = nn.Linear(32+self.action_dim,16)
        
        self.lineara1 = nn.Linear(16,self.mid_dim)
        
        self.lineara2 = nn.Linear(self.mid_dim,1)
        
       
        
    
     def forward(self,stock_news,stock_feats,action):
        
        
        bs,nstocks,ndays,num_news,embed_dim = stock_news.shape
        
        stock_news = stock_news.permute(1,2,0,3,4)
        
        stock_text_out = torch.zeros(bs,nstocks,64).to(self.device)
        
        for s in range(nstocks):
            
             text_output = torch.zeros(ndays,bs,self.hidden_input_dim).to(self.device)

             for d in range(ndays):
                    
                 for b in range(bs):
                
                      news = stock_news[s,d,b].to(self.device)

                      news_embed = self.TextEncoder[s](news.unsqueeze(0))
                    
                      text_output[d,b,:] = news_embed.squeeze()

                
             text_output = text_output.transpose(0,1)
              
             lstm_out = self.LSTMEncoder[s](text_output)
            
             lstm_out = self.drop(self.relu(self.linearX[s](lstm_out)))
                
             stock_text_out[:,s,:] = self.relu(self.linearY[s](lstm_out))
            
            
        stock_text_out = stock_text_out.contiguous().view(bs,-1)
        
        stock_feats = stock_feats.to(self.device)

        stock_feats = self.relu1(self.linear1(stock_feats))    
        
        stock_feats = self.linear2(stock_feats)
        
        stock_text_out = torch.cat((stock_text_out,stock_feats), dim = -1)
        
        stock_text_out = self.linearc(stock_text_out)

        action = action.to(self.device)
              
        combined_input = torch.cat((stock_text_out,action), dim = -1)
        
        combined_input  = self.linear_ca(combined_input)
        
        output = self.relu2(self.lineara1(combined_input))
        
        output = self.lineara2(output)
        
        return output    




class CriticPPO(nn.Module):
    
     def __init__(self,input_dim,state_dim,mid_dim,num_actions,hidden_size,device=torch.device("cuda")):
        
        super(CriticPPO, self).__init__()

        if torch.cuda.is_available():

           self.device = torch.device("cuda")
        
        self.state_dim = state_dim
        
        self.mid_dim = mid_dim
        
        self.window_size = []
        
        self.hidden_input_dim = 300
        
        self.hidden_size = hidden_size
        
        self.hidden_input_dim = 300
        
        self.TextEncoder = [TextCNN(input_dim).to(self.device) for i in range(num_actions)]
        
        self.LSTMEncoder = [LSTM(self.hidden_input_dim,hidden_size).to(self.device) for i in range(num_actions)]
        
        self.linearX = [nn.Linear(2 * hidden_size,hidden_size).to(self.device) for i in range(num_actions)]
        
        self.linearY = [nn.Linear(hidden_size,64).to(self.device) for i in range(num_actions)]
        
        self.drop = nn.Dropout(p = 0.1)
        
        self.relu = nn.ReLU()
        
        self.relu1 = nn.ReLU()
        
        self.relu2 = nn.ReLU()
        
        self.linear1 = nn.Linear(state_dim,24)
                
        self.linear2 = nn.Linear(24,16)
        
        self.combined_dim = 64 * num_actions  + 16
        
        self.linearc = nn.Linear(self.combined_dim,32)
        
        self.linear_ca = nn.Linear(32,16)
        
        self.lineara1 = nn.Linear(16,self.mid_dim)
        
        self.lineara2 = nn.Linear(self.mid_dim,1)
        
       
        
    
     def forward(self,stock_news,stock_feats):
        
        
        bs,nstocks,ndays,num_news,embed_dim = stock_news.shape
        
        stock_news = stock_news.permute(1,2,0,3,4)
        
        stock_text_out = torch.zeros(bs,nstocks,64).to(self.device)
        
        for s in range(nstocks):
            
             text_output = torch.zeros(ndays,bs,self.hidden_input_dim).to(self.device)

             for d in range(ndays):
                    
                 for b in range(bs):
                
                      news = stock_news[s,d,b].to(self.device)

                      news_embed = self.TextEncoder[s](news.unsqueeze(0))
                    
                      text_output[d,b,:] = news_embed.squeeze()

                
             text_output = text_output.transpose(0,1)
              
             lstm_out = self.LSTMEncoder[s](text_output)
            
             lstm_out = self.drop(self.relu(self.linearX[s](lstm_out)))
                
             stock_text_out[:,s,:] = self.relu(self.linearY[s](lstm_out))
            
            
        stock_text_out = stock_text_out.contiguous().view(bs,-1)
        
        stock_feats = stock_feats.to(self.device)

        stock_feats = self.relu1(self.linear1(stock_feats))    
        
        stock_feats = self.linear2(stock_feats)

        stock_text_out = torch.cat((stock_text_out,stock_feats), dim = -1)
        
        combined_input = self.linearc(stock_text_out)

        combined_input  = self.linear_ca(combined_input)
        
        output = self.relu2(self.lineara1(combined_input))
        
        output = self.lineara2(output)
        
        return output    

