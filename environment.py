import math
import torch.nn.functional as F
import copy
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import copy

EMBED_DIM = 768
import math
import torch.nn.functional as F
import copy
import torch.nn as nn
import torch

import numpy as np
import pandas as pd
import copy

class StockTradeEnvironment():
    
    def __init__(self,num_actions,df,df_eval,df_test,tech,news_tensor,date_index,turbulence_threshold = 100):
        
        super(StockTradeEnvironment, self).__init__()
        
        self.df = df
        
        self.df_eval = df_eval
        
        self.df_test = df_test
               
        self.NUM_STOCKS = num_actions
        
        self.budget = 1000000

        self.tech = tech
        
        self.portfolio_state = np.zeros([1,self.NUM_STOCKS * (len(tech) + 2) + 1])
               
        self.eval_portfolio_state = np.zeros([1,self.NUM_STOCKS * (len(tech) + 2) + 1]) 
        
        self.test_portfolio_state = np.zeros([1,self.NUM_STOCKS * (len(tech) + 2) + 1]) 
       
        self.datelist = df.index.unique().tolist()       
        
        self.eval_datelist = df_eval.index.unique().tolist()
        
        self.test_datelist = df_test.index.unique().tolist()
        
        self.eval_initial_state(0)
        
        self.test_initial_state(0)
               
        self.initial_state(0)
                
        self.maxShare = 1e2
        
        self.turbulence_threshold = turbulence_threshold
         
        self.turbulence = 0

        self.tech = tech
        
        self.date_To_day = date_index

        self.news_tensor = news_tensor
        
        self.day_To_date = {} 
        
        self.align_day_to_date()

       
    def align_day_to_date(self):

        d_list = self.df.date.unique().tolist()

        for _ in range(len(d_list)):

            self.day_To_date[_] = d_list[_]

        d_e_list = self.df_eval.date.unique().tolist()

        for _ in range(len(d_e_list)):

            self.day_To_date[_+ len(d_list)] = d_e_list[_]
     
        d_t_list = self.df_test.date.unique().tolist()

        for _ in range(len(d_t_list)):

            self.day_To_date[_+ len(d_list) + len(d_e_list)] = d_t_list[_]


    def get_news(self,batch_start_date):

        bs = len(batch_start_date)

        stock_news = torch.zeros(bs,self.NUM_STOCKS,7,30,EMBED_DIM)

        for _ in range(len(batch_start_date)):

            t = batch_start_date[_]

            date = self.day_To_date[t]
            
            if date not in self.date_To_day:
                
                return stock_news

            day = self.date_To_day[date]

            for i in range(7):
                
                if day-i-1 >= 0:
  
                   stock_news[_,:,i,:,:] = self.news_tensor[:,day-i-1,:,:]

        return stock_news
    
         
    def eval_initial_state(self,date):
        
        date = self.eval_datelist[date]
       
        self.eval_portfolio_state[0][0] = self.budget
        
        self.eval_portfolio_state[0][1:1+self.NUM_STOCKS] = self.df_eval.loc[date]["close"].values
        
        for i,tech in enumerate(self.tech):
        
              self.eval_portfolio_state[0,1 + (i+2) * self.NUM_STOCKS: 1 + (i+3) *self.NUM_STOCKS] = self.df_eval.loc[date][tech].values
        
        self.turbulence = self.df_eval.loc[0]["turbulence"].values[0]
        
    def test_initial_state(self,date):
        
        date = self.test_datelist[date]
       
        self.test_portfolio_state[0][0] = self.budget
        
        self.test_portfolio_state[0][1:1+self.NUM_STOCKS] = self.df_test.loc[date]["close"].values
        
        for i,tech in enumerate(self.tech):
        
            self.test_portfolio_state[0,1 + (i+2) * self.NUM_STOCKS: 1 + (i+3) *self.NUM_STOCKS] = self.df_test.loc[date][tech].values
     
        self.turbulence = self.df_test.loc[0]["turbulence"].values[0]
        
    def initial_state(self,date):
        
        date = self.datelist[date]
        
        self.portfolio_state[0][0] = self.budget
        
        self.portfolio_state[0][1:1+self.NUM_STOCKS] = self.df.loc[date]["close"].values
        
        for i,tech in enumerate(self.tech):
        
              self.portfolio_state[0,1 + (i+2) * self.NUM_STOCKS: 1 + (i+3) *self.NUM_STOCKS] = self.df.loc[date][tech].values

        self.turbulence = self.df.loc[0]["turbulence"].values[0]
        
        
    def step_(self,actions,t,T,is_eval = False,is_test = False):
           
        prev_state = self.get_state(is_eval,is_test).copy()
 
        self.make_transaction(actions,is_eval,is_test)

        self.update(t+1,is_eval,is_test)

        reward = self.CalculateReward(prev_state,is_eval,is_test)

        dones = 1 if t == T - 2 else 0 

        next_state = self.get_state(is_eval,is_test).copy() if ( t < T - 2 or is_eval or is_test) else self.reset_()
            
        next_time = 0 if t == T - 2 else t+1 
        
        return next_state,reward,dones,next_time

    
    def make_transaction(self,actions,is_eval,is_test):
               
        actions = actions * self.maxShare
        
        sell_index = copy.deepcopy(actions)
        
        buy_index = copy.deepcopy(actions)
    
        
        buy_index[buy_index < 0] = 0
        
        sell_index[sell_index > 0] = 0
        
        
        self.sell_stock(sell_index.astype(int),is_eval,is_test)
        
        self.buy_stock(buy_index.astype(int),is_eval,is_test)
        
    
    def CalculateReward(self,prev_State,is_eval,is_test):

        state = self.get_state(is_eval,is_test)

        new_value = state[0,0] + np.sum(state[0,1:1+self.NUM_STOCKS] * state[0,1+self.NUM_STOCKS:1+ 2 * self.NUM_STOCKS])
        
        old_value = prev_State[0,0] + np.sum(prev_State[0,1:1+self.NUM_STOCKS] * prev_State[0,1+self.NUM_STOCKS:1+ 2 * self.NUM_STOCKS])
        
        return new_value - old_value
    
    
    def update(self,date,is_eval,is_test):
        
        state = self.test_portfolio_state if is_test else (self.eval_portfolio_state if is_eval else self.portfolio_state)

        df = self.df_test if is_test else (self.df_eval if is_eval else self.df)

        date = self.datelist[date]
        
        state[0,1:1+self.NUM_STOCKS] = df.loc[date]["close"].values
        
        for i,tech in enumerate(self.tech):
        
              state[0,1 + (i+2) * self.NUM_STOCKS: 1 + (i+3) *self.NUM_STOCKS] = df.loc[date][tech].values           
            
        self.turbulence = df.loc[date]["turbulence"].values[0]
    
    def sell_stock(self,sell_index,is_eval,is_test):
        
        if self.turbulence > self.turbulence_threshold:
            
           state = self.get_state(is_eval,is_test)
        
           share_sell_quantity = state[0,1 + self.NUM_STOCKS : 1 + 2 * self.NUM_STOCKS].copy()
            
           state[0,1 + self.NUM_STOCKS : 1 + 2 * self.NUM_STOCKS] -= share_sell_quantity
        
           state[0,0] += np.sum(share_sell_quantity * state[0,1 : 1 + self.NUM_STOCKS])
        
        else:

           state = self.get_state(is_eval,is_test)

           share_sell_quantity = np.minimum(-sell_index, state[0,1 + self.NUM_STOCKS : 1 + 2 * self.NUM_STOCKS])

           state[0,1 + self.NUM_STOCKS : 1 + 2 * self.NUM_STOCKS] -= share_sell_quantity.squeeze()
        
           state[0,0] += np.sum(share_sell_quantity.squeeze() * state[0,1 : 1 + self.NUM_STOCKS])


    def buy_stock(self,buy_index,is_eval,is_test):
        
        state = self.get_state(is_eval,is_test)
        
        if self.turbulence > self.turbulence_threshold:
                
            pass

        for i in range(len(buy_index)):
           
            if(state[0,0] / state[0,1+i] >= buy_index[i]):
                
                state[0,0] -= state[0,1+i] * buy_index[i]
                
                state[0,1+self.NUM_STOCKS+i] += buy_index[i]            
            
            else:
                
                buy_quantity = math.floor(state[0,0] / state[0,1+i])
                
                state[0,0] -= state[0,1+i] * buy_quantity
                
                state[0,1+self.NUM_STOCKS+i] += buy_quantity
                
    def reset_(self):
        
        self.reset()
            
        self.initial_state(0)
        
        next_state = self.get_state(is_eval = False,is_test = False).copy()
            
        return next_state

    def eval_reset_(self):
        
        self.eval_reset()
            
        self.eval_initial_state(0)
        
        next_state = self.get_state(is_eval = True,is_test = False).copy()
                     
        return next_state
    
    def test_reset_(self):
        
        self.test_reset()
            
        self.test_initial_state(0)
        
        next_state = self.get_state(is_eval = False,is_test = True).copy().squeeze()
                     
        return next_state
    
    def reset(self):

        self.portfolio_state[0,1 + self.NUM_STOCKS : 1 + 2 * self.NUM_STOCKS] = 0

    def eval_reset(self):

        self.eval_portfolio_state[0,1 + self.NUM_STOCKS : 1 + 2 * self.NUM_STOCKS] = 0
        
    def test_reset(self):

        self.test_portfolio_state[0,1 + self.NUM_STOCKS : 1 + 2 * self.NUM_STOCKS] = 0
                            
    def get_state(self,is_eval,is_test):
 
        return self.test_portfolio_state if is_test else (self.eval_portfolio_state if is_eval else self.portfolio_state)
                            
 
     
