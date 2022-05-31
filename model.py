import math
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import torch.optim as optim
from tqdm.auto import tqdm
from empyrical import sharpe_ratio,sortino_ratio
from matplotlib import pyplot as plt

import pandas as pd
from stats import calculate_statistics,plot
from net import Critic, Actor, ActorPPO, CriticPPO
from environment import StockTradeEnvironment
from memory import ReplayBuffer, Trajectory
from random_process import OrnsteinUhlenbeckProcess

class DDPGAgent:
    
    def __init__(self,input_dim,hid_dim,state_dim,mid_dim1,mid_dim2,num_actions,df,df_eval,df_test,tech,news_tensor,date_index,args):
         
        super(DDPGAgent, self).__init__()
        
        self.state_dim = state_dim
        
        self.num_actions = num_actions
        
        self.df = df
        
        self.df_eval = df_eval

        self.df_test = df_test
        
        self.critic = Critic(input_dim,state_dim,mid_dim1,num_actions,hid_dim)
        
        self.actor = Actor(input_dim,state_dim,mid_dim2,num_actions,hid_dim)
        
        self.actor_target = Actor(input_dim,state_dim,mid_dim2,num_actions,hid_dim)
        
        self.critic_target = Critic(input_dim,state_dim,mid_dim1,num_actions,hid_dim)
            
        self.clone_model(from_model = self.critic, to_model = self.critic_target)
        
        self.clone_model(from_model = self.actor, to_model = self.actor_target)
        
        self.lr = 2 ** -14
        
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=self.lr, eps=1e-4)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=self.lr, eps=1e-4)
        
        self.NUM_STOCKS = num_actions
       
        self.replay_buffer = ReplayBuffer()
        
        self.env = StockTradeEnvironment(num_actions,df,df_eval,df_test,tech,news_tensor,date_index)
        
        self.gamma = 0.99
        
        self.batchsize = args.batchsize
        
        self.random_seed = 0      

        self.device = torch.device("cuda")

        self.random_process = OrnsteinUhlenbeckProcess(
            size=num_actions, theta=0.15, mu=0, sigma=0.2
        )
        
        np.random.seed(self.random_seed)
        
        torch.manual_seed(self.random_seed)
        
        self.cuda()
        

    def clone_model(self,from_model, to_model):
        
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            
            to_model.data.copy_(from_model.data.clone())     
        
    
    def evaluate(self,args):
                  
         reward_ = []
 
         statistics = []

         if args.is_test:

            self.load_weights(args.output)
                   
         for episode in range(args.validate_episodes):
        
             t = 0
         
             if args.is_test:

                T = len(self.df_test.index.unique().tolist())

             else:
         
                T = len(self.df_eval.index.unique().tolist())

             rewards = 0 

             if args.is_eval:

                obs  = self.env.eval_reset_()

             else:

                obs  = self.env.test_reset_()
        
             total_asset_value = list()
          
             total_asset_value.append(self.env.budget)
            
             for t in range(T-1):

                 obs1 = obs.copy()

                 obs1 = obs1 * (2 ** -6)

                 obs1[0] = obs1[0] * (2 ** -6)

                 torch_obs = torch.Tensor(obs1)

                 t1 = t+len(self.env.datelist)
                     
                 actions = self.step(torch_obs,t1,explore=False)
            
                 actions = actions.squeeze()
            
                 actions = actions.cpu().detach().numpy()
                
                 if args.is_eval:
                 
                    obs, reward, dones, next_time = self.env.step_(actions,t,T,is_eval = True) 

                 else:

                    obs, reward, dones, next_time = self.env.step_(actions,t,T,is_test = True) 

                                  
                 total_asset_value.append(total_asset_value[len(total_asset_value)-1] + reward)
            
                 rewards += reward               
                
             reward_.append(rewards)     

             print(rewards) 
            
             sharpe_ratio,sortino_ratio,mdd = calculate_statistics(total_asset_value)

             statistics.append((sharpe_ratio,sortino_ratio,mdd))
                
        
         res = [sum(x)/ len(statistics) for x in zip(*statistics)] 
     
         out = (res[0],res[1],res[2])       
 
         return out,np.mean(reward_),total_asset_value
    
    
    def retrieve_news(self,t_list):

        return self.env.get_news(t_list)
        
                
    def run(self,obs,episode_length,t):
        
          finish = False
        
          T = len(self.df.index.unique().tolist())
        
          total_steps = 0 
        
          rewards = 0 
    
          if obs is None:

             obs = self.env.get_state(is_eval = False,is_test = False)


          while not finish:
              
              total_steps += 1  

              obs1 = obs.copy()

              obs1 = obs1 * (2 ** -6)
                
              obs1[0] = obs1[0] * (2 ** -6)
                   
              torch_obs = torch.Tensor(obs1)
                
              actions = self.step(torch_obs,t,explore=True)

              actions = actions.squeeze()
            
              actions = actions.cpu().detach().numpy()
              
              next_obs, rewards, dones, next_time = self.env.step_(actions,t,T)

              reward_sum = rewards * (2 ** -5)

              rewards = reward_sum

              nobs1 = next_obs.copy()

              nobs1 = nobs1 * (2 ** -6)

              nobs1[0] = nobs1[0] * (2 ** -6)

              self.replay_buffer.push(obs1,actions,nobs1,rewards,dones,t+1) 

              obs = next_obs                 
                
              if total_steps == episode_length:
                
                     finish = True
              
              t = next_time
        
          self.train_net()
     
          return t,obs   

    
    def net_update(self,from_model,to_model):
        
        tau = 2 ** -8
        
        for target_param, local_param in zip(to_model.parameters(), from_model.parameters()):
            
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
               
        
    def optimizer_update(self, optimizer, network, loss):
        
        optimizer.zero_grad() #reset gradients to 0
        
        loss.backward(retain_graph=False) #this calculates the gradients
        
        optimizer.step()
        
    def step(self, obs, t, explore=False):
        
        news_tensor = self.retrieve_news([t])
        
        action = self.actor(news_tensor,obs)
        
        if explore:                
           
           action += torch.from_numpy(self.random_process.sample()).to(self.device).float()

            
        action = action.clamp(-1, 1)
        
        return action
    
    def get_critic_loss(self):
       
        with torch.no_grad():
                 
                 cur_obs,action,next_obs,rew,done,time = self.replay_buffer.sample(self.batchsize)
        
                 news = self.retrieve_news(time)

                 next_obs_ten = torch.FloatTensor(next_obs).squeeze()

                 predict_action = self.actor_target(news,next_obs_ten).squeeze()
             
                 Qval_predict= self.critic_target(news,next_obs_ten,predict_action).squeeze()
 
                 rew_ten = torch.FloatTensor(rew).to(torch.device("cuda"))
                
                 done_ten = torch.FloatTensor(done).to(torch.device("cuda"))
                                       
                 y_predict = rew_ten + self.gamma * (1-done_ten) * Qval_predict                
                 
                 cur_obs_ten = torch.FloatTensor(cur_obs).squeeze()
                
                 action_ten = torch.FloatTensor(action).squeeze()
                    
           
        y_actual = self.critic(news,cur_obs_ten,action_ten).squeeze()
            
        critic_loss = F.mse_loss(y_actual,y_predict).mean()
                
        return critic_loss,cur_obs_ten,news
            
    
    def train_net(self):
               
        num_learning_iterations = int((self.replay_buffer.size/self.batchsize) * 0.3) 
               
        for _ in tqdm(range(num_learning_iterations)):
            
            print("iteration",_)

            critic_loss,cur_obs_tensor,news_tensor = self.get_critic_loss()
      
            self.optimizer_update(self.critic_optimizer,self.critic, critic_loss)
     
            self.net_update(from_model = self.critic, to_model = self.critic_target)   
            
            action_tensor = self.actor(news_tensor,cur_obs_tensor)
           
            qval = self.critic(news_tensor,cur_obs_tensor,action_tensor).squeeze()
   
            actor_loss = -1 * qval.mean()
   
            print(actor_loss)
                    
            self.optimizer_update(self.actor_optimizer,self.actor, actor_loss)
        
            self.net_update(from_model = self.actor, to_model = self.actor_target)        
                
    def save_model(self, output):
        
        torch.save(self.actor.state_dict(), "{}/actor.pkl".format(output))
        
        torch.save(self.critic.state_dict(), "{}/critic.pkl".format(output))
        
        torch.save(self.actor_target.state_dict(), "{}/actor_target.pkl".format(output))
        
        torch.save(self.critic_target.state_dict(), "{}/critic_target.pkl".format(output))
             
            
    def load_weights(self, output):
      
        self.actor.load_state_dict(torch.load("{}/actor.pkl".format(output)))
    
        self.actor_target.load_state_dict(
            
              torch.load("{}/actor_target.pkl".format(output))
        
        )
        
        self.critic.load_state_dict(torch.load("{}/critic.pkl".format(output)))
        
        self.critic_target.load_state_dict(
            
              torch.load("{}/critic_target.pkl".format(output))
        
        )
        
    def eval(self):
        
        self.actor.eval()
        
        self.actor_target.eval()
        
        self.critic.eval()
        
        self.critic_target.eval()

    def train(self):
        
        self.actor.train()
        
        self.actor_target.train()
        
        self.critic.train()
        
        self.critic_target.train()
          
    
    def cuda(self):

        if torch.cuda.is_available():

           device = torch.device("cuda") 
        
           self.actor.to(device)
        
           self.actor_target.to(device)
        
           self.critic.to(device)
        
           self.critic_target.to(device)



class PPOAgent:
    
    def __init__(self,input_dim,hid_dim,state_dim,mid_dim1,mid_dim2,num_actions,df,df_eval,df_test,tech,news_tensor,date_index,args):
        
        super(PPOAgent, self).__init__()
        
        self.state_dim = state_dim
        
        self.num_actions = num_actions
        
        self.df = df
        
        self.df_eval = df_eval

        self.df_test = df_test
        
        self.critic = CriticPPO(input_dim,state_dim,mid_dim1,num_actions,hid_dim)

        self.critic_old = CriticPPO(input_dim,state_dim,mid_dim1,num_actions,hid_dim)

        self.actor = ActorPPO(input_dim,state_dim,mid_dim2,num_actions,hid_dim)
        
        self.actor_old = ActorPPO(input_dim,state_dim,mid_dim2,num_actions,hid_dim)
          
        self.clone_model(from_model = self.critic, to_model = self.critic_old)

        self.clone_model(from_model = self.actor, to_model = self.actor_old)
        
        self.lr = 2 ** -14
        
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=self.lr, eps=1e-4)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=self.lr, eps=1e-4)
        
        self.NUM_STOCKS = num_actions
       
        self.replay_buffer = Trajectory()
        
        self.env = StockTradeEnvironment(num_actions,df,df_eval,df_test,tech,news_tensor,date_index)
        
        self.gamma = 0.99
        
        self.batchsize = args.batchsize
        
        self.random_seed = 0      
        
        np.random.seed(self.random_seed)
        
        torch.manual_seed(self.random_seed)
        
        self.last_value = 0

        self.epsilon = 0.1

        self.cuda()
    
        if torch.cuda.is_available():

           self.device = torch.device("cuda") 


    def clone_model(self,from_model, to_model):
        
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            
            to_model.data.copy_(from_model.data.clone())     


    def evaluate(self,args):
                   
         reward_ = []

         statistics = []

         if args.is_test:
        
            self.load_weights(args.output)
           
        
         for episode in range(args.validate_episodes):
        
             t = 0
         
             if args.is_test:

                T = len(self.df_test.index.unique().tolist())

             else:
         
                T = len(self.df_eval.index.unique().tolist())
        
             rewards = 0 

             if args.is_eval:

                obs  = self.env.eval_reset_()

             else:
 
                obs  = self.env.test_reset_()
        
             total_asset_value = list()
          
             total_asset_value.append(self.env.budget)
            
             for t in range(T-1):

                 obs1 = obs.copy()

                 obs1 = obs1 * (2 ** -6)

                 obs1[0] = obs1[0] * (2 ** -6)

                 torch_obs = torch.Tensor(obs1)

                 t1 = t+len(self.env.datelist)
                     
                 actions = self.step(torch_obs,t1,explore=False)

                 actions = actions.tanh()
            
                 actions = actions.squeeze()
            
                 actions = actions.cpu().detach().numpy()

                 if args.is_eval:
                
                    obs, reward, dones, next_time = self.env.step_(actions,t,T,is_eval = True) 

                 else:

                    obs, reward, dones, next_time = self.env.step_(actions,t,T,is_test = True) 

                      
                 total_asset_value.append(total_asset_value[len(total_asset_value)-1] + reward)
            
                 rewards += reward               
                
             reward_.append(rewards)     

             print(rewards) 
            
             sharpe_ratio,sortino_ratio,mdd = calculate_statistics(total_asset_value)

             statistics.append((sharpe_ratio,sortino_ratio,mdd))
                
        
         res = [sum(x)/ len(statistics) for x in zip(*statistics)] 
     
         out = (res[0],res[1],res[2])       
 
         return out,np.mean(reward_),total_asset_value
    
    def run(self,obs,episode_length,t):
        
          finish = False
        
          T = len(self.df.index.unique().tolist())
        
          total_steps = 0 
        
          rewards = 0 

          self.replay_buffer.reset()
    
          if obs is None:

             obs = self.env.get_state(is_eval = False, is_test = False)


          while not finish:
                                     
              total_steps += 1  

              obs1 = obs.copy()

              obs1 = obs1 * (2 ** -6)
                
              obs1[0] = obs1[0] * (2 ** -6)
                                   
              torch_obs = torch.Tensor(obs1)
                
              action = self.step(torch_obs,t,explore=True)
 
              actions = action.clone()

              action = action.squeeze().cpu().detach().numpy()

              actions = actions.tanh().squeeze().cpu().detach().numpy()
  
              next_obs, rewards, dones, next_time = self.env.step_(actions,t,T)

              rewards = rewards * (2 ** -5)

              with torch.no_grad():
                    
                   news = self.retrieve_news([t])

                   val = self.critic_old(news,torch.FloatTensor(obs1))
 
                   logprob,entropy = self.compute_logprob(obs1,action,[t])

                   logprob = logprob.cpu().data

                   val = val.cpu().data


              nobs1 = next_obs.copy()

              nobs1 = nobs1 * (2 ** -6)

              nobs1[0] = nobs1[0] * (2 ** -6)

              self.replay_buffer.push(obs1,action,nobs1,rewards,dones,val,logprob,t+1) 

              obs = next_obs   
                            
              if total_steps == episode_length:
                
                     finish = True
               
              t = next_time
        
          self.train_net()
                      
          return t,obs    

    def retrieve_news(self,t_list):

        return self.env.get_news(t_list)
        
    
    def net_update(self,from_model,to_model,tau):
        
        for target_param, local_param in zip(to_model.parameters(), from_model.parameters()):
            
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
        
    def optimizer_update(self, optimizer, network, loss):
        
        optimizer.zero_grad() #reset gradients to 0
        
        loss.backward(retain_graph=False) #this calculates the gradients
        
        optimizer.step()

    def get_advantage_value(self):
        
        with torch.no_grad():
        
             values = self.replay_buffer.value_queue
     
             done = self.replay_buffer.done_queue
                
             done = np.stack(done).squeeze()
        
             rewards = self.replay_buffer.reward_queue 
                   
             reward, advantage = self.compute_advantage_estimate(rewards,values,done)
        
             reward_ten = torch.FloatTensor(reward)
    
             advantage_ten =  torch.FloatTensor(advantage)
                     
             advantage_ten = (advantage_ten - advantage_ten.mean()) / (advantage_ten.std() + 1e-4)
                
             reward_ten = reward_ten.to(self.device)   

             advantage_ten = advantage_ten.to(self.device) 
  
        return reward_ten,advantage_ten    
            
                   
              
    def train_net(self):
      
        reward_ten,advantage_ten = self.get_advantage_value()
        
        num_learning_iterations = int((self.replay_buffer.size/ self.batchsize) * 0.3 )
           
        for _ in tqdm(range(num_learning_iterations)):    
            
            print("iteration",_)
            
            self.train_ppo(_,reward_ten,advantage_ten)
                
                
            
    def train_ppo(self,_,reward_ten,advantage_ten):
        
         
        indices = torch.randint(self.replay_buffer.size, size=(self.batchsize,), requires_grad=False).tolist()
                        
        obs = [self.replay_buffer.cur_obs_queue[index] for index in indices]
        
        obs_ = np.stack(obs)
        
        action = [self.replay_buffer.action_queue[index] for index in indices]
        
        action_ = np.stack(action)
        
        obs_ = obs_.squeeze()

        date = [self.replay_buffer.date_queue[index] for index in indices]
            
        new_logprob, dist_entropy = self.compute_logprob(obs_,action_,date)
        
        old_logprob =  [self.replay_buffer.oldlogprob_queue[index] for index in indices]
 
        old_logprob = torch.FloatTensor(old_logprob).to(self.device)
        
        ratio = torch.exp(new_logprob - old_logprob)
                        
        clipped_ratio = torch.clip(ratio,1-self.epsilon,1+self.epsilon)       

        clipped_adv = clipped_ratio * advantage_ten[indices]
    
        nonclipped_adv = ratio * advantage_ten[indices]
    
        advantage_final = torch.min(clipped_adv,nonclipped_adv) 
             
        actor_loss = -1 * advantage_final.mean() + 0.02 * dist_entropy
              
        print(actor_loss)
        
        self.optimizer_update(self.actor_optimizer,self.actor, actor_loss)
        
        news = self.retrieve_news(date)
                        
        q_val = self.critic(news,torch.from_numpy(obs_).float())
                                       
        critic_loss = F.mse_loss(reward_ten[indices],q_val.squeeze())/ (reward_ten[indices].std() + 1e-6)
            
        print(critic_loss)
        
        self.optimizer_update(self.critic_optimizer,self.critic, critic_loss)
                
        tau = 2 ** -8
        
        self.net_update(self.critic, self.critic_old,tau)
            
    def step(self,torch_obs,t,explore=True):

        obs = torch_obs.clone()

        news = self.retrieve_news([t])     
        
        mu, sigma = self.actor(news,obs)
        
        cov_mat = torch.diag(sigma).unsqueeze(dim=0)
            
        dist = MultivariateNormal(mu, cov_mat)
        
        action = dist.sample()
           
        return action
  
    def compute_logprob(self,obs,action,date):
    
    
        obs1 = torch.FloatTensor(obs)

        news = self.retrieve_news(date)
                
        mu, sigma = self.actor(news,obs1)
        
        action_var = sigma.expand_as(mu)
            
        cov_mat = torch.diag_embed(action_var)
            
        dist = MultivariateNormal(mu, cov_mat)
            
        action_logprobs = dist.log_prob(torch.FloatTensor(action).to(self.device))    
        
        dist_entropy = (action_logprobs.exp() * action_logprobs).mean()
            
        return action_logprobs,dist_entropy
    
    
  
    def compute_advantage_estimate(self,rewards,values,done):
        
        advantage = list()
        
        reward = list()
        
        gamma = 0.98
        
        alpha = 0.95
            
        next_v = self.last_value
            
        next_A = 0
        
        next_done = 0
            
        for j in reversed(range(len(rewards))):
                              
                              
                delta = rewards[j] + gamma * next_v * (1 - next_done) - values[j]
            
                new_adv_val = delta + alpha * gamma * (1 - next_done) * next_A
                
                reward.append(new_adv_val + values[j])
                
                next_A = new_adv_val
                
                next_v = values[j]
                                                       
                next_done = done[j]                                      
                              
                advantage.append(new_adv_val)
            
                              
        advantage.reverse()   
            
        reward.reverse()
                                      
        return reward, advantage       
                                           
        
    def eval(self):
        
        self.actor.eval()
        
        self.actor_old.eval()
        
        self.critic.eval()
        
        self.critic_old.eval()
        
    def train(self):
        
        self.actor.train()
        
        self.actor_old.train()
        
        self.critic.train()
        
        self.critic_old.train()
        
        
    def load_weights(self, output):
      
        
        self.actor.load_state_dict(torch.load("{}/actor.pkl".format(output)))
    
        self.actor_old.load_state_dict(
            
             torch.load("{}/actor_old.pkl".format(output))
        
        )
        
        self.critic.load_state_dict(torch.load("{}/critic.pkl".format(output)))
        
        self.critic_old.load_state_dict(
            
             torch.load("{}/critic_old.pkl".format(output))
        
        )

    def save_model(self, output):
        
        torch.save(self.actor.state_dict(), "{}/actor.pkl".format(output))
        
        torch.save(self.critic.state_dict(), "{}/critic.pkl".format(output))
        
        torch.save(self.actor_old.state_dict(), "{}/actor_old.pkl".format(output))
        
        torch.save(self.critic_old.state_dict(), "{}/critic_old.pkl".format(output))
    
    def cuda(self):

        if torch.cuda.is_available():

           device = torch.device("cuda") 
        
           self.actor.to(device)
        
           self.actor_old.to(device)
        
           self.critic.to(device)
        
           self.critic_old.to(device)

   

class TD3Agent:
    
    def __init__(self,input_dim,hid_dim,state_dim,mid_dim1,mid_dim2,num_actions,df,df_eval,df_test,tech,news_tensor,date_index,args):
        
        super(TD3Agent, self).__init__()
        
        self.state_dim = state_dim
        
        self.num_actions = num_actions
        
        self.df = df
        
        self.df_eval = df_eval

        self.df_test = df_test
        
        self.critic1 = Critic(input_dim,state_dim,mid_dim1,num_actions,hid_dim)

        self.critic2 = Critic(input_dim,state_dim,mid_dim1,num_actions,hid_dim)

        self.critic1_target = Critic(input_dim,state_dim,mid_dim1,num_actions,hid_dim)

        self.critic2_target = Critic(input_dim,state_dim,mid_dim1,num_actions,hid_dim)
        
        self.actor = Actor(input_dim,state_dim,mid_dim2,num_actions,hid_dim)
        
        self.actor_target = Actor(input_dim,state_dim,mid_dim2,num_actions,hid_dim)
          
        self.clone_model(from_model = self.critic1, to_model = self.critic1_target)

        self.clone_model(from_model = self.critic2, to_model = self.critic2_target)
        
        self.clone_model(from_model = self.actor, to_model = self.actor_target)
        
        self.lr = 2 ** -14

        self.policy_update = 2
        
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(),lr=self.lr, eps=1e-4)

        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(),lr=self.lr, eps=1e-4)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=self.lr, eps=1e-4)
        
        self.NUM_STOCKS = num_actions
       
        self.replay_buffer = ReplayBuffer()
        
        self.env = StockTradeEnvironment(num_actions,df,df_eval,df_test,tech,news_tensor,date_index)
        
        self.gamma = 0.99
        
        self.batchsize = args.batchsize
        
        self.random_seed = 0      
        
        np.random.seed(self.random_seed)
        
        torch.manual_seed(self.random_seed)
        
        self.cuda()
    
    def clone_model(self,from_model, to_model):
        
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            
            to_model.data.copy_(from_model.data.clone())     
    
    def evaluate(self,args):
        
         statistics = []
            
         reward_ = []

         if args.is_test:
        
            self.load_weights(args.output)           
        
         for episode in range(args.validate_episodes):
        
             t = 0
         
             if args.is_test:

                T = len(self.df_test.index.unique().tolist())

             else:
         
                T = len(self.df_eval.index.unique().tolist())
        
             rewards = 0 

             if args.is_eval:

                obs  = self.env.eval_reset_()

             else:
 
                obs  = self.env.test_reset_()
        
             total_asset_value = list()
          
             total_asset_value.append(self.env.budget)
            
             for t in range(T-1):

                 obs1 = obs.copy()

                 obs1 = obs1 * (2 ** -6)

                 obs1[0] = obs1[0] * (2 ** -6)

                 torch_obs = torch.Tensor(obs1)

                 t1 = t+len(self.env.datelist)
                     
                 actions = self.step(torch_obs,t1,explore=False)
            
                 actions = actions.squeeze()
            
                 actions = actions.cpu().detach().numpy()

                 if args.is_eval:
                
                    obs, reward, dones, next_time = self.env.step_(actions,t,T,is_eval = True) 

                 else:

                    obs, reward, dones, next_time = self.env.step_(actions,t,T,is_test = True) 

                      
                 total_asset_value.append(total_asset_value[len(total_asset_value)-1] + reward)
            
                 rewards += reward               
                
             reward_.append(rewards)     

             print(rewards) 
            
             sharpe_ratio,sortino_ratio,mdd = calculate_statistics(total_asset_value)

             statistics.append((sharpe_ratio,sortino_ratio,mdd))
                
        
         res = [sum(x)/ len(statistics) for x in zip(*statistics)] 
     
         out = (res[0],res[1],res[2])       
 
         return out,np.mean(reward_),total_asset_value
        
    
    
    def retrieve_news(self,t_list):

        return self.env.get_news(t_list)
        
                
    def run(self,obs,episode_length,t):
        
          finish = False
        
          T = len(self.df.index.unique().tolist())
        
          total_steps = 0 
        
          rewards = 0 
    
          if obs is None:

             obs = self.env.get_state(is_eval = False, is_test = False)


          while not finish:
                                         
              total_steps += 1 

              obs1 = obs.copy()

              obs1 = obs1 * (2 ** -6)
                
              obs1[0] = obs1[0] * (2 ** -6)
                                   
              torch_obs = torch.Tensor(obs1)
                
              actions = self.step(torch_obs,t,explore=True)

              actions = actions.squeeze()
            
              actions = actions.cpu().detach().numpy()
  
              next_obs, rewards, dones, next_time = self.env.step_(actions,t,T)

              reward_sum = rewards * (2 ** -5)

              rewards = reward_sum

              nobs1 = next_obs.copy()

              nobs1 = nobs1 * (2 ** -6)

              nobs1[0] = nobs1[0] * (2 ** -6)

              self.replay_buffer.push(obs1,actions,nobs1,rewards,dones,t+1) 

              obs = next_obs   
                            
              if total_steps == episode_length:
                
                     finish = True
               
              t = next_time
        
          self.train_net()
                      
          return t,obs    

    
    def step(self, obs, t, explore=False):
        
        news_tensor = self.retrieve_news([t])
        
        action = self.actor(news_tensor,obs)
        
        if explore:                
           
           action += torch.randn_like(action) * 0.1
            
        action = action.clamp(-1, 1)
        
        return action
    

    def retrieve_news(self,t_list):

        return self.env.get_news(t_list)
        
    
    def net_update(self,from_model,to_model):
        
        tau = 2 ** -8
        
        for target_param, local_param in zip(to_model.parameters(), from_model.parameters()):
            
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
        
    def optimizer_update(self, optimizer, network, loss):
        
        optimizer.zero_grad() #reset gradients to 0
        
        loss.backward(retain_graph=False) #this calculates the gradients
        
        optimizer.step()
        
    
    def get_critic_loss(self):

        action_std = 0.15
       
        with torch.no_grad():
                 
                 cur_obs,action,next_obs,rew,done,time = self.replay_buffer.sample(self.batchsize)
        
                 news = self.retrieve_news(time)
            
                 next_obs_ten = torch.FloatTensor(next_obs).squeeze()
                   
                 predict_action = self.actor_target(news,next_obs_ten).squeeze()
                           
                 noise = (torch.randn_like(predict_action) * action_std).clamp(-0.5, 0.5)
        
                 predict_action = (predict_action + noise).clamp(-1.0, 1.0)

                 Qval_predict1 = self.critic1_target(news,next_obs_ten,predict_action).squeeze()

                 Qval_predict2 = self.critic2_target(news,next_obs_ten,predict_action).squeeze()

                 Qval_predict = torch.min(Qval_predict1, Qval_predict2)
                    
                 rew_ten = torch.FloatTensor(rew).to(torch.device("cuda"))
                
                 done_ten = torch.FloatTensor(done).to(torch.device("cuda"))
                                       
                 y_predict = rew_ten + self.gamma * (1-done_ten) * Qval_predict                 
                    
                 cur_obs_ten = torch.FloatTensor(cur_obs).squeeze()
                
                 action_ten = torch.FloatTensor(action).squeeze()
  
        y_actual_1 = self.critic1(news,cur_obs_ten,action_ten).squeeze()
 
        y_actual_2 = self.critic2(news,cur_obs_ten,action_ten).squeeze()
        
        critic_loss1 = F.mse_loss(y_actual_1,y_predict).mean()

        critic_loss2 = F.mse_loss(y_actual_2,y_predict).mean()
                
        return critic_loss1,critic_loss2,cur_obs_ten,news
            
    
    def train_net(self):
        
        num_learning_iterations = int((self.replay_buffer.size/self.batchsize) * 0.4 ) 
               
        for _ in tqdm(range(num_learning_iterations)):
            
            print("iteration",_)
            
            critic_loss1,critic_loss2,cur_obs_tensor,news_tensor = self.get_critic_loss()
      
            self.optimizer_update(self.critic1_optimizer,self.critic1, critic_loss1)

            self.optimizer_update(self.critic2_optimizer,self.critic2, critic_loss2)
        
            if _ % self.policy_update == 0:


               action_tensor = self.actor(news_tensor,cur_obs_tensor)
                       
               qval = self.critic1_target(news_tensor,cur_obs_tensor,action_tensor).squeeze()
            
               actor_loss = -1 * qval.mean()
    
               print(actor_loss)
                    
               self.optimizer_update(self.actor_optimizer,self.actor, actor_loss)

               self.net_update(from_model = self.actor, to_model = self.actor_target) 

               self.net_update(from_model = self.critic1, to_model = self.critic1_target)      
                
               self.net_update(from_model = self.critic2, to_model = self.critic2_target) 
            
            
    def load_weights(self, output):
      
        self.actor.load_state_dict(torch.load("{}/actor.pkl".format(output)))
    
        self.actor_target.load_state_dict(
            
              torch.load("{}/actor_target.pkl".format(output))
        
        )
        
        self.critic1.load_state_dict(torch.load("{}/critic1.pkl".format(output)))
        
        self.critic1_target.load_state_dict(
            
              torch.load("{}/critic1_target.pkl".format(output))
        
        )

    
        self.critic2.load_state_dict(torch.load("{}/critic2.pkl".format(output)))
        
        self.critic2_target.load_state_dict(
            
              torch.load("{}/critic2_target.pkl".format(output))
        
        )
    
    def save_model(self, output):
        
        torch.save(self.actor.state_dict(), "{}/actor.pkl".format(output))
        
        torch.save(self.critic1.state_dict(), "{}/critic1.pkl".format(output))
    
        torch.save(self.critic2.state_dict(), "{}/critic2.pkl".format(output))
        
        torch.save(self.actor_target.state_dict(), "{}/actor_target.pkl".format(output))
        
        torch.save(self.critic1_target.state_dict(), "{}/critic1_target.pkl".format(output))
    
        torch.save(self.critic2_target.state_dict(), "{}/critic2_target.pkl".format(output))

    
    def eval(self):
        
        self.actor.eval()
        
        self.actor_target.eval()
        
        self.critic1.eval()
        
        self.critic1_target.eval()
        
        self.critic2.eval()
        
        self.critic2_target.eval()

    def train(self):
       
        self.actor.train()
        
        self.actor_target.train()
        
        self.critic1.train()
        
        self.critic1_target.train()
        
        self.critic2.train()
        
        self.critic2_target.train()      
    

    def cuda(self):

        if torch.cuda.is_available():

           device = torch.device("cuda") 
        
           self.actor.to(device)
        
           self.actor_target.to(device)
        
           self.critic1.to(device)
        
           self.critic1_target.to(device)

           self.critic2.to(device)
        
           self.critic2_target.to(device)

