import pandas as pd
import argparse
import yaml
from Data_preprocess import DataDownloader
from Data_preprocess import Feature_Engineering
from Data_preprocess import StockNewsExtract
from tqdm.auto import tqdm
import itertools
from Data_preprocess import data_split
import torch
from model import DDPGAgent
from model import PPOAgent
from model import TD3Agent
MODELS = {"ddpg": DDPGAgent, "td3": TD3Agent, "ppo": PPOAgent}
import os
import torch

from stats import plot,plot_portfolio_value,plot_multiple_portfolio_value,calculate_statistics

from matplotlib import pyplot as plt

def train(total_steps,agent,episode_length,output,args):
    
    agent.is_training = True
    
    best_sharpe = -9999
    
    best_reward = -99999
    
    eval_episode_reward = []

    step = 0

    state = None

    num_epochs = int(total_steps/episode_length) + 1

    for _ in tqdm(range(num_epochs)):
     
          step,state = agent.run(state,episode_length,step)
            
          agent.is_training = False
            
          agent.eval()   

          args.is_test = False    
        
          stat,cur_reward,total_asset_value = agent.evaluate(args)

          eval_episode_reward.append(cur_reward)
    
          agent.train()
            
          agent.is_training = True
        
          cur_sharpe = stat[0]        
        
          if cur_sharpe > best_sharpe:
               
             print("saving model!!!!")
                   
             best_sharpe = cur_sharpe
                   
             best_reward = cur_reward
                            
             agent.save_model(output)

    x_date = agent.env.df_eval.date.unique().tolist()

    date_range = pd.to_datetime(x_date) 
            
    plot_portfolio_value(total_asset_value,date_range,args.agent,args) 
    
    print(total_asset_value)

    result = {
                  "sharpe": stat[0][0],
                  "sortino": stat[1],
                  "mdd": stat[2],
                  "return": (total_asset_value[len(total_asset_value)-1] - agent.env.budget)/agent.env.budget
             }

    print(result)
                             

def test(agent,output,args):

    agent.load_weights(output)
    
    agent.is_training = False
    
    agent.eval()

    args.is_test = True
    
    stat,reward,portfolio_value = agent.evaluate(args)

    result = {
                  "sharpe": stat[0][0],
                  "sortino": stat[1],
                  "mdd": stat[2],
                  "return": (portfolio_value[len(portfolio_value)-1] - agent.env.budget)/agent.env.budget
             }

    print(result)

    x_date = agent.env.df_test.date.unique().tolist()

    date_range = pd.to_datetime(x_date) 

    plot_portfolio_value(portfolio_value,date_range,args.agent,args) 
    
           
if __name__ == "__main__":
    
   config_file = os.path.join(os.getcwd(),"config.yml")

   print(config_file)
    
   parser = argparse.ArgumentParser(description="PyTorch on stock trading using reinforcement learning")


   parser.add_argument(
      
       "--mode", default="train", type=str, help="option: train/test"
   )
    
   parser.add_argument(
       "--mid_dim1",
       default=70,
       type=int,
       help="hidden num of first fully connect layer of critic",
   )

   parser.add_argument(
       "--mid_dim2",
       default=70,
       type=int,
       help="hidden num of second fully connect layer of actor",
   )

   
   parser.add_argument("--batchsize", default = 2 ** 6, type=int, help="minibatch size")

   parser.add_argument("--input_dim", default = 768, type=int, help="BERT embedding size")
    
   parser.add_argument("--hid_dim", default = 100, type=int) 


   parser.add_argument(
      
       "--tau", default=2 **-8, type=float, help="moving average for target network"

   )

   parser.add_argument("--episode_length", default= 2 ** 12, type=int, help="")

   parser.add_argument(
      
       "--train_steps", default=36000, type=int, help =""

   )

   parser.add_argument(
      
       "--validate_episodes", default = 3, type=int, help =""

   )

   parser.add_argument(
      
       "--START_DATE", default = "2019-02-13", type=str, help =""

   )
   parser.add_argument(
      
       "--START_TRADE_DATE", default = "2021-10-01", type=str, help =""
   )

   parser.add_argument(
      
       "--END_TRADE_DATE", default = "2022-01-01", type=str, help =""
   )

   parser.add_argument(
      
       "--START_TEST_DATE", default = "2021-10-01", type=str, help =""
   )

   parser.add_argument(
      
       "--END_TEST_DATE", default = "2022-01-01", type=str, help =""

   )

   parser.add_argument(
      
       "--agent", default = "td3", type=str, help =""

   )
  
   parser.add_argument(
      
       "--num_agents", default = 3, type=str, help =""

   )

   parser.add_argument(
      
       "--is_test", default = False, type=str, help =""

   )

   parser.add_argument(
      
       "--is_eval", default = True, type=str, help =""

   )
 
   args = parser.parse_args()
    
     
   with open(config_file, "r") as ymlfile:
    
        cfg = yaml.safe_load(ymlfile)
        
   tickers = cfg['DOW_11_TICKER']

   tech_indicators = cfg['TECHNICAL_INDICATORS_LIST']
    
   args.output = os.path.join(os.getcwd(),"output")

   output = args.output

   news_db = os.path.join(os.getcwd(),"financial_data.db")

   START_DATE = args.START_DATE
  
   END_DATE = args.END_TEST_DATE

   START_TRADE_DATE = args.START_TRADE_DATE

   END_TRADE_DATE = args.END_TRADE_DATE

   START_TEST_DATE = args.START_TEST_DATE

   END_TEST_DATE = args.END_TEST_DATE

   df = DataDownloader(
          start_date = START_DATE,
          end_date = END_DATE,
          stock_list = tickers,
        )

   df = df.fetch_data()

   feature = Feature_Engineering(tech_indicators)

   processed = feature.preprocess(df)

   list_ticker = processed["tic"].unique().tolist()

   list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))

   combination = list(itertools.product(list_date,list_ticker))

   df_final = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")

   df_final = df_final[df_final['date'].isin(processed['date'])]

   df_final = df_final.sort_values(['date','tic'])

   train_df = data_split(df_final, START_DATE, START_TRADE_DATE)
    
   trade_df = data_split(df_final, START_TRADE_DATE, END_TRADE_DATE)

   test_df = data_split(df_final, START_TEST_DATE, END_TEST_DATE)


   state_dim = len(tickers) * 5 +1

   num_actions = len(tickers)

   mid_dim1 = args.mid_dim1

   mid_dim2 = args.mid_dim2

   stockextract = StockNewsExtract(news_db,tickers,output)

   date_index = stockextract.get_datelist()    
 
   news_embedding_file = "{}/news_tensor.pkl".format(os.getcwd())

   news_tensor = None

   if os.path.exists(news_embedding_file):

      news_tensor = torch.load(news_embedding_file)

   else:

      news_tensor = stockextract.process_news()

    
   if args.mode == "train":

        agent = MODELS[args.agent](args.input_dim,args.hid_dim,state_dim,mid_dim1,mid_dim2,num_actions,train_df,trade_df,test_df,tech_indicators,news_tensor,date_index,args)  

        train(args.train_steps,agent,args.episode_length,output,args)


   elif args.mode == "test":

        agent = MODELS[args.agent](args.input_dim,args.hid_dim,state_dim,mid_dim1,mid_dim2,num_actions,train_df,trade_df,test_df,tech_indicators,news_tensor,date_index,args)

        test(agent,args.output,args)

   else:
        raise RuntimeError("undefined mode {}".format(args.mode))
 
   