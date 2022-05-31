import pandas as pd
import yfinance as yf
import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
import itertools
from transformers import BertTokenizer,BertModel
import sqlite3
import torch
EMBED_DIM = 768
from scipy.spatial import distance
import linalg
from numpy.linalg import pinv


class DataDownloader:
    
    def __init__(self, start_date: str, end_date: str, stock_list: list):

        self.start_date = start_date
        self.end_date = end_date
        self.stock_list = stock_list
        
    def format_time(self,df):
        
        df["date"] =  df["date"].dt.strftime("%Y-%m-%d")
        
        return df

    def fetch_data(self) -> pd.DataFrame:
       
        df = pd.DataFrame()
        for tic in self.stock_list:
            df_ = yf.download(tic, start=self.start_date, end=self.end_date)
            df_["tic"] = tic
            df = df.append(df_)
        # reset the index, we want to use numbers as index instead of dates
        df = df.reset_index()
        # convert the column names to standardized names
        df.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjcp",
                "volume",
                "tic",
        ]
        # use adjusted close price instead of close price
        df["close"] = df["adjcp"]
        df = df.drop("adjcp", 1)
       
        # create day of the week column (monday = 0)
        df["day"] = df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        df = self.format_time(df)
        print(df["date"])
        # drop missing data
        df = df.dropna()
        df = df.reset_index(drop=True)
        print("Shape of DataFrame: ", df.shape)
        
        df = df.sort_values(by=['date','tic'])
        print(df)
        df = df.reset_index(drop=True)

        return df


def data_split(df, start, end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.date >= start) & (df.date < end)]
    data = data.sort_values(["date", "tic"], ignore_index=True)
    data.index = data.date.factorize()[0]
    return data


class Feature_Engineering:
   
    def __init__(self,technical_index):
        
        self.technical_index = technical_index
        

    def preprocess(self, df):
        
        df = self.clean_data(df)
        
        df = self.add_technical_indicator(df)
            
        df = df.fillna(method="bfill").fillna(method="ffill")
        
        return df
    
    def clean_data(self, data):
    
        df = data.copy()
        df = df.sort_values(['date','tic'],ignore_index=True)
        df.index = df.date.factorize()[0]
        merged_closes = df.pivot_table(index = 'date',columns = 'tic', values = 'close')
        merged_closes = merged_closes.dropna(axis=1)
        tics = merged_closes.columns
        df = df[df.tic.isin(tics)]
        
        return df
    
    def smooth(self,turbulence_data, half_life):
       
        import math
        
        half_life = float(half_life)
        
        smoothing_factor = 1 - math.exp(math.log(0.5) / half_life)
        
        smoothed_values = [turbulence_data[0]]
        
        for index in range(1, len(turbulence_data)):
            previous_smooth_value = smoothed_values[-1]
            
            new_unsmooth_value = turbulence_data[index]
            
            new_smooth_value = ((smoothing_factor * new_unsmooth_value)
                + ((1 - smoothing_factor) * previous_smooth_value))
            
            smoothed_values.append(new_smooth_value)
        
        return(smoothed_values)
    
    def add_technical_indicator(self, data):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        
        df = df.sort_values(by=['tic','date'])
        
        stock = Sdf.retype(df.copy())
               
        stock_names = stock.tic.unique()       
        
        tech_indicator = self.technical_index
        
        for indicator in tech_indicator:
        
             df_indicator = pd.DataFrame()
        
             for i in range(len(stock_names)):
            
                 try:
                   
                     indicator_ = stock[stock.tic == stock_names[i]][indicator]
                     indicator_ = pd.DataFrame(indicator_)
                     indicator_['tic'] = stock_names[i]
                     size = len(df[df.tic == stock_names[i]]['date'].to_list())                
                     assert len(indicator_) == size
                     indicator_['date'] = df[df.tic == stock_names[i]]['date'].to_list()
                     df_indicator = df_indicator.append(
                           indicator_, ignore_index=True
                     )
                 except Exception as e:
                 
                     print(e)
               
        
             df = df.merge(df_indicator[['tic','date',indicator]],on=['tic','date'],how='left')
        
        df = df.sort_values(by=['date','tic'])
        
        df1 = df.copy()
    
        dates = df1.date.unique()
    
        start = 250
    
        i = start
    
        df_pivot = df1.pivot(index="date", columns="tic", values="close")
    
        turbulence_index = [0] * start

        while(i<len(dates)):

        
             historical_price = df_pivot.iloc[i-start:i]
        
             historical_price_mean = historical_price.mean()

             current_price = df_pivot.iloc[[i]]

             historical_price_cov = pinv(historical_price.cov())
   
             dist = distance.mahalanobis(current_price, historical_price_mean, historical_price_cov)**2
        
             turbulence_index.append(dist)
        
             i += 1


        turbulence_index = self.smooth(turbulence_index,12)

        turbulence_index = pd.DataFrame(
            {"date": df_pivot.index, "turbulence": turbulence_index}
        )

        df = df.merge(turbulence_index, on="date")
        
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
           
        return df

class StockNewsExtract:
    
    def __init__(self,db,TICKER,output):
        
        self.db = db
        
        self.TICKER = TICKER
        
        self.NUM_STOCKS = len(TICKER)
        
        self.date_index = {}
        
        self.dindex = {}
        
        self.sindex = {}
        
        self.ticker_idx = {}
        
        self.output = output
        
        self.news_tensor = torch.zeros(self.NUM_STOCKS,len(self.date_index),30,EMBED_DIM)
        
        
    
    def get_datelist(self):  
        
        con = sqlite3.connect(self.db)
        
        mydatelist = []

        for ticker in self.TICKER:

            t_ = ticker + "_"
    
            print(t_)

            query = 'SELECT * FROM {}'.format(t_)
       
            df = pd.read_sql_query(query,con)
    
            df['date'] = pd.to_datetime(df['date'])
    
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    
            mydatelist.extend(set(df["date"]))
        
        start_ind = 0

        for s in sorted(set(mydatelist)):    
    
            self.date_index[s] = start_ind
    
            start_ind += 1 
        
        index = 0
        
        for ticker in self.TICKER:
    
            self.ticker_idx[ticker] = index
    
            index += 1

        return self.date_index
      
    def get_embeddings(self,dindex,sindex,newslist):
    
        storage = []

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )


        for text in newslist:
      
            # Add the special tokens.
            marked_text = "[CLS] " + text + " [SEP]"

            # Split the sentence into tokens.
            tokenized_text = tokenizer.tokenize(marked_text)
   
            # Map the token strings to their vocabulary indeces.
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

            segments_ids = [1] * len(tokenized_text)

            tokens_tensor = torch.tensor([indexed_tokens])
      
            segments_tensors = torch.tensor([segments_ids])

            # Run the text through BERT, and collect all of the hidden states produced
            # from all 12 layers. 
            with torch.no_grad():

                 outputs = model(tokens_tensor, segments_tensors)
 
        
            hidden_states = outputs[2]


            # `hidden_states` has shape [13 x 1 x 22 x 768]

            # `token_vecs` is a tensor with shape [22 x 768]
            token_vecs = hidden_states[-2][0]

            # Calculate the average of all 22 token vectors.
            sentence_embedding = torch.mean(token_vecs, dim=0)

            storage.append(sentence_embedding)
        
        storage = torch.stack(storage,dim = 0) 
    
        capacity = self.news_tensor.shape[2] if storage.shape[0] > self.news_tensor.shape[2] else storage.shape[0]
    
        self.news_tensor[sindex,dindex,:capacity,:] = storage[:capacity]    
    
    def process_news(self):
        
        con = sqlite3.connect(self.db)
            
        for ticker in self.TICKER:

            t_ = ticker + "_"

            query = 'SELECT * FROM {}'.format(t_)
       
            df = pd.read_sql_query(query,con)
    
            df['date'] = pd.to_datetime(df['date'])
    
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    
            for d in sorted(set(df["date"])):
        
                newslist = []
        
                mylist = df[df["date"] == d].headline
        
                for item in mylist:
        
                    newslist.append(item)
        
        
                dindex = self.date_index[d]
        
                sindex = self.ticker_idx[ticker]
        
                print(dindex,sindex)
        
                self.get_embeddings(dindex,sindex,newslist)    

        torch.save(self.news_tensor,"{}/news_tensor.pkl".format(self.output))

        return self.news_tensor