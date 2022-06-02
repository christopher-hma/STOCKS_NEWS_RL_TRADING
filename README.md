# STOCKS TRADING USING DEEP REINFORCEMENT LEARNING AND NATURAL LANGUAGE PROCESSING

This respository contains implementation of stocks trading by means of deep reinforcement learning algorithms and natural language processing techniques.

# File Contents and Description

memory.py  -> This file implements the replay buffers for various single agent deep reinforcement learning algorithms

config.py -> This file contains all the tickers and technical indicators used for the stock market environment.

models.py -> This file contains implementation of the network architectures used in the deep reinforcement learning algorithms (DDPG,TD3,PPO). 

environment.py -> This file contains the code for the stock trading market environment.

Data_preprocess.py -> This file contains implementation for extracting stocks' historical pricing data and its news headline.

main.py -> This file kicks off the training of the model.

random_process.py -> This file contains implementation of random sampling for the DDPG algorithm.

stats.py -> This file contains implementation of sharpe ratio/max-drawdown/sortino ratio calculation of the portfolio.

net.py -> This file contains CNN based news embedding model and LSTM based Actor & Critic model for the deep reinforcement learning algorithm.

financial_data.db -> A SQL based database storing the news headline of selected stocks.

# Installation Steps
Execute the following command to install the dependencies:
<code> pip install -r requirement.txt </code>

# News-headline Processing

To encode the texts, we utilized the 768-dimensional embedding obtained per news item of each company by averaging the token-level outputs from 
the final BERT layer. Our algorithm then used the CNN text embedding architecture to obtain a 1-dimensional text embedding. For each trading day,
we compile the news-headline of the last 7 days for each company and feed into the CNN text-embedding architecture to obtain the news encoding. 
We then encode the news encodings using LSTM and then combine the news encoding with the stocks' pricing data as well as their technical indicators and turbulence value which then input into the actor-critic model for decision making.

# Download the financial news and tensor embeddings
<code> python setup.py </code>

# Training
Execute the following command to train STOCKS_NEWS_RL:
<code>python main.py --train_steps 100000 --agent td3</code>


# Experimental Results
| 2021-10-01 - 2022-01-01 | DDPG | TD3
| --- | --- | --- |
| Initial Value| 1000000 | 1000000 |
| Final Value |1246178  | 1235190 |
| Sharpe Ratio |2.24  | 2.15 |
| Max DrawDown | -10.1 | -9.6 |
<img src="https://github.com/christopher-hma/STOCKS_NEWS_RL_TRADING/blob/main/total_asset_value.png" width=150% height=100%>
