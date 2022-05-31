import os
import gdown
def download_news():
    
    urls = ['https://drive.google.com/uc?id=1ZKRMwuls1AnDZYyajKwi2HfrVbpMp3Lb','https://drive.google.com/uc?id=1ZwroRRpSdBugzLTy81oKha4UmxlSJn94']
       
    news_tensor = os.path.join(os.getcwd(),"news_tensor.pkl")
    
    news_db = os.path.join(os.getcwd(),"financial_data.db")
    
    destinations = [news_tensor,news_db]
    
    for url,dest in zip(urls,destinations):
        
        gdown.download(url, dest, quiet=False)


if __name__ == "__main__":

   download_news()