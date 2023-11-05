import pandas as pd

import sys
import logging


logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

logging.info('Fetching data...')


fake_news_dataset = pd.read_csv("../data/fake_news.csv")
esp_fake_news = pd.read_csv("../data/fakes1000.csv")

df1 = fake_news_dataset[["texto", "fake_news"]].rename(columns={"texto":"Text","fake_news":"label"})
df2 = esp_fake_news.rename(columns={"class":"label"})

full_data = pd.concat([df1, df2])
full_data['label'] = full_data['label'].apply(lambda x: "FAKE" if x is False else "REAL")

full_data.to_csv("./full_data.csv")


logger.info('Data Fetched and prepared...')