from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm
import wget

import time
import re
import os


def main():

    dataset_urls = [
                     'https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/code?datasetId=551982&sortBy=voteCount',
                     'https://www.kaggle.com/jessicali9530/animal-crossing-new-horizons-nookplaza-dataset/code?datasetId=661950&sortBy=voteCount',
                     'https://www.kaggle.com/mlg-ulb/creditcardfraud/code?sortBy=voteCount',
                     'https://www.kaggle.com/shivamb/netflix-shows/code?sortBy=voteCount',
                     'https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset/code?sortBy=voteCount',
                     'https://www.kaggle.com/ronitf/heart-disease-uci/code?sortBy=voteCount',
                     'https://www.kaggle.com/datasnaek/youtube-new/code?sortBy=voteCount',
                     'https://www.kaggle.com/gregorut/videogamesales/code?sortBy=voteCount',
                     'https://www.kaggle.com/lava18/google-play-store-apps/code?sortBy=voteCount',
                     'https://www.kaggle.com/karangadiya/fifa19/code?sortBy=voteCount',
                     'https://www.kaggle.com/hugomathien/soccer/code?sortBy=voteCount',
                     'https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/code?sortBy=voteCount',
                     'https://www.kaggle.com/unsdsn/world-happiness/code?sortBy=voteCount',
                     'https://www.kaggle.com/zynicide/wine-reviews/code?sortBy=voteCount',
                     'https://www.kaggle.com/spscientist/students-performance-in-exams/code?sortBy=voteCount',
                     'https://www.kaggle.com/tmdb/tmdb-movie-metadata/code?sortBy=voteCount',
    ]
    for dataset_url in dataset_urls_test:
        dataset_name = re.search('/([^/]*)/code', dataset_url).groups()[0]
        print('='*50, dataset_name, '='*50)

        driver = webdriver.Firefox()
        driver.set_window_size(1000, 1000)
        driver.get(dataset_url)

        WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, "//i[text() = 'arrow_drop_up']")))
        time.sleep(3)

        for i in range(10):
            elem = driver.find_element_by_id('site-content')
            elem.send_keys(Keys.END)
            time.sleep(5)
            elem.send_keys(Keys.HOME)
            time.sleep(5)

        download_elems = driver.find_elements_by_xpath("//a[starts-with(@href,'https://www.kaggle.com/kernels/scriptcontent')]")
        download_urls = [i.get_attribute('href') for i in download_elems]
        driver.quit()

        save_path = f'kaggle_notebooks_test/{dataset_name}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print('Total Number of scripts:', len(download_urls))
        with open(save_path+'urls.txt', 'w') as f:
            f.write('\n'.join(download_urls))

        i = 0
        for url in tqdm(download_urls):
            if i > 100:
                break
            time.sleep(1)
            file_name = wget.download(url, out=save_path)
            if file_name.lower().endswith('.py') or file_name.lower().endswith('.ipynb'):
                i += 1
                # print('+', end='')
            else:
                # keep only python or jupyter files
                os.remove(file_name)
        print(':', i)

if __name__ == '__main__':
    main()
