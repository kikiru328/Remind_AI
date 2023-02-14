import scrapy
import scrapy
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append('../')
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import time
import importlib
import chrome_driver_auto
from bs4 import BeautifulSoup as bs
import pandas as pandas
from selenium.webdriver.common.keys import Keys
import re
from urllib import parse
from scrapy_selenium import SeleniumRequest 


class XSpider(scrapy.Spider):
    name = 'youtube'
    
    ##  start url 
    
    
    start_urls = 'https://www.youtube.com/'
    
    # fixed (selenium)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        driver_path = chrome_driver_auto.Chrome_driver.install_driver_path()
        
        ############## CHROME OPTION ##############
        chrome_options = Options()
        # chrome_options.add_argument('--headless')
        # chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
        # chrome_options.add_argument(f"--window_size")        
    
        self.driver = webdriver.Chrome(executable_path = driver_path, chrome_options=chrome_options)
        # self.driver.get(pass_py.urls()[0])
    
    
    def parse(self,response):
        
        # using selenium, driver always to self.driver
        
        ############### start_selenium ##############
        # self.driver.get(self.start_urls[0])
        urls = []
        keywords = [parse.quote("금연정책")]
        
        for keyword in keywords:
            urls.append(self.start_urls+ 'results?search_query={}&sp=CAISAhAB'.format(keyword))
            ############### movement setting ############    
        for url in urls:
            self.driver.get(url)
            
            title_list = []
            url_list = [] 
            date_list = []       
            
            body = self.driver.find_element_by_xpath('/html/body')
            while True:
                if self.driver.find_element_by_id("message").text.strip() == '결과가 더 이상 없습니다.':
                    print("결과가 더 이상 없습니다.")
                    break
            body.send_keys(Keys.PAGE_DOWN)
            
            soup = bs(self.driver.page_source, 'html.parser')
            
            ############### Crawling Object setting #####
            
            title = soup.select('a#video-title') # 유튜브 동영상 제목
            video_url = soup.select('a#video-title') # 유튜브 동영상 URL
            yDate = soup.select('a#video-title') # 유튜브 업로드 날짜
            
            for i in range(len(title)):
                yTitle= title[i].text.strip()
                print(yTitle)
                yTitle = re.sub(r"[^ㄱ-ㅎ|ㅏ-ㅣ|가-힣|A-Z|a-z|0-9| ]", " ", yTitle)
                yTitle = re.sub(" {2,}", " ", yTitle)
                
                if len(yTitle) < 2:
                    continue
                urlDate= yDate[i].get('aria-label')
                youtubeDateOrg = youtubeDateSearchKey.search(urlDate)
            
                url_list.append('{}{}'.format('https://www.youtube.com',video_url[i].get('href')))
            
                try:
                    uDate = youtubeDateOrg.group().split(' ')[-2] + " 전"
                    print(uDate)
                    title_list.append(yTitle)
                    date_list.append(uDate)
                except: 
                    title_list.append("Null")
                    date_list.append("Null")
                
            

        yield{
            '제목': title_list,
            '주소': url_list,
            '날짜': date_list
                    }
    
    ## SETTING ##
    custom_settings = {
        
        # Detour selenium robots
        ## Invaild page crawling == false
        'ROBOTSTXT_OBEY' : False,
        ## take time to download for detour robots
        'DOWNLOAD_DELAY' : 1,
        
        # save settings
        'FEEDS' : {
            # Save file name and extension
            'youtube.csv' : {
                # format : extension
                'format': 'csv'
            }
        }
    }
    

        
    
    
    
    
    
    
    
    
    
    # def start_requests(self):
    #     urls = ['https://movie.naver.com/movie/running/current.nhn']
    #     for url in urls:
    #         yield scrapy.Request(url=url, callback=self.parse)
    # def parse(self, response):
    #     # pass
    #     movie_sels = response.css('ul.lst_detail_t1 > li')
    #     for movie_sel in movie_sels:
    #         title= movie_sel.css('.tit > a::text').get()
    #         print('tit' , title)