import scrapy
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import scrapy
import logging
from scrapy.http import HtmlResponse
from scrapy_selenium import SeleniumRequest
from bs4 import BeautifulSoup as bs
import pandas as pd
from selenium.webdriver.common.keys import Keys
import re
from urllib import parse
from selenium import webdriver
import os
import pandas as pd
import time
from selenium.common.exceptions import NoSuchElementException

class BasicmodelSpider(scrapy.Spider):
    
    """
    사용법은 terminal scrapy crawl {지정명} 입니다. (해당 폴더 경로로 이동해야 됩니다.)
    해당 파일을 예시로 들면
    scrapy crawl FF
    로 실행합니다.
    """
    name = 'acu'

    def start_requests(self):
        
        """
        Chrome option 적용
        """
        options = Options()
        options.add_argument('--headless') #headless모드 브라우저가 뜨지 않고 실행됩니다.
        # options.add_argument('--window-size= x, y') #실행되는 브라우저 크기를 지정할 수 있습니다.
        # options.add_argument('--start-maximized') #브라우저가 최대화된 상태로 실행됩니다.
        # options.add_argument('--start-fullscreen') #브라우저가 풀스크린 모드(F11)로 실행됩니다.
        # options.add_argument('--blink-settings=imagesEnabled=false') #브라우저에서 이미지 로딩을 하지 않습니다.
        # options.add_argument('--mute-audio') #브라우저에 음소거 옵션을 적용합니다.
        # options.add_argument('incognito') #시크릿 모드의 브라우저가 실행됩니다.
        options.add_experimental_option("excludeSwitches", ["enable-logging"]) #selenium 작동 안될 때
        self.driver = webdriver.Chrome(executable_path=ChromeDriverManager().install(),chrome_options=options)    
    
            ##  start url
        """
        selenium이 시작할 최초의 url입니다.
        """
        # url = {url}
        # self.driver.get(url)
        
        """
        여러 urls도 가능합니다.
        """
        # urls = pd.read_csv(r'C:\Users\baeky\OneDrive\바탕 화면\study\study\practice_code\NLP\Scrapy\down\down\arcu.csv',index_col=0)
        # for url in urls['0']:
        #     self.driver.get(url)

        url_list = [
        'https://www.kmcric.com/database/acupoint/LU/LU1',
        'https://www.kmcric.com/database/acupoint/LU/LU2'
        ]
        for url in url_list:
            self.driver.get(url)
        
        
        """
        self.driver 이하부터 selenium의 동적 크롤링 코드를 작성하시면 됩니다. (url 포함)
        
        selenium 동적 크롤링 코드가 마무리 되면, 
        
        yield scrpy.Request(url=url, callback=self.parse) 
        로 해당 페이지의 정보를 parse로 전송합니다.
        """
        
    def parse(self,response):
        
########## Crawling Object setting #####
        """
        selenium으로 크롤링할 페이지가 적용이 되면,
        Beautifulsoup을 이용하여 html, lxml등 page_source를 받아오면 됩니다.
        
        이후 원하는 정보의 tag를 css,xpath등 방법을 활용하여 코드를 작성하시면 됩니다.
        
        특히 list에 포함한 후, for문을 활용,
        yield를 사용하면 각각 한개씩 해당 column 에 추가됩니다. (csv,tsv등)
        """
        soup0 = self.driver.page_source
        soup = bs(soup0,'html')

        big = self.driver.find_element_by_xpath('//*[@id="sub"]/section[2]/div[2]/div/table/thead/tr/td/div/font[1]').text()
        
        yield{
            print(big)
        }












        # product = response.css('li.product-grid__item')
        # for item in product:
        #     yield{
        #         'name' :item.css('p.product-card__meta::text').get()
        #             }
    
    ## SETTING ##
    
        """
        Crawling에 필요한 settings과
        저장방식을 설정하는 곳입니다.
        """
        
        
    Save_path = '../data/' # 절대경로 사용 불가 (적용이 잘 안되네요. \ -> / 로 바꾸어서 상대경로로 사용하면 가능합니다.)
    Save_file_name = '{원하시는 파일이름}' 
    Save_extension = '{확장자}'

    if not os.path.exists(Save_path):
        os.makedirs(Save_path)
        
    if not os.path.exists('../Log/'):
        os.makedirs('../Log/')        
            
            
    custom_settings = {
        # Detour selenium robots
        ## Invaild page crawling == false
        'ROBOTSTXT_OBEY' : False,
        ## take time to download for detour robots
        'DOWNLOAD_DELAY' : 1,
        'LOG_LEVEL' : 'INFO',
        'LOG_STDOUT' : True,
        'LOG_FILE' : '../Log/Log.txt',
        # save settings
        
        """
        LOG 파일도 저장 및 보기 상태를 설정할 수 있습니다.
        log_level을 INFO로 지정하면, terminal에는 가장 기본적인 내용만 보여지고,
        LOG 파일에 DEBUG포함 모든 것이 txt파일로 저장됩니다.
        """
        
        
        ## scrapy crawl name -o {filename.extension}
        'FEEDS' : {
            # Save file name and extension
            Save_path + Save_file_name + '.' + Save_extension : {
                # format : extension
                'format': Save_extension
            }
        }
    }            
        