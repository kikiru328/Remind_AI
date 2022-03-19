import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import time
class Chrome_driver:
    
    def install_driver_path():
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

        import chromedriver_autoinstaller
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        import os

        chrome_version = chromedriver_autoinstaller.get_chrome_version().split('.')[0]
        driver_path = f'{chrome_version}\chromedriver.exe'

        if os.path.exists(driver_path):
            print(f'chrome driver is installed : {driver_path}')
        else:
            print(f'chrome driver is installed : {driver_path}')
            print(f'install the chrome driver > version : {chrome_version}')
            print(f'install complete > {driver_path}')
            install_driver = chromedriver_autoinstaller.install(True)
            print(driver_path)
            
        return driver_path
        
    def chrome_option():
        # re_text
        from selenium.webdriver.chrome.options import Options
        import os
        
        chrome_options = Options()
        # chrome_options.add_argument('--headless')
        # chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
        # chrome_options.add_argument(f"--window_size")
        
        # driver = webdriver.Chrome(executable_path = driver_path, chrome_options=chrome_options)
        return chrome_options

def urls():
    urls = ['https://www.thewhiskyexchange.com/c/40/single-malt-scotch-whisky?pg=1']
    return urls


