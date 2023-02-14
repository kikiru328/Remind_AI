import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class Chrome_driver:

    def install_driver_path():
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
        
