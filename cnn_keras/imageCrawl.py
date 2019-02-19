import urllib.request
import random
from selenium import webdriver
from bs4 import BeautifulSoup
import re
class AppURLopener(urllib.request.FancyURLopener):
    version = "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.69 Safari/537.36"
urllib._urlopener = AppURLopener()
def download_img(datas,cnt):
    full_name = "./images/"+"athlete_"+str(cnt)+".png"
    urllib._urlopener.retrieve(str(datas),full_name)


browser = webdriver.Chrome('/home/minkj1992/code/facial_project/chromedriver')
browser.implicitly_wait(3)
url = "https://asiangames2018.id/athletes/detail/"
for cnt,j in enumerate(random.sample(range(1,1000),300)):
    try:
        browser.get(url+str(j))
        html = browser.page_source
        soup = BeautifulSoup(html, 'html.parser')
        # select_one 쓰는게 핵심
        datas = soup.select_one('body > div.sports-detail-wrapper > div.container-sport-detail.col-xs-12 > div.container-sport-detail-flag-desc.col-xs-8.col-xs-offset-2 > div.left.col-md-6 > div > img[src]')
        download_img(datas['src'],cnt)
    except:
        # 사진이 없을 경우 또는 id 가 없을 경우
        print("pass happend!")
        pass

