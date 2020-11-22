import os
import random
import time

from bs4 import BeautifulSoup
import requests


def get_celebrity_img_urls(url):
    headers = {
        "Connection": "keep-alive",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.6821.400 QQBrowser/10.3.3040.400",
        "Upgrade-Insecure-Requests": "1",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Cookie": "__51cke__=; UM_distinctid=169c7147623b32-041b6f54ccddbe-3257487f-232800-169c7147624ddd; CNZZDATA30054349=cnzz_eid%3D515205138-1553821270-https%253A%252F%252Fwww.baidu.com%252F%26ntime%3D1553821270; PHPSESSID=9btdnv30htpj54ies9em19pan1; right_adv=%7B%22time%22%3A%222019329%22%2C%22number%22%3A20%7D; __tins__18838395=%7B%22sid%22%3A%201553825001869%2C%20%22vd%22%3A%203%2C%20%22expires%22%3A%201553827085186%7D; __51laig__=21",
        # "Referer": url,
    }

    response = requests.get(url, headers=headers)
    html = response.text

    soup = BeautifulSoup(html, 'lxml')

    lst_imgs = []
    try:
        for item in soup.find("ul", class_="page_starphoto").find_all("img"):
            lst_imgs.append(item["src"])
        # print(item["src"])
    except:
        pass
    return lst_imgs


def get_celebrities_one_page(url, idx_page):
    headers = {
        "Connection": "keep-alive",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.6821.400 QQBrowser/10.3.3040.400",
        "Upgrade-Insecure-Requests": "1",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Cookie": "__51cke__=; UM_distinctid=169c7147623b32-041b6f54ccddbe-3257487f-232800-169c7147624ddd; CNZZDATA30054349=cnzz_eid%3D515205138-1553821270-https%253A%252F%252Fwww.baidu.com%252F%26ntime%3D1553821270; PHPSESSID=9btdnv30htpj54ies9em19pan1; right_adv=%7B%22time%22%3A%222019329%22%2C%22number%22%3A20%7D; __tins__18838395=%7B%22sid%22%3A%201553825001869%2C%20%22vd%22%3A%203%2C%20%22expires%22%3A%201553827085186%7D; __51laig__=21",
    }

    params = {
        "p": idx_page
    }
    response = requests.get(url, params=params, headers=headers)
    html = response.text
    # print(html)

    soup = BeautifulSoup(html, 'lxml')
    # print(soup.find("div", class_="page_starlist").find_all("img"))

    lst_celebrities = []
    for item in soup.find("div", class_="page_starlist").find_all("img"):
        lst_celebrities.append({"name": item.get("alt").strip(),
                                "url": "http://www.mingxing.com" + item.find_parent("a").get("href"),
                                "img_urls": [item.get("src")]})
        print(item.find_parent("a")["href"])
        # print(item["src"], item["alt"])

    return lst_celebrities


def get_img(url, path):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.6821.400 QQBrowser/10.3.3040.400",
        "Referer": url,
    }

    response = requests.get(url, headers=headers)
    # print(response.content)

    with open(path, "wb") as fw:
        fw.write(response.content)


def get_celebrities(url, num_pages):
    lst_celebrities = []
    for idx_page in range(60, num_pages):
        lst_celebrities.extend(
            get_celebrities_one_page(url, idx_page))
        # time.sleep(0.5)
    return lst_celebrities


# URL_MINGXING_CELEBRITY_LIST = "http://www.mingxing.com/ziliao/index"
URL_MINGXING_CELEBRITY_LIST = "http://www.mingxing123.com/ziliao/index/type/neidi"

if __name__ == "__main__":
    NUM_PAGES = 116
    DATASET_PATH = "./dataset"
    # 明星列表
    lst_celebrities = get_celebrities(URL_MINGXING_CELEBRITY_LIST, NUM_PAGES)

    for celebrity in lst_celebrities:

        # 明星文件夹
        celebrity_dir = os.path.join(DATASET_PATH, celebrity["name"])
        print("*" * 10)
        print("celebrity: {}".format(celebrity["name"]))

        if not os.path.exists(celebrity_dir):
            os.makedirs(celebrity_dir)

        # 明星页面
        celebrity["img_urls"].extend(get_celebrity_img_urls(celebrity["url"]))

        idx_img = 0
        for img_url in celebrity["img_urls"]:
            idx_img += 1
            img_path = os.path.join(celebrity_dir, celebrity["name"]
                                    + "_{:04d}.jpg".format(idx_img))
            get_img(img_url, img_path)
            print("download {} ---> {}".format(img_url, img_path))
            time.sleep(random.uniform(0, 0.5))
