"""
https://rest.aydinlik.com.tr/api/authors?page=1&count=200
"""


import requests
from datetime import datetime
import os
import json
import time


# requests.get
HOST = "https://rest.aydinlik.com.tr"
YAZAR = "api/authors?page=1&count=300"
ARTILCE = "https://rest.aydinlik.com.tr/api/authors/{0}/contents?page=1&count=100"


YAZAR_URL = "/".join((HOST, YAZAR))



# Download dir
DOWNLOAD_DIR = "./yazarlar"
# Log dir
LOG_DIR = "./log"
######################################################
FILE_DATE = datetime.now().strftime("%Y%m%d_%H%M")
# ----------------------------------------------------
def main():
    response = None
    ERR_LOG_PATH = "/".join((LOG_DIR, "err_{}.txt".format(FILE_DATE)))
    INFO_LOG_PATH = "/".join((LOG_DIR, "info_{}.txt".format(FILE_DATE)))

    with open(ERR_LOG_PATH, 'w') as log_err:
        with open(INFO_LOG_PATH, 'w') as log_inf:
            try:
                    response = requests.get(YAZAR_URL)
                    if response.status_code != 200:
                        raise Exception("http response status code: {}".format(response.status_code))

                    text = response.text
                    yazarlar_json = json.loads(text)
                    if yazarlar_json["success"]:
                        for author in yazarlar_json["data"]["authors"]:
                            download_author(author)
                            time.sleep(1)

            except Exception as e:
                raise e

#-----------------------------------------------------
def download_author(author):
    slug_url = author["slugUrl"]
    dir = "/".join(("yazarlar",slug_url))
    if not os.path.exists(dir):
        os.mkdir(dir)


    ARTICLES_URL = ARTILCE.format(slug_url)
    articles = requests.request(url=ARTICLES_URL, method="GET").text
    with open(dir +"/son100_{}.json".format(FILE_DATE),"w",encoding="utf-8") as file:
        file.write(articles)
        print(slug_url," completed")


# ----------------------------------------------------
def write_to_err(log_err, ex, response):
    log_err.write(str(ex))
    if response != None:
        response.close()


# ----------------------------------------------------
if __name__ == '__main__':
    main()
