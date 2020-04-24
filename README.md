 # SPbU Information Retrieval

 Simple ML project based on [The Most Followed Accounts on Twitter analysis](https://www.brandwatch.com/blog/most-twitter-followers/)

## Scraping

Data was obtained using [scrapy web crawler](https://scrapy.org/). To reproduce results you need to do the following:

1. Create file `users.txt` (in scraper folder) on each file of which should be a username.

    For example:
    ```
    jimmyfallon
    Cristiano
    narendramodi
    ```
2. Go to the scraper dir and run the following command:
    ```
    scrapy crawl twitter -a users_file=users.txt
    ```

3. The data will be located in the folder specified by SAVE_PATH variable in `settings.py`. For each user, the scraper will create jsonl file (for example, `jimmyfallon.jsonl`)

## Basic preprocessing

Basic preprocessing class located in `src/utils/preprocessing.py` script.

To do basic preprocessing run the following command:
```
make basic_preprocess
```
It will create a preprocessed file in `data/interim` folder.
