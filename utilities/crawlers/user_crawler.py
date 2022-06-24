import time
import datetime
import requests
import logging.handlers

from pymongo.database import Database
from utilities.utils import bearer_oauth, config, load_db


def setup_logging():
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG,
                        filename='user_crawler.log')

    logger = logging.getLogger(__name__)
    smtp_handler = logging.handlers.SMTPHandler(
        mailhost=(config['MAIL_HOST'], config['MAIL_HOST_PORT']),
        fromaddr=config['EMAIL_ADDRESS'],
        toaddrs=[config['SENDTO']],
        subject='User Crawler',
        credentials=(config['EMAIL_ADDRESS'], config['LOGGING_PASSWORD']),
        secure=()
    )

    logger.addHandler(smtp_handler)
    return logger


def update_user(user_id, db: Database):
    user_collection = db['users']
    cached = user_collection.find_one({'user_id': user_id})

    if cached is not None and cached != []:
        return False

    response = requests.get(f'https://api.twitter.com/2/users/{user_id}?user.fields=created_at,description,entities,'
                            f'id,location,name,pinned_tweet_id,profile_image_url,protected,public_metrics,url,'
                            f'username,verified,withheld', auth=bearer_oauth)

    if response.status_code == 429:
        raise OverloadException()
    elif response.status_code != 200:
        raise ConnectionError('Cannot get (HTTP {}): {}'.format(response.status_code, response.text))

    db['users'].insert_one({'user_id': user_id, 'data': response.json()})
    return True


def crawl_users():
    db = load_db()
    table_size = db['rules_augmented'].count_documents({})
    rng = int(table_size / 10_000)

    for i in range(rng):
        raw = list(db['rules_augmented'].find().skip(skip=i * 10_000).limit(limit=10_000))

        for x in raw:
            if not x.get('includes', {}).get('users', []):
                continue
            user_id = x['data']['author_id']

            hit_api = update_user(user_id, db)
            if hit_api:
                time.sleep(2)


def main(logger):
    tries = 0
    has_errored = False

    while tries < 10:
        try:
            crawl_users()
        except OverloadException:
            time.sleep(2 ** tries)
            tries += 1
            has_errored = True
        except ConnectionError as e:
            logger.error('error', exc_info=e)
            exit(0)
    if has_errored:
        logger.debug(f'Sleeping for 15 minutes at {datetime.datetime.now()}')
        time.sleep(900)
    else:
        time.sleep(7_200)


class OverloadException(Exception):
    pass


if __name__ == '__main__':
    lg = setup_logging()
    while True:
        main(lg)
