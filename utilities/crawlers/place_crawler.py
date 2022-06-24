import time
import datetime
import requests
import logging.handlers

from pymongo.database import Database
from utilities.utils import bearer_oauth, config, load_db


def setup_logging():
    """
    Email logging setup
    :return: logger
    """
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG,
                        filename='place_crawler.log')

    logger = logging.getLogger(__name__)
    smtp_handler = logging.handlers.SMTPHandler(
        mailhost=(config['MAIL_HOST'], config['MAIL_HOST_PORT']),
        fromaddr=config['EMAIL_ADDRESS'],
        toaddrs=[config['SENDTO']],
        subject='Place Crawler',
        credentials=(config['EMAIL_ADDRESS'], config['LOGGING_PASSWORD']),
        secure=()
    )

    logger.addHandler(smtp_handler)
    return logger


def update_place(tweet_id, place_id, db: Database):
    """
    Resolves place.
    :param tweet_id: id of tweet
    :param place_id: id of place
    :param db: connection to remote database
    :return: whether Twitter API was hit
    """
    place_collection = db['places']
    cached = place_collection.find_one({'place_id': place_id})

    if cached is not None and cached != []:
        return False

    response = requests.get(f'https://api.twitter.com/2/tweets?ids={tweet_id}&expansions=geo.place_id&place.fields'
                            f'=contained_within,country,country_code,full_name,geo,id,name,place_type',
                            auth=bearer_oauth)

    if response.status_code == 429:
        raise OverloadException()
    elif response.status_code != 200:
        raise ConnectionError('Cannot get (HTTP {}): {}'.format(response.status_code, response.text))

    db['places'].insert_one({'place_id': place_id, 'data': response.json()})
    return True


def crawl_places():
    """
    Crawls all stored places for resolving.
    :return:
    """
    db = load_db()
    table_size = db['rules_augmented'].count_documents({})
    rng = int(table_size / 10_000)

    for i in range(rng):
        raw = list(db['rules_augmented'].find().skip(skip=i * 10_000).limit(limit=10_000))

        for x in raw:
            if not x.get('includes', {}).get('places', []):
                continue
            tweet_id = x['data']['id']
            place_id = x['data'].get('geo', {}).get('place_id', None)

            hit_api = False
            if place_id is not None:
                hit_api = update_place(tweet_id, place_id, db)
            if hit_api:
                time.sleep(2)


def main(logger):
    """
    Main function for crawling places.
    Contains logic for handling crawler errors.
    :param logger:
    :return:
    """
    tries = 0
    has_errored = False
    while tries < 10:
        try:
            crawl_places()
        except OverloadException:
            time.sleep(2 ** tries)
            tries += 1
            has_errored = True
        except ConnectionError as e:
            logger.error('error', exc_info=e)
            exit(0)
    if has_errored:
        logger.warning(f'Sleeping for 15 minutes at {datetime.datetime.now()}')
        time.sleep(900)
    else:
        time.sleep(7_200)


class OverloadException(Exception):
    """
    Custom exception for handling API overloads.
    """
    pass


if __name__ == '__main__':
    lg = setup_logging()
    while True:
        main(lg)
