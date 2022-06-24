import time
import logging.handlers

from geopy import Nominatim
from utilities.utils import config, load_db
from geopy.extra.rate_limiter import RateLimiter


def setup_logging():
    """
    Email logging setup
    :return: logger
    """
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG,
                        filename='location_crawler.log')

    logger = logging.getLogger(__name__)
    smtp_handler = logging.handlers.SMTPHandler(
        mailhost=(config['MAIL_HOST'], config['MAIL_HOST_PORT']),
        fromaddr=config['EMAIL_ADDRESS'],
        toaddrs=[config['SENDTO']],
        subject='Location Crawler',
        credentials=(config['EMAIL_ADDRESS'], config['LOGGING_PASSWORD']),
        secure=()
    )

    logger.addHandler(smtp_handler)
    return logger


def get_place_annotations(entities):
    """
    Loads place annotations from entities
    :param entities:
    :return: place annotations
    """
    annotations = entities.get('annotations', None)

    if annotations is not None:
        types = [(annotation.get('type', None), annotation) for annotation in annotations]
        return [annotation for t, annotation in types if t == 'Place']
    return []


def handle_record(key, geocoder, cache, has_errored):
    """
    Geocodes the locations for each tweet.
    :param key: record key
    :param geocoder: used for geocoding
    :param cache: local cache of locations
    :param has_errored: whether an error has been experienced and this is a retry
    :return: has_errored
    """
    cached = cache.find_one({'key': key})

    if cached is None:
        try:
            location = geocoder(key)
            has_errored = False
        except Exception as exc:
            if has_errored:
                raise exc
            else:
                has_errored = True
                location = None

        if location is not None:
            cache.insert_one({'key': key, 'value': location.raw})
        else:
            cache.insert_one({'key': key, 'value': {}})
    return has_errored


def entity_location_geocoding():
    """
    Handles geocoding resolved entities.
    :return:
    """
    db = load_db()
    tweet_collection = db['rules_augmented']
    size = tweet_collection.count_documents({})
    locator = Nominatim(user_agent='location_geocoder')
    geocoder = RateLimiter(locator.geocode, min_delay_seconds=3, swallow_exceptions=False)
    cached_locations = db['locations']
    has_errored = False

    for i in range(int(size / 1_000)):
        entities = [tweet.get('data', {}).get('entities', {}) for tweet in
                    tweet_collection.find().skip(i * 1_000).limit(1_000)]
        entities = [entity for entity in entities if entity != {}]

        for entity in entities:
            places = get_place_annotations(entity)

            for place in places:
                key = place.get('normalized_text', None)
                if key is None:
                    continue

                has_errored = handle_record(key, geocoder, cached_locations, has_errored)


def user_location_geocoding():
    """
    Handles geocoding user locations.
    :return:
    """
    db = load_db()
    user_collection = db['users']
    size = user_collection.count_documents({})
    locator = Nominatim(user_agent='location_geocoder')
    geocoder = RateLimiter(locator.geocode, min_delay_seconds=3, swallow_exceptions=False)
    cached_locations = db['locations']
    has_errored = False

    for i in range(int(size / 1_000)):
        records = [x['data'].get('data', {}).get('location', '') for x in
                   user_collection.find().skip(i * 1_000).limit(1_000)]
        for record in records:
            has_errored = handle_record(record, geocoder, cached_locations, has_errored)


def main(logger):
    """
    Main function, includes retry logic
    :param logger:
    :return:
    """
    try:
        user_location_geocoding()
        logger.info('Finished user geocoding. Beginning entity geocoding')
        entity_location_geocoding()
    except Exception as e:
        logger.error('error', exc_info=e)
        exit(0)
    time.sleep(7_200)


if __name__ == '__main__':
    lg = setup_logging()
    while True:
        main(lg)
