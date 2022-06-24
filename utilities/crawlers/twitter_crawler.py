import json
import time
import requests
import logging.handlers

from datetime import datetime
from utilities.utils import bearer_oauth, config, load_db, load_rules


def setup_logging():
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG,
                        filename='twitter_crawler.log')

    logger = logging.getLogger(__name__)
    smtp_handler = logging.handlers.SMTPHandler(
        mailhost=(config['MAIL_HOST'], config['MAIL_HOST_PORT']),
        fromaddr=config['EMAIL_ADDRESS'],
        toaddrs=[config['SENDTO']],
        subject='Twitter Crawler',
        credentials=(config['EMAIL_ADDRESS'], config['LOGGING_PASSWORD']),
        secure=()
    )

    logger.addHandler(smtp_handler)
    return logger


def get_rules():
    response = requests.get(
        'https://api.twitter.com/2/tweets/search/stream/rules',
        auth=bearer_oauth
    )

    if response.status_code != 200:
        error = 'Cannot get rules (HTTP {}): {}'.format(response.status_code, response.text)
        raise Exception(error)

    if config['MODE'] == 'DEBUG':
        print(json.dumps(response.json()))
    return response.json()


def delete_all_rules(rules):
    if rules is None or 'data' not in rules:
        return None

    ids = [rule['id'] for rule in rules['data']]
    payload = {'delete': {'ids': ids}}
    response = requests.post(
        'https://api.twitter.com/2/tweets/search/stream/rules',
        auth=bearer_oauth,
        json=payload
    )

    if response.status_code != 200:
        error = 'Cannot delete rules (HTTP {}): {}'.format(response.status_code, response.text)
        raise Exception(error)

    if config['MODE'] == 'DEBUG':
        print(json.dumps(response.json()))


def set_rules(rules):
    payload = {'add': rules}
    response = requests.post(
        'https://api.twitter.com/2/tweets/search/stream/rules',
        auth=bearer_oauth,
        json=payload
    )

    if response.status_code != 201:
        error = 'Cannot add rules (HTTP {}): {}'.format(response.status_code, response.text)
        raise Exception(error)

    if config['MODE'] == 'DEBUG':
        print(json.dumps(response.json()))


def get_stream(handler, logger):
    response = requests.get(
        'https://api.twitter.com/2/tweets/search/stream?'
        'tweet.fields=created_at,geo,author_id,context_annotations,possibly_sensitive,'
        'public_metrics,non_public_metrics,lang,entities&expansions=author_id,'
        'geo.place_id&place.fields=contained_within,country,country_code,full_name,geo,id,name,'
        'place_type&user.fields=created_at,description,entities,id,'
        f'location,name,pinned_tweet_id,profile_image_url,protected,public_metrics,url,'
        f'username,verified,withheld',
        auth=bearer_oauth,
        stream=True,
        timeout=30
    )

    print('Getting stream response:', response.status_code)

    if response.status_code != 200:
        error = 'Cannot get stream (HTTP {}): {}'.format(response.status_code, response.text)
        raise Exception(error)

    for response_line in response.iter_lines():
        if response_line:
            try:
                handler(json.loads(response_line))
            except Exception as exc:
                logger.error('twitter crawler error', exc_info=exc)


def handle_json(db, data):
    tweet_id = data.get('data', {}).get('id', None)

    if tweet_id is None:
        return

    rules_collection = db['rules_augmented']
    times_collection = db['times']
    user_collection = db['users']
    place_collection = db['places']

    includes = data.get('includes', {})
    users = includes.get('users', [])
    places = includes.get('places', [])

    if includes != {}:
        data['includes'] = []

    rules_collection.insert_one(data)

    for user in users:
        cached = user_collection.find_one({'user_id': user['id']})
        if cached is None:
            user_collection.insert_one({'user_id': user['id'], 'data': user})

    for place in places:
        cached = place_collection.find_one({'place_id': place['id']})
        if cached is None:
            place_collection.insert_one({'place_id': place['id'], 'data': place})

    times_collection.insert_one({'id': tweet_id, 'datetime': datetime.utcnow()})


def twitter_handler(logger):
    rules = get_rules()
    db = load_db()
    delete_all_rules(rules)
    set_rules(rules=load_rules())
    sleep_counter = -1
    while sleep_counter < 10:
        try:
            get_stream(lambda x: handle_json(db, x), logger)
        except requests.RequestException:
            time.sleep(0 if sleep_counter == -1 else pow(2, sleep_counter))
            sleep_counter += 1
    raise EnvironmentError('Overloaded sleep counter')


if __name__ == '__main__':
    lg = setup_logging()
    try:
        twitter_handler(lg)
    except Exception as e:
        lg.error('Error', exc_info=e)
        exit(0)
