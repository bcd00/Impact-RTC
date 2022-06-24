import math
import pandas as pd

from tqdm import tqdm
from typing import Optional
from pymongo.database import Database
from utilities.utils import cache_dir


def load(db: Optional[Database], use_cache, limit=None):
    """
    Loads data from cache and remote and updates cache.
    Data is loaded from remote in chunks to improve throughput.
    :param db: connection to remote database
    :param use_cache: whether to only use local cache
    :param limit: size of chunk to load from remote
    :return: tweets, users, places, locations - all DataFrames of database records
    """
    print(f'Loading data{" and updating caches" if not use_cache else ""}')
    if use_cache:
        tweets = pd.read_pickle(f'{cache_dir}/tweets.pickle')
        users = pd.read_pickle(f'{cache_dir}/users.pickle')
        places = pd.read_pickle(f'{cache_dir}/places.pickle')
        locations = pd.read_pickle(f'{cache_dir}/locations.pickle')
    else:
        cached_tweets = [pd.read_pickle(f'{cache_dir}/tweets.pickle')]
        cached_users = [pd.read_pickle(f'{cache_dir}/users.pickle')]
        cached_places = [pd.read_pickle(f'{cache_dir}/places.pickle')]
        cached_locations = [pd.read_pickle(f'{cache_dir}/locations.pickle')]

        cached_tweets_size = len(cached_tweets[0])
        cached_users_size = len(cached_users[0])
        cached_places_size = len(cached_places[0])
        cached_locations_size = len(cached_locations[0])

        print(('-' * 20) + '\n')
        print(f'Cached Tweets\' Size: {cached_tweets_size}')
        print(f'Cached Users\' Size: {cached_users_size}')
        print(f'Cached Places\' Size: {cached_places_size}')
        print(f'Cached Locations\' Size: {cached_locations_size}')
        print('\n' + ('-' * 20))

        print(('-' * 20) + '\n')
        print(f'Total Tweets: {db["rules_augmented"].count_documents({})}')
        print(f'Total Users: {db["users"].count_documents({})}')
        print(f'Total Places: {db["places"].count_documents({})}')
        print(f'Total Locations: {db["locations"].count_documents({})}')
        print('\n' + ('-' * 20))

        loaded_tweets = [pd.DataFrame(x) for x in load_table(db, 'rules_augmented', cached_tweets_size, limit=limit)]
        loaded_users = [pd.DataFrame(x) for x in load_table(db, 'users', cached_users_size, limit=limit)]
        loaded_places = [pd.DataFrame(x) for x in load_table(db, 'places', cached_places_size, limit=limit)]
        loaded_locations = [pd.DataFrame(x) for x in load_table(db, 'locations', cached_locations_size, limit=limit)]

        tweets = cached_tweets + loaded_tweets
        users = cached_users + loaded_users
        places = cached_places + loaded_places
        locations = cached_locations + loaded_locations

        tweets = pd.concat(tweets)
        users = pd.concat(users)
        places = pd.concat(places)
        locations = pd.concat(locations)

    tweets_size = len(tweets)
    users_size = len(users)
    places_size = len(places)
    locations_size = len(locations)

    if not use_cache:
        tweets.to_pickle(f'{cache_dir}/tweets.pickle')
        users.to_pickle(f'{cache_dir}/users.pickle')
        places.to_pickle(f'{cache_dir}/places.pickle')
        locations.to_pickle(f'{cache_dir}/locations.pickle')

    print(('-' * 20) + '\n')
    print(f'Tweets\' Columns: {tweets.columns}')
    print(f'Tweets\' Size: {tweets_size}')
    print(f'Users\' Columns: {users.columns}')
    print(f'Users\' Size: {users_size}')
    print(f'Places\' Columns: {places.columns}')
    print(f'Places\' Size: {places_size}')
    print(f'Locations\' Columns: {locations.columns}')
    print(f'Locations\' Size: {locations_size}')
    print('\n' + ('-' * 20))

    return tweets, users, places, locations


def load_table(db: Database, table_name, cache_size, limit=None):
    """
    Loads a table from the remote database in chunks.
    :param db: connection to remote database
    :param table_name: name of the table to load
    :param cache_size: size of the existing cache to ensure only fresh records are fetched
    :param limit: size of each chunk to load
    :return: chunks as a list of JSON records
    """
    table_size = db[table_name].count_documents({})

    rng = math.ceil((table_size - cache_size) / 10_000)
    rng = rng if limit is None or rng < limit else limit
    chunks = []

    for i in tqdm(range(rng)):
        chunks.append(list(db[table_name].find().skip(skip=cache_size + (i * 10_000)).limit(limit=10_000)))

    return chunks
