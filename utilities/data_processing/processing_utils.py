import typing
import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from tqdm import tqdm
from pathlib import Path
from os.path import exists
from itertools import combinations
from preprocessing import PreProcessing
from wordcloud import WordCloud, STOPWORDS
from utilities.utils import load_rules, write_json, output_dir, shared_dir, rules_dir, geojson_dir


def get_agreement(annotations_df: pd.DataFrame, y_hat: int, y: int):
    """
    Gets the number of samples each of three annotators shares with the consensus truth
    :param annotations_df: dataset containing three annotators' labels
    :param y_hat: annotator's label
    :param y: consensus truth
    :return: three annotators' agreement with consensus labels
    """
    return [
        len(annotations_df[(annotations_df.label_one == y_hat) & (annotations_df.label == y)]),
        len(annotations_df[(annotations_df.label_two == y_hat) & (annotations_df.label == y)]),
        len(annotations_df[(annotations_df.label_three == y_hat) & (annotations_df.label == y)])
    ]


def stack_df(df, key, stk=None, drop=None, set_index=None):
    """
    Stacks dataframe by duplicating for every item in a list within a dataframe field
    :param df: dataframe to stack
    :param key: label of list field to stack
    :param stk: pre-saved series acting as indexing stack, default is None
    :param drop: optional columns to drop from stacked dataframe, default is None
    :param set_index: optional index for reindexing, default is None
    :return: stacked dataframe and indexing series
    """
    stk = stk if stk is not None else df[key].apply(pd.Series).stack().rename(key).reset_index()
    ndf = pd.merge(stk, df, left_on='id', right_index=True, suffixes=('', '_old'))
    if drop is not None:
        ndf.drop(drop, axis=1, inplace=True)
    if set_index is not None:
        ndf.set_index(set_index, inplace=True)
    return ndf, stk


def display_rules(df, color, filename=None, stk=None, show=True):
    """
    Displays rules as a Plotly histogram
    :param df: dataframe with rules
    :param color: color of graph
    :param filename: optional filename to save figure to, default is None
    :param stk: optional cached indexing series for improving runtime, default is None
    :param show: whether to show the histogram, default is True
    :return: figure, stacked dataframe, indexing series
    """
    print('Beginning Merge')

    r_df, stk = stack_df(df, 'rules', stk)

    print('Processing Rule Suffixes')
    r_df['rule'] = r_df['rules'].progress_apply(lambda x: x.split('_')[-1])
    r_df['rule'] = r_df['rule'].astype(int)

    rules_fig = px.histogram(r_df, x="rule", log_y=True, color_discrete_sequence=[color])

    if filename is not None:
        rules_fig.write_json(f'{rules_dir}/rules_{filename}.json')
        rules_fig.write_html(f'{rules_dir}/rules_{filename}.html')
    if show:
        rules_fig.show()
    return rules_fig, r_df, stk


def display_rules_text(df):
    """
    Displays samples for each rule
    :param df: dataframe to sample
    :return:
    """
    grouped_rules = df.groupby(by=['rule'])
    rules = {int(rule['tag'].split('_')[-1]): rule['value'] for rule in load_rules()}

    for label, group in grouped_rules:
        print(f'Matching Rule: {rules[label]}\n')

        for r in group.head().itertuples():
            print(r.text + '\n')


def augment_dataframe(df: pd.DataFrame, key: str, secondary_key: str = None) -> pd.DataFrame:
    """
    Augments dataframe with columns drawn from dictionary columns. Takes child keys for use as columns
    :param df: dataframe to augment
    :param key: key containing dictionary
    :param secondary_key: dictionary key for use as second column
    :return:
    """
    df[secondary_key or key] = df['data'].apply(
        lambda x: x.get(key, '') if secondary_key is None else x.get(key, {}).get(secondary_key, ''))
    return df


def get_place_annotation(x):
    """
    Gets all the place annotations for a tweet
    :param x: tweet data
    :return: all annotations
    """
    if x is None or x == '':
        return []

    return [annotation for annotation in x.get('annotations', []) if annotation.get('type', '') == 'Place']


def display_overlaid_histogram(df: pd.DataFrame, geo_df: pd.DataFrame, colors, stacks):
    """
    Displays rule histograms as overlay in Plotly
    :param df: raw data
    :param geo_df: geospatial data
    :param colors: colors for each histogram
    :param stacks: cached indexed series for fast computation
    :return:
    """
    fig = go.Figure()
    figures = (
        display_rules(df.copy(), color=colors[0], show=False, stk=stacks[0])[0],
        display_rules(geo_df.copy(), color=colors[1], show=False, stk=stacks[1])[0]
    )
    names = ['Raw', 'Geospatial']
    for i, f in enumerate(figures):
        f.data[0]['name'] = names[i]
    fig.add_trace(figures[0].data[0])
    fig.add_trace(figures[1].data[0])

    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.75)
    fig.update_yaxes(type='log')
    fig.update_layout(
        xaxis_title="Rule",
        yaxis_title="Count",
        showlegend=True
    )

    fig.write_html(f'{rules_dir}/rules.html')
    fig.write_json(f'{rules_dir}/rules.json')
    fig.show()


def get_user_location_tweets(df: pd.DataFrame, invert: bool = False) -> pd.DataFrame:
    """
    Filter for tweets containing user profile location
    :param df: tweets
    :param invert: whether to invert the filter, default is False
    :return: copy of filtered tweets
    """
    mask = df.user_location != {}
    return (df[mask] if not invert else df[~mask]).copy()


def get_tweet_place_tweets(df: pd.DataFrame, invert: bool = False) -> pd.DataFrame:
    """
    Filter for tweets containing unresolved tweet locations
    :param df: tweets
    :param invert: whether to invert the filter, default is False
    :return: copy of filtered tweets
    """
    mask = df.tweet_place != {}
    return (df[mask] if not invert else df[~mask]).copy()


def get_tweet_location_tweets(df: pd.DataFrame, invert: bool = False) -> pd.DataFrame:
    """
    Filter for tweets containing resolved tweet locations
    :param df: tweets
    :param invert: whether to invert the filter, default is False
    :return: copy of filtered tweets
    """
    mask = df.tweet_location != {}
    return (df[mask] if not invert else df[~mask]).copy()


def get_entity_locations_tweets(df: pd.DataFrame, invert: bool = False) -> pd.DataFrame:
    """
    Filter for tweets containing locations resolved through named entity recognition
    :param df: tweets
    :param invert: whether to invert the filter, default is False
    :return: copy of filtered tweets
    """
    mask = (~df.entity_locations.isnull()) & (df.entity_locations.map(len) != 0)
    return (df[mask] if not invert else df[~mask]).copy()


def get_tweet_geo(df: pd.DataFrame, invert: bool = False) -> pd.DataFrame:
    """
    Filter for tweets containing location data
    :param df: tweets
    :param invert: whether to invert the filter, default is False
    :return: copy of filtered tweets
    """
    mask = (df.tweet_place != {}) & (df.tweet_location != {})
    return (df[mask] if not invert else df[~mask]).copy()


def get_geo(df: pd.DataFrame, invert: bool = False) -> pd.DataFrame:
    """
    Filter for tweets containing geospatial features
    :param df: tweets
    :param invert: whether to invert the filter, default is False
    :return: copy of filtered tweets
    """
    mask = (df.user_location != {}) | (df.tweet_place != {}) | (df.tweet_location != {}) | (
            df.entity_locations.map(len) != 0)
    return (df[mask] if not invert else df[~mask]).copy()


def get_all_geo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter for tweets containing every source of geospatial feature
    :param df: tweets
    :return: copy of filtered tweets
    """
    return df[(df.user_location != {}) & (df.tweet_place != {}) & (df.tweet_location != {}) & (
            df.entity_locations.map(len) != 0)].copy()


def handle_user_coords(x):
    """
    Loads the latitude, longitude and bounding box from user location
    :param x: user location
    :return: coordinates
    """
    return x.get('lat', None), x.get('lon', None), x.get('boundingbox', None)


def get_lat_lon(x: dict) -> typing.Optional[tuple]:
    """
    Loads the latitude, longitude and bounding box from Twitter place data
    :param x: geospatial data
    :return: coordinates
    """
    ps = x.get('includes', {}).get('places', 0)
    if ps == 0:
        return None
    bboxes = [place['geo']['bbox'] for place in ps if place.get('geo', {}).get('bbox', None) is not None]
    return [((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, bbox) for bbox in bboxes][0]


def bbox_subset(x, y):
    """
    Adapted code for determining whether one bounding box subsumes the other
    :param x: box assumed to be subsumed
    :param y: box assumed to subsume
    :return: whether subsumption occurs
    """
    x1, y1, w1, h1 = x
    x2, y2, w2, h2 = y

    minx1, minx2 = (x1 - w1 / 2, x2 - w2 / 2)
    maxx1, maxx2 = (x1 + w1 / 2, x2 + w2 / 2)
    miny1, miny2 = (y1 - h1 / 2, y2 - h2 / 2)
    maxy1, maxy2 = (y1 + h1 / 2, y2 + h2 / 2)

    return not (minx2 < minx1 or maxx2 > maxx1 or miny2 < miny1 or maxy2 > maxy1)


def get_entity_coords(xs: list) -> list:
    """
    Loads the latitude, longitude and bounding box for each resolved entity
    :param xs: resolved location entities
    :return: coordinates for each entity
    """
    return [(x.get('lat', None), x.get('lon', None), x.get('boundingbox', None)) for x in xs if x is not None]


def bounding_box_subset(data):
    """
    Filters resolved entities to find those that do not subsume others, i.e. the most precise locations available
    :param data: resolved location entities
    :return: filtered entities
    """
    def recursive_bbox_subset(xs):
        """
        Recursively filters bounding boxes for subsumption
        :param xs: bounding boxes
        :return: formatted bounding boxes, filtered bounding boxes
        """
        def format_elem(y):
            """
            x, y, w, h from bounding box
            :param y: bounding box
            :return: x, y, w, h
            """
            return float(y[0]), float(y[1]), abs(float(y[2][0]) - float(y[2][1])), abs(float(y[2][2]) - float(y[2][3]))

        xs = {i: format_elem(x) for i, x in enumerate(xs)}
        remove = {i: False for i in range(len(xs))}
        for x in combinations(range(len(xs)), 2):
            if bbox_subset(xs[x[0]], xs[x[1]]):
                remove[x[1]] = True
            elif bbox_subset(xs[x[1]], xs[x[0]]):
                remove[x[0]] = True
        ys = {i: data[i] for i in range(len(xs)) if not remove[i]}
        return xs, ys

    dx, dy = recursive_bbox_subset(data)

    while len(dx.keys()) != len(dx.keys() & dy.keys()):
        dx, dy = recursive_bbox_subset(dy.values())
    return [data[i] for i in dy.keys()]


def get_location(x):
    """
    Selects location from the three sources. First preference is tweet coordinates, second is resolved entities and
    third is user profile location
    :param x: tweet with location data
    :return: resolved location
    """
    locations = []
    loc_type = None
    if x.tweet_coords is not None and None not in x.tweet_coords:
        locations.append(x.tweet_coords)
        loc_type = 'coordinates'
    if x.entity_coords:
        entity_coords = [y for y in x.entity_coords if None not in y]
        if entity_coords:
            locations.extend(entity_coords)
            loc_type = 'entity'
    if not locations:
        if x.user_coords is not None and None not in x.user_coords:
            return {'locations': [x.user_coords], 'type': 'user'}
        else:
            return None
    locations = bounding_box_subset(locations)
    return {'locations': locations, 'type': loc_type}


def save_for_annotation(df, filename: str = f'{output_dir}/geospatial_sampled.json', n: int = 5000):
    """
    Saves a sample of the geospatial data for annotation
    :param df: external data
    :param filename: filename to save sample to, default is output_dir/geospatial_sampled_test.json
    :param n: size of sample to take, default is 5,000
    :return:
    """
    geospatial = []

    for r in df.head(n).itertuples():
        geospatial.append({'data': {'id': r.Index, 'text': r.text}})

    write_json(geospatial, filename=filename)


def generate_word_cloud(df: pd.DataFrame, sp) -> WordCloud:
    """
    Creates a word cloud for the given text
    :param df: dataframe containing text to visualise
    :param sp: Spacy environment for tokenization and processing
    :return: word cloud
    """
    preprocessor = PreProcessing(df)
    pdf = preprocessor.strip_newlines().strip_links().strip_emojis().strip_mentions().strip_hashtags(
        ).convert_html_entities().df
    text = ""
    for doc in tqdm(sp.pipe(pdf.text.astype('unicode').values, batch_size=50, n_process=3)):
        if doc.has_annotation('DEP'):
            text += ' '.join(n.lower_ for n in doc if not n.is_punct)

    return WordCloud(width=3000, height=2000, random_state=1, background_color='white', collocations=False,
                     stopwords=STOPWORDS).generate(text)


def get_labelled(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters for labelled data
    :param df: data to filter
    :return: copy of filtered data
    """
    return df[~df['label'].isnull()].copy()


def format_geojson(df: pd.DataFrame, key: str):
    """
    Formats data for output as GeoJSON
    :param df: data to format
    :param key: key of data to format
    :return: formatted GeoJSON
    """
    return df.apply(lambda x: [x[key], key, x.name, '#D5C7BC'], axis=1).tolist()


def format_geojson_entities(df: pd.DataFrame):
    """
    Formats entities for output as GeoJSON
    :param df: data to format
    :return: formatted GeoJSON
    """
    data = []
    for row in df.itertuples():
        # noinspection PyUnresolvedReferences
        data = data + [[x, 'entity', row.Index, '#D5C7BC'] for x in row.entity_coords]
    return data


def format_geojson_all_types(df):
    """
    Formats all forms of location for output as GeoJSON
    :param df: data to format
    :return: formatted GeoJSON
    """
    data = []
    for i, x in tqdm(enumerate(df.itertuples()), total=len(df)):
        for j in range(3):
            if j == 0:
                ll = handle_user_coords(x.user_location)
                data.append([ll, 'user', i, '#E9FAE3'])
            elif j == 1:
                ll = get_lat_lon(x.tweet_place)
                data.append([ll, 'tweet', i, '#D5C7BC'])
            else:
                lls = get_entity_coords(x.entity_locations)

                for ll in lls:
                    data.append([ll, 'entity', i, '#AC92A6'])
    return data


def to_geojson(data, label: str, override: bool = False, dir_: str = geojson_dir):
    """
    Saves data to file as GeoJSON
    :param data: data to output as GeoJSON
    :param label: label for output file
    :param override: whether to overwrite existing GeoJSON, default is False
    :param dir_: directory to save to, default is geojson_dir
    :return:
    """
    if not exists(Path(f'{dir_}/{label}_config.json')) or override:
        generate_geojson_config(label, dir_=dir_)

    geojson = {
        "features": [
            {
                "geometry": {
                    "coordinates": [
                        float(x[0][1]),
                        float(x[0][0])
                    ],
                    "type": "Point"
                },
                "id": "0ef23d1cb491375bcd61636e96ccb91bc57e5f62e5a0ff80016fc467e32a245d",
                "properties": {
                    "marker-color": f"{x[3]}",
                    "marker-size": "medium",
                    "marker-symbol": "",
                    "changeset": "0",
                    "name": f"{x[1]}: {x[2]}",
                    "timestamp": datetime.datetime.now(),
                    "version": "1"
                },
                "type": "Feature"
            }
            for x in data],
        "type": "FeatureCollection"
    }

    write_json(geojson, f'{dir_}/{label}.json')


def generate_geojson_config(label, dir_, locations_url='http://localhost:8000/all_locations.json'):
    """
    Builds GeoJSON config file from template
    :param label: label for output file
    :param dir_: directory to save config file to
    :param locations_url: URL for associated locations
    :return:
    """
    config = {
        "layers": [
            {
                "type": "L.tileLayer",
                "visible": False,
                "wms": False,
                "url": "https://services.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{"
                       "x}.jpg"
            },
            {
                "type": "L.geoJSON",
                "visible": True,
                "wms": False,
                "url": f"{locations_url}",
                "style": {
                    "fill": {
                        "color": "#B29255"
                    },
                    "stroke": {
                        "color": "#715E3A",
                        "width": 4
                    }
                }
            }
        ],
        "center": [
            -157081.37129179656,
            6606026.977066623
        ],
        "resolution": 77,
        "zoom": 12
    }

    write_json(config, f'{dir_}/{label}_config.json')


def get_geo_df(x):
    """
    Loads data and filters for geospatial data
    :param x: filename of data to load
    :return: copy of filtered data
    """
    temp = pd.read_pickle(f'{shared_dir}/{x}.pickle')
    return temp[(temp['entity_locations'].str.len() > 2) | (temp['user_location'].str.len() > 2) | (
        ~temp['place_data'].isnull())].copy()
