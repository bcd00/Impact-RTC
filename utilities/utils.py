import json
import math
import torch
import numpy as np
import pandas as pd
import urllib.error
import urllib.parse
import seaborn as sns
import urllib.request
import plotly.express as px
import matplotlib.pyplot as plt

from treelib import Tree
from matplotlib import cm
from os.path import exists
from pymongo import MongoClient
from dotenv import dotenv_values
from collections import namedtuple, Counter
from IPython.core.display_functions import display

config = dotenv_values()
bearer_token = config['BEARER_TOKEN']

cache_dir = './cache'
input_dir = './input'
output_dir = './output'
shared_dir = './shared'
stats_dir = './output/stats'
figures_dir = './output/figures'
hashtags_dir = './input/hashtags'
annotated_dir = './input/annotated'
checkpoints_dir = './output/checkpoints'
labels_dir = './output/clustering_labels'
rules_dir = './output/figures/data_processing/rules'
visualisation_dir = './output/figures/visualisation'
arima_dir = './output/figures/impact_prediction/arima'
geojson_dir = './output/figures/data_processing/geojson'
final_model_dir = './output/figures/classifier/final_model'
classifier_tuning_dir = './output/figures/classifier/tuning'
nn_model_dir = './output/figures/impact_prediction/nn_model'
annotations_dir = './output/figures/data_processing/annotations'
word_clouds_dir = './output/figures/data_processing/word_clouds'
custom_tuning_dir = './output/figures/clustering/custom_clusters/tuning'
kmeans_tuning_dir = './output/figures/clustering/kmeans_clusters/tuning'
data_exploration_dir = './output/figures/impact_prediction/data_exploration'
linear_regression_dir = './output/figures/impact_prediction/linear_regression'


def bearer_oauth(r):
    """
    Method required by bearer token authentication
    :param r: request being made
    :return: authorized request
    """
    r.headers['Authorization'] = f'Bearer {bearer_token}'
    r.headers['User-Agent'] = config['TWITTER_USERNAME']
    return r


def load_db():
    """
    Loads the Mongo database containing the tweets
    :return: loaded database
    """
    mongo_username = urllib.parse.quote_plus(config['MONGO_USERNAME'])
    mongo_password = urllib.parse.quote_plus(config['MONGO_PASSWORD'])
    client = MongoClient(config['MONGO_URL'],
                         username=mongo_username,
                         password=mongo_password)
    return client[config['MONGO_DATABASE_NAME']]


def env_bool(key: str) -> bool:
    """
    Loads a boolean value from the environment config
    :param key: key for boolean value
    :return: the stored boolean value
    """
    return config[key] == 'true'


def match_rule(x, all_rules):
    """

    :param x:
    :param all_rules:
    :return:
    """
    return list(filter(lambda y: y['tag'] == x['matching_rules'][0]['tag'], all_rules))


def split_dataset(df: pd.DataFrame, split: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a dataset into two according to a percentage
    :param df: dataframe to split
    :param split: percentage split, default is 80%
    :return: a pair of split datasets
    """
    df = df.copy()
    rng = np.random.default_rng(seed=42)
    mask = rng.random(len(df)) <= split
    return df[mask], df[~mask]


def _recurse_keys(xs: dict):
    """
    Helper function for recursively getting all keys from a dictionary
    :param xs: dictionary to fetch keys for
    :return: list of (key, value) pairs
    """
    for key, value in xs.items():
        if type(value) is dict:
            yield from [(key, value)] + list(_recurse_keys(value))
        else:
            yield key, value


def get_all_keys(xs: dict):
    """
    Recursively gets all keys from a dictionary
    :param xs: dictionary to fetch keys for
    :return: all keys
    """
    return [key for key, _ in _recurse_keys(xs)]


def _display_dict_helper(xs, parent, tree, key_counter):
    """
    Recursive helper function for displaying dictionary keys as a tree
    :param xs: dictionary to display
    :param parent: parent identifier in tree
    :param tree: tree being built
    :param key_counter: counter for keys for unique indexing
    :return:
    """
    for key, value in xs.items():
        identifier = f'{key.lower()}_{key_counter[key][0]}'
        key_counter[key] = (key_counter[key][0] + 1, key_counter[key][1])
        tree.create_node(key, identifier, parent=parent)
        if type(value) is dict:
            _display_dict_helper(value, parent=identifier, tree=tree, key_counter=key_counter)


def display_dict(xs: dict):
    """
    Displays the keys of a nested dictionary as a tree
    :param xs: dictionary to display
    :return:
    """
    tree = Tree()
    keys = get_all_keys(xs)
    key_counter = {key: (0, keys.count(key)) for key in keys}
    tree.create_node('root', 'root', parent=None)
    for key, value in xs.items():
        identifier = f'{key.lower()}_{key_counter[key][0]}'
        key_counter[key] = (key_counter[key][0] + 1, key_counter[key][1])
        tree.create_node(key, identifier, parent='root')
        if type(value) is dict:
            _display_dict_helper(value, parent=identifier, tree=tree, key_counter=key_counter)
    tree.show()


def check_network():
    """
    Tests whether the network is available by trying Google with a 5-second timeout
    :return: whether the network is available
    """
    try:
        urllib.request.urlopen('https://www.google.com', timeout=5)
        return True
    except urllib.error.URLError:
        return False


def read_json(filename):
    """
    Reads a JSON file
    :param filename: filename to read from
    :return: loaded object, {} if filename does not exist
    """
    if not exists(filename):
        return {}

    with open(filename, encoding='utf-8') as f:
        return json.loads(f.read())


def read_jsonl(filename):
    """
    Reads a JSONL file
    :param filename: filename to read from
    :return: list with each element as a line in the JSONL file, {} if filename does not exist
    """
    if not exists(filename):
        return []

    with open(filename, encoding='utf-8') as f:
        xs = list(f)

    return [json.loads(x) for x in xs]


def write_json(obj, filename):
    """
    Writes a JSON object to file
    :param obj: object to save
    :param filename: filename for saving to
    :return:
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(obj, f, default=str)


def load_rules():
    """
    Loads the rules for Twitter filtering from the rules file
    :return: all rules
    """
    return read_json(f'{input_dir}/twitter_rules.json')['rules']


def downsample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Code for downsampling the negative class
    :param df: dataframe to downsample
    :return: downsampled data
    """
    df = df[df.label == 1]
    npos = len(df)

    return pd.concat([df, df[df.label == 0][:npos]])


def plot_confusion_matrix(confusion_matrix, filename=None, format_labels=True, group_names=None):
    """
    Formatting and plotting code for confusion matrices showing name, count and percentage for each element
    :param confusion_matrix: confusion matrix to plot
    :param filename: optional filename for saving plot to, default is None
    :param format_labels: whether to include percentages in formatted labels, default is True
    :param group_names: optionally provide explicit group names, default is [TN, FP, FN, TP]
    :return:
    """
    if group_names is None:
        group_names = ['TN', 'FP', 'FN', 'TP']

    if format_labels:
        group_counts = ["{0:0.0f}".format(value) for value in confusion_matrix.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in confusion_matrix.flatten() / np.sum(confusion_matrix)]

        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    else:
        group_counts = ['{0:0.2f}'.format(value) for value in confusion_matrix.flatten()]
        labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names, group_counts)]

    labels = np.asarray(labels).reshape(2, 2)

    heatmap = sns.heatmap(confusion_matrix, annot=labels, fmt='', cmap='Blues')

    figure = heatmap.get_figure()
    if filename is not None:
        figure.savefig(filename, dpi=400)
    plt.show(figure)
    plt.close(figure)


def plot_graph(xs, ys, xlabel, ylabel, title, filename=None, show=True):
    """
    Helper function for graph plotting using Plotly
    :param xs: x values to plot
    :param ys: y values to plot
    :param xlabel: label for x-axis
    :param ylabel: label for y-axis
    :param title: title for graph
    :param filename: optional filename for saving figure to, saves as JSON and HTML, default is None
    :param show: whether to show or to return the figure, default is True
    :return: optional figure if show is False
    """
    df = pd.DataFrame({xlabel: xs, ylabel: ys})
    fig = px.line(df, x=xlabel, y=ylabel, title=title, template='plotly')
    if filename is not None:
        fig.write_html(f'{figures_dir}/{filename}.html')
        fig.write_json(f'{figures_dir}/{filename}.json')
    if show:
        fig.show()
    else:
        return fig


def plot_lrs(lrs, filename=None, show=True, override_filename=False):
    """
    Plots a learning rate graph using Plotly
    :param lrs: learning rates to plot
    :param filename: optional filename to save figure to, default is None
    :param show: whether to show or to return the figure, default is True
    :param override_filename: whether to override the default filename formatting, default is False
    :return: optional figure if show is False
    """
    filename = f'learning_rates{"" if filename is None else f"_{filename}"}' if not override_filename else filename
    fig = plot_graph(
        xs=list(range(len(lrs))),
        ys=lrs,
        xlabel='Iteration',
        ylabel='Learning Rate',
        title='Learning Rate Graph',
        filename=filename,
        show=show
    )
    if not show:
        return fig


def plot_losses(losses, filename=None, show=True, override_filename=False):
    """
    Plots a loss graph using Plotly
    :param losses: losses to plot
    :param filename: optional filename to save figure to, default is None
    :param show: whether to show or to return the figure, default is True
    :param override_filename: whether to override the default filename formatting, default is False
    :return: optional figure if show is False
    """
    filename = f'losses{"" if filename is None else f"_{filename}"}' if not override_filename else filename
    fig = plot_graph(
        xs=list(range(len(losses))),
        ys=losses,
        xlabel='Iteration',
        ylabel='Loss',
        title='Loss Graph',
        filename=filename,
        show=show
    )
    if not show:
        return fig


def print_report(report):
    """
    Prints the key figures from the sklearn report
    :param report: report to print
    :return:
    """
    print(f'accuracy: {report["accuracy"]}')
    print(f'label 0 f1-score: {report["negative"]["f1-score"]}')
    print(f'label 1 f1-score: {report["positive"]["f1-score"]}')
    print(f'Cohen\'s Kappa: {report["kappa"]}')
    if report['negative']['f1-score'] == 0 or report['positive']['f1-score'] == 0:
        print(f"report: {report}")


def save_predictions(outputs, path):
    """
    Saves predictions to file
    :param outputs: predictions
    :param path: path of file to save to
    :return:
    """
    with open(path, 'w') as f:
        f.write(','.join([str(x) for x in outputs]) + '\n')


def jaccard_similarity(tokens: list) -> float:
    """
    Calculates the Jaccard similarity between two sets of tokens
    :param tokens: sets of tokens to find similarity of
    :return: Jaccard similarity
    """
    tokens = [set(x) for x in tokens]
    return len(tokens[0].intersection(tokens[1])) / len(tokens[0].union(tokens[1]))


def calculate_dataset_similarity(tokenizer, one: pd.DataFrame, two: pd.DataFrame) -> float:
    """
    Calculates the Jaccard similarity between two datasets
    :param tokenizer: tokenizer for tokenizing the datasets
    :param one: first dataset
    :param two: second dataset
    :return: Jaccard similarity
    """
    texts = [one['text'].str.cat(sep=' '), two['text'].str.cat(sep=' ')]
    tokens = [tokenizer(text, return_tensors='pt', padding=False, truncation=True, max_length=2_500_000)[
                  'input_ids'].detach().cpu().numpy().tolist()[0] for text in texts]
    return jaccard_similarity(tokens)


def split_by_length(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a dataframe into two by the median length of text
    :param df: dataframe to split
    :return: a pair of dataframes after splitting
    """
    median = np.median([len(text) for text in df['text'].tolist()])
    mask = (df['text'].str.len() < median)
    return df[mask], df[~mask]


def get_positive(df: pd.DataFrame, label: str = 'label') -> pd.DataFrame:
    """
    Filters for the positive class in a labelled dataframe
    :param df: dataframe to filter
    :param label: label to filter on
    :return: copy of filtered dataframe
    """
    return df[df[label] == 1].copy()


def get_negative(df: pd.DataFrame, label: str = 'label') -> pd.DataFrame:
    """
    Filters for the negative class in a labelled dataframe
    :param df: dataframe to filter
    :param label: label to filter on
    :return: copy of filtered dataframe
    """
    return df[df[label] == 0].copy()


def display_image(img, filename: str = None):
    """
    Displays a given image and saves the figure
    :param img: image to display
    :param filename: name of file to save image to
    :return:
    """
    fig = plt.figure(figsize=(40, 30))

    plt.imshow(img)
    plt.axis("off")

    if filename is not None:
        fig.savefig(filename, dpi=fig.dpi)
    plt.show()


def get_tp(df: pd.DataFrame, y_label: str = 'label', yhat_label: str = 'prediction') -> pd.DataFrame:
    """
    Filters for true positives in a dataset.
    :param df: source of data to filter
    :param y_label: label of true value, default is 'label'
    :param yhat_label: label of predicted value, default is 'prediction'
    :return: copy of the dataset containing only true positives
    """
    return df[(df[y_label] == 1) & (df[yhat_label] == 1)].copy()


def get_tn(df: pd.DataFrame, y_label: str = 'label', yhat_label: str = 'prediction') -> pd.DataFrame:
    """
    Filters for true negatives in a dataset.
    :param df: source of data to filter
    :param y_label: label of true value, default is 'label'
    :param yhat_label: label of predicted value, default is 'prediction'
    :return: copy of the dataset containing only true negatives
    """
    return df[(df[y_label] == 0) & (df[yhat_label] == 0)].copy()


def get_fp(df: pd.DataFrame, y_label: str = 'label', yhat_label: str = 'prediction') -> pd.DataFrame:
    """
    Filters for false positives in a dataset.
    :param df: source of data to filter
    :param y_label: label of true value, default is 'label'
    :param yhat_label: label of predicted value, default is 'prediction'
    :return: copy of the dataset containing only false positives
    """
    return df[(df[y_label] == 0) & (df[yhat_label] == 1)].copy()


def get_fn(df: pd.DataFrame, y_label: str = 'label', yhat_label: str = 'prediction') -> pd.DataFrame:
    """
    Filters for false negatives in a dataset.
    :param df: source of data to filter
    :param y_label: label of true value, default is 'label'
    :param yhat_label: label of predicted value, default is 'prediction'
    :return: copy of the dataset containing only false negatives
    """
    return df[(df[y_label] == 1) & (df[yhat_label] == 0)].copy()


def plot_bihistogram(
        dfs: list[pd.DataFrame],
        key: str,
        labels: list[str],
        filename: str = None,
        xlabel: str = 'count',
        ylabel: str = None
):
    """
    Plots a bihistogram drawn from two dataframes
    :param dfs: dataframes to use as source
    :param key: key of variable to find distribution of
    :param labels: labels for each part of the histogram
    :param filename: optional filename to save figure to
    :param xlabel: optional label of x-axis, default is 'count'
    :param ylabel: optional label of y-axis, default is to use the key
    :return:
    """

    def format_tick(x):
        if x > 0:
            return '%.1f' % math.log10(abs(x))
        else:
            return '-%.1f' % math.log10(abs(x))

    if ylabel is None:
        ylabel = key

    fig = plt.figure()
    normed = False
    colormap = cm.get_cmap('tab10').colors
    hn = plt.hist(dfs[0][key], color=colormap[0], orientation='horizontal', density=normed, stacked=normed,
                  rwidth=0.8, label=labels[0], log=True)
    hs = plt.hist(dfs[1][key], color=colormap[4], bins=hn[1], orientation='horizontal', density=normed,
                  stacked=normed,
                  rwidth=0.8, label=labels[1], log=True)

    plt.xscale('symlog')

    for p in hs[2]:
        p.set_width(- p.get_width())

    xmin = min([min(w.get_width() for w in hs[2]), min([w.get_width() for w in hn[2]])])
    xmin = np.floor(xmin)
    xmax = np.ceil(max([max(w.get_width() for w in hs[2]), max([w.get_width() for w in hn[2]])]))
    delta = 10 * (xmax - xmin)
    plt.xlim([xmin - delta, xmax + delta])
    xt = plt.xticks()
    xt = xt[0]

    s = [i for i in xt if i != 0]
    plt.xticks(s)
    s_new = [format_tick(i) for i in xt if i != 0]
    plt.xticks(s, s_new)

    plt.legend(loc='upper left')
    plt.axvline(0.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if filename is not None:
        plt.savefig(filename, dpi=fig.dpi)

    plt.show()


def get_distance(points) -> float:
    """
    Calculates the distance in kilometres between two points
    :param points: points to calculate distance between
    :return: calculated distance
    """

    def deg2rad(deg: float) -> float:
        """
        Converts degrees to radians
        :param deg: value of degrees to convert
        :return: radian conversion
        """
        return deg * (math.pi / 180)

    r = 6371
    d_lat = deg2rad(points[1].lat - points[0].lat)
    d_lon = deg2rad(points[1].lon - points[0].lon)
    a = math.sin(d_lat / 2) * math.sin(d_lat / 2) + math.cos(deg2rad(points[0].lat)) * math.cos(
        deg2rad(points[1].lat)) * math.sin(d_lon / 2) * math.sin(d_lon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def ck_interpretation(kappa_: float) -> str:
    """
    Provides a human-readable interpretation of the Cohen's Kappa value.
    :param kappa_: value of the kappa
    :return: interpretation
    """
    if kappa_ < 0.1:
        return 'No agreement'
    elif 0.1 < kappa_ < 0.2:
        return 'Slight agreement'
    elif 0.21 < kappa_ < 0.4:
        return 'Fair agreement'
    elif 0.41 < kappa_ < 0.6:
        return 'Moderate agreement'
    elif 0.61 < kappa_ < 0.8:
        return 'Substantial agreement'
    elif 0.81 < 0.99:
        return 'Near perfect agreement'
    else:
        return 'Perfect agreement'


def kappa(tp: int, tn: int, fp: int, fn: int) -> tuple[float, str]:
    """
    Calculates Cohen's Kappa.
    :param tp: number of true positives
    :param tn: number of true negatives
    :param fp: number of false positives
    :param fn: number of false negatives
    :return: kappa value
    """
    numerator = 2 * (tp * tn - fn * fp)
    denominator = (tp + fp) * (fp + tn) + (tp + fn) * (fn + tn)
    kappa_ = numerator / denominator
    return kappa_, ck_interpretation(kappa_)


def visualise_cf_samples(df: pd.DataFrame):
    """
    Visualise samples of the four elements of a confusion matrix
    :param df: dataframe with samples to visualise
    :return:
    """
    print('TP')
    display(get_tp(df).head())
    print('TN')
    display(get_tn(df).head())
    print('FP')
    display(get_fp(df).head())
    print('FN')
    display(get_fn(df).head())


def select_annotation(annotations: list[int]) -> int:
    """
    Produces the consensus label from multiple annotations
    :param annotations: all annotations
    :return: most common annotation
    """
    return Counter(annotations).most_common(1)[0][0]


def load_raw_annotations(filename: str) -> list:
    """
    Loads annotations from file. Annotations contain the tweet's ID, text and label.
    :param filename: name of the file containing the annotations
    :return: raw list of annotations
    """
    annotations = [(x['data']['id'], x['data']['text'], x['annotations'][0]['result'][0]['value']['choices'][0]) for x
                   in
                   read_json(filename)]
    annotations = {x: (y, 0 if z == 'Negative' else 1) for x, y, z in annotations}
    annotations = [(k, v[0], v[1]) for k, v in annotations.items()]

    print(f'Size of annotated dataset: {len(annotations)}')
    print(f'Example annotation: {annotations[0]}')

    unzipped_annotations = list(zip(*annotations))
    return [list(annotation) for annotation in unzipped_annotations]


def load_annotations(filename: str) -> pd.DataFrame:
    """
    Loads annotations from file. Annotations contain the tweet's ID, text and label.
    :param filename: name of the file containing the annotations
    :return: processed annotations
    """
    unzipped_annotations = load_raw_annotations(filename)

    annotations_df = pd.DataFrame(
        {'id': unzipped_annotations[0], 'text': unzipped_annotations[1], 'label': unzipped_annotations[2]})
    annotations_df.set_index('id', inplace=True)
    annotations_df['label'].astype(int)

    return annotations_df


def get_cuda_availability():
    """
    Checks CUDA availability and returns the available device name
    :return: 'cuda' if CUDA is available, else 'cpu'
    """
    cuda_available = torch.cuda.is_available()
    print(f'Cuda available: {cuda_available}')

    if not torch.cuda.is_available():
        print('WARNING: You may want to change the runtime to GPU for faster training!')
        return 'cpu'
    else:
        return 'cuda'


def get_max_key(keys, values):
    """
    Filters keys for maximum test f1-score on positive class
    :param keys: hyperparameters
    :param values: model results for those hyperparameters
    :return: key for maximum results and hyperparameters
    """
    xs = [(k, values[k]['eval_report']['positive']['f1-score']) for k in keys.keys()]
    print(xs)
    return max(xs, key=lambda x: x[1])[0]


# noinspection PyTypeChecker
Point = namedtuple('Point', 'lat lon')
