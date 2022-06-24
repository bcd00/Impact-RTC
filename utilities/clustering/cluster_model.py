import math
import pytz
import datetime
import numpy as np
import pandas as pd
import itertools as it
import plotly.express as px

from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from datetime import datetime as dt
from timezonefinder import TimezoneFinder
from IPython.core.display_functions import display
from utilities.utils import get_distance, Point, write_json, labels_dir, output_dir, read_json, \
    custom_tuning_dir, kmeans_tuning_dir


# noinspection GrazieInspection
class ClusterModel:
    def __init__(self, df: pd.DataFrame):
        """
        Basic model setup.
        :param df: classified data to cluster
        """
        self.df = df
        self.cluster_data = []
        self.cluster_indices = []
        self.distances = []
        self.formatted_data = {}
        self.second_offset = 946_684_800
        self.min_date = None
        self.max_date = None
        self.min_time = df.time.min()
        self.time_rng = df.time.max() - self.min_time
        self.lat_f = lambda x: float(x) / 180 + 0.5
        self.lon_f = lambda x: float(x) / 360 + 0.5
        self.time_f = lambda x: (int(x) - self.min_time) / self.time_rng
        self.labelled_clusters = {}

        self.generate_data()

    def generate_data(self):
        """
        Generates the raw data that's used by clustering algorithms in the format (lat, lon, time, bounding box).
        Filters any location where the distance between the edges of its bounding box is greater than 1 to avoid
        clustering on data that can only be resolved at the county or country level, which lacks specificity.
        :return:
        """
        # noinspection PyUnresolvedReferences
        data = [[self.lat_f(row.location[0]), self.lon_f(row.location[1]), self.time_f(row.time),
                 [float(x) for x in row.location[2]]] for row in self.df.itertuples()]
        data = list(zip(*[(i, v[:3]) for i, v in enumerate(data) if
                          abs(v[3][1] - v[3][0]) <= 1. or abs(v[3][3] - v[3][2]) <= 1.]))
        self.df = self.df.iloc[np.array(data[0])].copy()
        self.cluster_data = np.array(data[1])

    def print_sample_data(self, n=5):
        """
        Displays a sample of the generated data used by the clustering algorithms.
        :param n: size of sample to display
        :return:
        """
        sample = pd.DataFrame(self.cluster_data[:n], columns=['Latitude', 'Longitude', 'Time'])
        display(sample.head())

    @staticmethod
    def _from_file_custom(time_limit, space_limit):
        """
        Loads labels for custom clustering from file.
        :param time_limit: hyperparameter for loading from file. Maximum time between tweets in cluster in hours.
        :param space_limit: hyperparameter for loading from file. Maximum distance between tweets in cluster
        in kilometres.
        :return: loaded labels as a Numpy array
        """
        with open(f'{labels_dir}/labels_custom_{time_limit}_{space_limit}.txt') as f:
            return np.loadtxt(f)

    @staticmethod
    def _from_file_kmeans(n_clusters):
        """
        Loads labels for kmeans clustering from file.
        :param n_clusters: hyperparameter for loading from file. Number of clusters expected.
        :return: loaded labels as a Numpy array
        """
        with open(f'{labels_dir}/labels_kmeans_{n_clusters}.txt') as f:
            return np.loadtxt(f)

    def from_file(self, cluster_type, *args):
        """
        Public function for loading labels from file.
        :param cluster_type: custom or kmeans
        :param args: custom: time_limit, space_limit --- kmeans: n_clusters
        :return: loaded labels as a Numpy array
        """
        match cluster_type:
            case 'custom':
                return self._from_file_custom(*args)
            case 'kmeans':
                return self._from_file_kmeans(*args)

    def _cluster_custom(self, space_limits, time_limits):
        """
        Clusters using custom clustering algorithm for hyperparameter tuning.
        :param time_limits: all possible time limits for custom clustering
        :param space_limits: all possible spatial limits for custom clustering
        :return:
        """
        self.distances = []
        params = list(it.product(space_limits, time_limits))
        for i, (space_limit, time_limit) in tqdm(enumerate(params), total=len(params)):
            custom = CustomCluster(self.time_rng, time_limit, space_limit)
            custom.fit(self.cluster_data)
            np.savetxt(f'{labels_dir}/labels_custom_{time_limit}_{space_limit}.txt', custom.labels_)
            self.distances.append({
                'wcss': custom.inertia_,
                'silhouette': custom.silhouette_,
                'dunn': custom.dunn_,
                'space_limit': space_limit,
                'time_limit': time_limit
            })
            del custom

    def _kmeans_clustering(
            self,
            start_n=5_000,
            end_n=16_000,
            step_n=1_000,
            init='k-means++',
            max_iter=1000,
            n_init=10,
            random_state=0
    ):
        """
        Clusters using KMeans for hyperparameter tuning.
        :param start_n: start of initial cluster size range
        :param end_n: end of initial cluster size range
        :param step_n: step size for cluster size range
        :param init: algorithm for initialisation, i.e. k-means++
        :param max_iter: maximum number of iterations
        :param n_init: number of centroid initialisations to complete
        :param random_state: sets random state for replication
        :return:
        """
        self.distances = []
        for i in tqdm(range(start_n, end_n + step_n, step_n)):
            kmeans = KMeansCluster(n_clusters=i, init=init, max_iter=max_iter, n_init=n_init,
                                   random_state=random_state)
            kmeans.fit(self.cluster_data)
            np.savetxt(f'{labels_dir}/labels_kmeans_{i}.txt', kmeans.labels_)
            self.distances.append({
                'wcss': kmeans.inertia_,
                'silhouette': kmeans.silhouette_,
                'dunn': kmeans.dunn_,
                'n_clusters': i
            })
            del kmeans

    def cluster(self, cluster_type, *args, **kwargs):
        """
        Public function for testing hyperparameters.
        :param cluster_type: clustering algorithm
        :param args:
            custom:
                time_limits: all possible time limits for custom clustering
                space_limits: all possible spatial limits for custom clustering
        :param kwargs:
            kmeans:
                start_n: start of initial cluster size range
                end_n: end of initial cluster size range
                step_n: step size for cluster size range
                init: algorithm for initialisation, i.e. k-means++
                max_iter: maximum number of iterations
                n_init: number of centroid initialisations to complete
                random_state: sets random state for replication
        :return:
        """
        match cluster_type:
            case 'custom':
                self._cluster_custom(*args, **kwargs)
            case 'kmeans':
                self._kmeans_clustering(*args, **kwargs)
            case _:
                raise ValueError(f'Unknown cluster type: {cluster_type}')

    @staticmethod
    def _plot_helper(df, x, y, cluster_type):
        """
        Generic helper function for plotting distance graphs.
        :param df: distance data
        :param x: label for x
        :param y: label for y
        :param cluster_type: clustering algorithm - used for saving graphs
        :return:
        """
        fig = px.line(df, x=x[0], y=y, template='plotly')

        if cluster_type == 'custom':
            fig.write_html(f'{custom_tuning_dir}/{x[1]}{f"_{y}" if y is not None else ""}_graph.html')
            fig.write_json(f'{custom_tuning_dir}/{x[1]}_{y}_graph.json')
        else:
            fig.write_html(f'{kmeans_tuning_dir}/{x[1]}{f"_{y}" if y is not None else ""}_graph.html')
            fig.write_json(f'{kmeans_tuning_dir}/{x[1]}_{y}_graph.json')

        display(fig)

    def _plot_custom(self):
        """
        Helper function for plotting custom clustering distance graphs.
        :return:
        """
        df = pd.DataFrame(self.distances)
        for y in ['wcss', 'silhouette', 'dunn']:
            self._plot_helper(df.groupby('space_limit').mean().reset_index(), ('space_limit', 'sl'), y, 'custom')
            self._plot_helper(df.groupby('time_limit').mean().reset_index(), ('time_limit', 'tl'), y, 'custom')

    def _plot_kmeans(self):
        """
        Helper function for plotting KMeans distance graphs.
        :return:
        """
        df = pd.DataFrame(self.distances)
        for y in ['wcss', 'silhouette', 'dunn']:
            self._plot_helper(df, ('n_clusters', 'n'), y, 'kmeans')

    def plot(self, cluster_type):
        """
        Plots the computed distance metrics.
        :param cluster_type: clustering algorithm: [custom, kmeans]
        :return:
        """
        match cluster_type:
            case 'custom':
                self._plot_custom()
            case 'kmeans':
                self._plot_kmeans()
            case _:
                raise ValueError(f'Unknown cluster type: {cluster_type}')

    def _generate_labels_custom(self, time_limit, space_limit):
        """
        Generates labels using custom clustering algorithm.
        :param time_limit: max time difference within a cluster
        :param space_limit: max spatial difference within a cluster
        :return: fitted labels
        """
        custom = CustomCluster(self.time_rng, time_limit, space_limit)
        custom.fit(self.cluster_data)
        self.distances.append({
            'wcss': custom.inertia_,
            'silhouette': custom.silhouette_,
            'dunn': custom.dunn_,
            'space_limit': space_limit,
            'time_limit': time_limit
        })
        return custom.fit_predict(self.cluster_data).tolist()

    def _generate_labels_kmeans(self, n_clusters, init='k-means++', max_iter=1000, n_init=10, random_state=0):
        """
        Generates labels using KMeans clustering.
        :param n_clusters: number of clusters to create
        :param init: algorithm for initialisation, i.e. k-means++
        :param max_iter: maximum number of iterations
        :param n_init: number of centroid initialisations to complete
        :param random_state: sets random state for replication
        :return: fitted labels
        """
        kmeans = KMeansCluster(n_clusters=n_clusters, init=init, max_iter=max_iter, n_init=n_init,
                               random_state=random_state)
        kmeans.fit(self.cluster_data)
        self.distances.append({
            'wcss': kmeans.inertia_,
            'silhouette': kmeans.silhouette_,
            'dunn': kmeans.dunn_,
            'n_clusters': n_clusters
        })
        return kmeans.fit_predict(self.cluster_data).tolist()

    def fit(self, cluster_type, *args, **kwargs):
        """
        Fits generated data into clusters.
        :param cluster_type: type of clustering
        :param args:
            custom:
                time_limit: max time difference within a cluster
                space_limit: max spatial difference within a cluster
            kmeans:
                n_clusters: number of clusters to create
        :param kwargs:
            kmeans:
                init: algorithm for initialisation, i.e. k-means++
                max_iter: maximum number of iterations
                n_init: number of centroid initialisations to complete
                random_state: sets random state for replication
        :return:
        """
        match cluster_type:
            case 'custom':
                labels = self._generate_labels_custom(*args, **kwargs)
            case 'kmeans':
                labels = self._generate_labels_kmeans(*args, **kwargs)
            case _:
                raise ValueError(f'Unknown cluster type: {cluster_type}')

        self.labelled_clusters = {}

        self.df['cluster_id'] = labels
        self.df['cluster_id'] = self.df['cluster_id'].astype(int)

        for i, cluster in self.df.groupby('cluster_id'):
            cluster.drop('cluster_id', axis=1, inplace=True)
            self.labelled_clusters[i] = cluster

        self.df.drop('cluster_id', axis=1, inplace=True)

    def save(self, filename):
        """
        Saves clusters to HTML file.
        :param filename: name of file to save cluster to
        :return:
        """
        html = [v.to_html() for _, v in self.labelled_clusters.items()]
        raw = "\n\n".join(html)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(raw)

    def display(self, m=5, n=5):
        """
        Displays labelled clusters.
        :param m: number of clusters to display
        :param n: how many cluster records to display
        :return:
        """
        i = 0

        for group in self.labelled_clusters.values():
            if i >= m:
                break
            display(group.head(n=n))
            i += 1

    def format_data(self, timezones=None, disable_tqdm=False):
        """
        Formats data for use in density prediction.
        :param timezones: cache of timezones
        :param disable_tqdm: whether to disable tqdm
        :return:
        """
        if timezones is None:
            timezones = {}

        for label, group in tqdm(self.labelled_clusters.items(), total=len(self.labelled_clusters),
                                 disable=disable_tqdm):
            x = 0.
            y = 0.
            time = 500_000_000_000_000

            for g in group.itertuples():
                x += float(g.location[0])
                y += float(g.location[1])
                if g.time < time:
                    time = int(g.time)

            self.formatted_data[label] = {
                'x': x / len(group),
                'y': y / len(group),
                'time': dt.utcfromtimestamp(time + self.second_offset),
                'size': len(group)
            }

        self.min_date = min([x['time'] for x in self.formatted_data.values()])
        self.max_date = max([x['time'] for x in self.formatted_data.values()])
        temp = read_json(f'{output_dir}/tzs.json')
        for i, k in tqdm(enumerate(self.formatted_data.keys()), total=len(self.formatted_data), disable=disable_tqdm):
            latitude, longitude = self.formatted_data[k]['x'], self.formatted_data[k]['y']
            idx = f'({latitude}, {longitude})'

            if temp.get(idx, None) is None:
                tf = TimezoneFinder()
                tzs = tf.timezone_at(lng=longitude, lat=latitude)
                temp[idx] = tzs
            else:
                tzs = temp[idx]

            if timezones.get(str(tzs), None) is None:
                tz = pytz.timezone(tzs)
                timezones[str(tzs)] = tz
                # print('Missed')
            else:
                tz = timezones[str(tzs)]

            local_time = self.formatted_data[k]['time'].replace(tzinfo=datetime.timezone.utc).astimezone(tz=tz)

            self.formatted_data[k]['local_time'] = local_time
            self.formatted_data[k]['timezone'] = tzs
            self.formatted_data[k]['date'] = (self.formatted_data[k]['time'] - self.min_date).days
            self.formatted_data[k]['time_of_day'] = self.formatted_data[k]['local_time'].hour
            self.formatted_data[k]['weekday'] = self.formatted_data[k]['local_time'].weekday()
        write_json(temp, filename=f'{output_dir}/tzs.json')
        return timezones

    def _sample_geojson(self, n=10):
        """
        Formats a sample of clusters into GeoJSON.
        :param n: size of sample
        :return: GeoJSON
        """
        layers = []
        keys = list(self.labelled_clusters.keys())
        indices = np.random.choice(len(self.labelled_clusters), size=n, replace=False)
        for idx in indices:
            key = keys[idx]
            value = self.labelled_clusters[key]
            features = [{
                'geometry': {
                    'coordinates': [
                        float(row.location[1]),
                        float(row.location[0])
                    ],
                    'type': 'Point'
                },
                'id': f'{key}',
                'properties': {
                    'id': f'{key}'
                },
                'type': 'Feature'
            } for row in value.itertuples()]
            layers.append({
                'features': features,
                'type': 'FeatureCollection'
            })
        return layers

    def _format_time_geojson(self):
        """
        Formats clusters to GeoJSON with a layer per day.
        :return: GeoJSON
        """
        layers = []
        for i in range((self.max_date - self.min_date).days):
            current_data = [(k, v) for k, v in self.formatted_data.items() if v['date'] == i]
            features = [{
                'geometry': {
                    'coordinates': [
                        v['y'],
                        v['x']
                    ],
                    'type': 'Point'
                },
                'id': f'{k}',
                'properties': {
                    'id': f'{k}'
                },
                'type': 'Feature'
            } for k, v in current_data]

            layers.append({
                'features': features,
                'type': 'FeatureCollection'
            })
        return layers

    def data_to_visualise(self, filter_type, **kwargs):
        """
        Creates GeoJSON layers for visualising data.
        :param filter_type: type of filter to apply when formatting. `time` formats with a layer for each day.
            `sample` formats a sample of clusters.
        :param kwargs:
            n: size of sample to save when filter_type == sample
        :return: formatted GeoJSON
        """
        match filter_type:
            case 'time':
                return self._format_time_geojson()
            case 'sample':
                return self._sample_geojson(**kwargs)
            case _:
                raise ValueError(f'Unknown filter type: {filter_type}')

    def to_file(self, labelled_filename, formatted_filename):
        """
        Saves labelled and formatted data to file
        :param labelled_filename: name of file to save labelled data to
        :param formatted_filename: name of file to save formatted data to
        :return:
        """
        xs = {k: v.to_dict() for k, v in self.labelled_clusters.items()}
        write_json(xs, labelled_filename)
        write_json(self.formatted_data, formatted_filename)


def rand_index(cluster_ids):
    ab = 0
    counter = 0
    id_map_kmeans = {idx: i for i, ids_ in enumerate(cluster_ids[0]) for idx in ids_}
    id_map_custom = {idx: i for i, ids_ in enumerate(cluster_ids[1]) for idx in ids_}
    ids = list(id_map_kmeans.keys())

    for i_, id_one in tqdm(enumerate(ids[:-1]), total=len(ids) - 1):
        for id_two in ids[i_ + 1:]:
            paired_kmeans = id_map_kmeans[id_one] == id_map_kmeans[id_two]
            paired_custom = id_map_custom[id_one] == id_map_custom[id_two]

            if paired_kmeans == paired_custom:
                ab += 1
            counter += 1

    print(f'RI: {ab / counter}')


def adjusted_rand_index(clusters):
    n_total = len([idx for ids_ in clusters[0] for idx in ids_])
    m = np.zeros((len(clusters[0]), len(clusters[1])))

    for i_, x in tqdm(enumerate(clusters[0]), total=len(clusters[0])):
        for j_, y in enumerate(clusters[1]):
            m[i_, j_] = len(set(x).intersection(set(y)))

    n = 0
    df_ = pd.DataFrame(m)
    display(df_.head())
    as_x = []
    bs_x = []

    for row in df_.itertuples():
        s = 0
        for x in row[1:]:
            s += int(x)
            n += math.comb(int(x), 2)
        as_x.append(math.comb(s, 2))

    for col in list(df_.columns):
        bs_x.append(math.comb(int(sum(df_[col].tolist())), 2))

    print(as_x)
    print(bs_x)

    a_ = sum(as_x)
    b = sum(bs_x)

    index_value = n
    expected_index_value = (a_ * b) / math.comb(n_total, 2)
    maximum_index_value = (a_ + b) / 2

    print(f'ARI: {(index_value - expected_index_value) / (maximum_index_value - expected_index_value)}')


# noinspection PyPep8Naming,GrazieInspection
class CustomCluster:
    """
    Model for custom clustering. Enhanced functionality using a custom clustering algorithm and providing Dunn's Index
    and the Silhouette Coefficient.
    """

    def __init__(self, time_rng, time_limit, space_limit):
        """
        Basic model setup.
        :param time_rng: max time - min time
        :param time_limit: limit on time difference within cluster
        :param space_limit: limit on spacial difference within cluster
        """
        self.time_rng = time_rng
        self.inv_lat_f = lambda x: (x - 0.5) * 180
        self.inv_lon_f = lambda x: (x - 0.5) * 360
        self.time_limit = time_limit
        self.space_limit = space_limit
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.silhouette_ = None
        self.dunn_ = None

    def fit(self, X):
        """
        Fits the data into clusters using custom algorithm relying on limiting space and time distance.
        :param X: data to fit
        :return:
        """

        def _cluster_condition(a, b):
            """
            Whether the points are within the space limit.
            :param a: first pair of coordinates
            :param b: second pair of coordinates
            :return: whether the condition is met
            """
            point_x = Point(self.inv_lat_f(a[0]), self.inv_lon_f(a[1]))
            point_y = Point(self.inv_lat_f(b[0]), self.inv_lon_f(b[1]))
            return abs(get_distance((point_x, point_y))) < self.space_limit

        clusters = []
        mask = np.ma.make_mask(np.ones(len(X)), shrink=False)
        self.labels_ = np.zeros(len(X))
        for i, x in tqdm(enumerate(X), total=len(X), leave=True):
            if not mask[i]:
                continue

            cluster = []
            temp_data = list(it.takewhile(lambda z: abs(z[2] - x[2]) < (self.time_limit * 3600) / self.time_rng,
                                          X[i + 1:][mask[i + 1:]]))

            m_idx = np.where(mask[i + 1:])[0]
            for j, y in enumerate(temp_data):
                if _cluster_condition(x, y):
                    cluster.append(y)
                    k = m_idx[j]
                    mask[i + k + 1] = False
                    self.labels_[i + k + 1] = len(clusters) + 1

            cluster.append(x)
            clusters.append(np.array(cluster))
            self.labels_[i] = len(clusters)

        self.cluster_centers_ = [np.mean(cluster, axis=0) for cluster in clusters]
        self.inertia_ = sum([np.sum((cluster - self.cluster_centers_[i]) ** 2) for i, cluster in enumerate(clusters)])
        intra, inter = self._intra(clusters), self._inter()
        self.silhouette_ = self._silhouette(intra, inter)
        self.dunn_ = self._dunn(inter, max([len(x) for x in clusters]))

    @staticmethod
    def _intra(clusters):
        """
        Calculates the average intra-cluster distance for each cluster.
        Uses average distance between every item in a cluster.
        :param clusters: all clusters
        :return: intra-cluster distances
        """
        return [np.mean([np.mean(np.abs(np.array(cluster[i + 1:]) - np.array(x))) for i, x in enumerate(cluster[:-1])])
                if len(cluster) > 1 else 0 for cluster in clusters]

    def _inter(self):
        """
        Calculates the nearest inter-cluster distance for each cluster.
        Uses centroid distance.
        :return: distances
        """
        return [np.min(np.sum(np.abs(np.array(self.cluster_centers_[i + 1:]) - np.array(c)), axis=1)) for i, c in
                enumerate(self.cluster_centers_[:-1])]

    @staticmethod
    def _silhouette(intra, inter):
        """
        Calculates the Silhouette Coefficient:
            (mean inter-cluster distance - mean intra-cluster distance) /
            max(mean inter-cluster distance, mean intra-cluster distance)
        :param intra: array of intra-cluster distances
        :param inter: array of inter-cluster distances
        :return: Silhouette Coefficient
        """
        intra_ = np.mean(intra)
        inter_ = np.mean(inter)

        return (inter_ - intra_) / max(intra_, inter_)

    @staticmethod
    def _dunn(inter, max_size):
        """
        Calculates Dunn's Index:
            minimum inter-cluster distance / maximum cluster size
        :param inter: array of all inter-cluster distances
        :param max_size: maximum cluster size
        :return: Dunn's index
        """
        return np.min(inter) / max_size

    # noinspection PyUnusedLocal
    def fit_predict(self, X):
        """
        Dummy method for getting predicted cluster labels of data.
        Only returns existing labels from calling the fit method.
        :param X: data to predict - unused
        :return: predicted labels
        """
        return self.labels_


# noinspection PyPep8Naming,GrazieInspection
class KMeansCluster:
    """
    Model for KMeans clustering. Wraps sklearn implementation with extra functionality: Dunn's Index
    and Silhouette Coefficient.
    """

    def __init__(self, n_clusters, init, max_iter, n_init, random_state):
        """
        Basic model setup.
        :param n_clusters: number of clusters
        :param init: initialisation algorithm, i.e. k-means++
        :param max_iter: maximum number of iterations
        :param n_init: number of times to run with different centroid seeds
        :param random_state: sets random state for replication
        """
        self.n_clusters = n_clusters
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.silhouette_ = None
        self.dunn_ = None
        self.model = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, n_init=n_init,
                            random_state=random_state)

    def fit(self, X):
        """
        Fits the data.
        Wrapper for KMeans fit function, also calculates distance metrics - inertia, Dunn's Index and
        Silhouette Coefficient
        :param X: data to fit
        :return:
        """
        self.model.fit(X)
        self.labels_ = self.model.fit_predict(X)

        clusters = self._from_labels(X, [int(x) for x in self.labels_])
        self.cluster_centers_ = self.model.cluster_centers_
        self.inertia_ = self.model.inertia_
        intra, inter = self._intra(clusters), self._inter()
        self.silhouette_ = self._silhouette(intra, inter)
        self.dunn_ = self._dunn(inter, max([len(x) for x in clusters]))

    @staticmethod
    def _from_labels(X, labels):
        """
        Loads clusters from labels.
        :param X: records for labelling
        :param labels: cluster labels
        :return: clusters
        """
        xs = sorted(list(zip(labels, X)), key=lambda x: x[0])
        return [np.array(list([vv[1] for vv in v])) for _, v in it.groupby(xs, key=lambda x: x[0])]

    @staticmethod
    def _intra(clusters):
        """
        Calculates the average intra-cluster distance for each cluster.
        Uses average distance between every item in a cluster.
        :param clusters: all clusters
        :return: intra-cluster distances
        """
        return [np.mean([np.mean(np.abs(cluster[i + 1:] - x)) for i, x in enumerate(cluster[:-1])])
                if len(cluster) > 1 else 0 for cluster in clusters]

    def _inter(self):
        """
        Calculates the nearest inter-cluster distance for each cluster.
        Uses centroid distance.
        :return: distances
        """
        return [np.min(np.sum(np.abs(self.cluster_centers_[i + 1:] - c), axis=1)) for i, c in
                enumerate(self.cluster_centers_[:-1])]

    @staticmethod
    def _silhouette(intra, inter):
        """
        Calculates the Silhouette Coefficient:
            (mean inter-cluster distance - mean intra-cluster distance) /
            max(mean inter-cluster distance, mean intra-cluster distance)
        :param intra: array of intra-cluster distances
        :param inter: array of inter-cluster distances
        :return: Silhouette Coefficient
        """
        intra_ = np.mean(intra)
        inter_ = np.mean(inter)

        return (inter_ - intra_) / max(intra_, inter_)

    @staticmethod
    def _dunn(inter, max_size):
        """
        Calculates Dunn's Index:
            minimum inter-cluster distance / maximum cluster size
        :param inter: array of all inter-cluster distances
        :param max_size: maximum cluster size
        :return: Dunn's index
        """
        return np.min(inter) / max_size

    def fit_predict(self, X):
        """
        Fits the data to existing clusters.
        Wrapper for fit_predict on the KMeans object.
        :param X: data to fit
        :return: predicted cluster labels
        """
        return self.model.fit_predict(X)
