import itertools

from tqdm.auto import tqdm
from utilities.clustering.cluster_model import ClusterModel
from utilities.utils import shared_dir, write_json


def cluster_custom(cluster_df, timezones):
    distances = []
    time_limits = [1, 3, 6, 9, 12, 15, 18, 21, 24]
    space_limits = [1, 5, 10, 15, 20, 25, 30, 35, 40]

    for time_limit, space_limit in tqdm(list(itertools.product(time_limits, space_limits))):
        cm = ClusterModel(df=cluster_df)
        cm.fit('custom', time_limit, space_limit)

        distances.extend(cm.distances)
        timezones = cm.format_data(timezones)
        formatted_filename = f'{shared_dir}/custom_clusters/formatted/{time_limit}_{space_limit}.json'

        cm.to_file(labelled_filename=f'{shared_dir}/custom_clusters/labelled/{time_limit}_{space_limit}.json',
                   formatted_filename=formatted_filename)
        write_json({'min_date': cm.min_date, 'max_date': cm.max_date},
                   filename=f'{shared_dir}/custom_clusters/config/{time_limit}_{space_limit}.json')
        write_json(cm.data_to_visualise(filter_type='time'),
                   filename=f'{shared_dir}/custom_clusters/geojson/{time_limit}_{space_limit}.json')

    return distances, timezones


def cluster_kmeans(cluster_df, timezones):
    n_clusters = list(range(5000, 16000, 1000))
    distances = []

    for n in tqdm(n_clusters):
        cm = ClusterModel(df=cluster_df)
        cm.fit('kmeans', n)

        distances.extend(cm.distances)
        timezones = cm.format_data(timezones)

        cm.to_file(labelled_filename=f'{shared_dir}/kmeans_clusters/labelled/{n}.json',
                   formatted_filename=f'{shared_dir}/kmeans_clusters/formatted/{n}.json')
        write_json({'min_date': cm.min_date, 'max_date': cm.max_date},
                   filename=f'{shared_dir}/kmeans_clusters/config/{n}.json')
        write_json(cm.data_to_visualise(filter_type='time'),
                   filename=f'{shared_dir}/kmeans_clusters/geojson/{n}.json')

    return distances, timezones
