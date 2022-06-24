import glob
import pandas as pd
import geopandas as gpd
import plotly.express as px

from datetime import datetime
from utilities.utils import read_json
from utilities.visualisation.vis_utils import build_dropdown
from utilities.visualisation.app import long_callback_manager, app
from dash import dcc, html, dash_table, State, Input, Output, callback


def get_n_clusters(f):
    return int(f[:-4].split('/')[-1].split('_')[-1])


file_config = {
    'labels': './output/clustering_labels',
    'config': './shared/kmeans_clusters/config',
    'geojson': './shared/kmeans_clusters/geojson',
    'labelled': './shared/kmeans_clusters/labelled'
}

def_n_clusters = 7000
initial_df = gpd.GeoDataFrame.from_features(read_json(f'{file_config["geojson"]}/{def_n_clusters}.json')[0]['features'])
initial_config = read_json(f'{file_config["config"]}/{def_n_clusters}.json')

kmeans_params = [get_n_clusters(f) for f in glob.glob(f'{file_config["labels"]}/labels_kmeans_*.txt')]

n_clusters_default_header = f'No. of Clusters: {def_n_clusters}'
date_picker_default_header = f'Date \
Selected: {datetime.strftime(datetime.strptime(initial_config["min_date"], "%Y-%m-%d"), "%d-%m-%Y")}'

layout = [
    build_dropdown(
        label='No. of Clusters',
        idx='kmeans_n_clusters',
        cols=sorted(kmeans_params),
        value=def_n_clusters
    ),
    dcc.DatePickerSingle(
        id='kmeans_date_picker',
        min_date_allowed=initial_config['min_date'],
        max_date_allowed=initial_config['max_date'],
        initial_visible_month=initial_config['min_date'],
        date=initial_config['min_date']
    ),
    html.Div(id=f'kmeans_n_clusters_header', children=n_clusters_default_header),
    html.Div(id=f'kmeans_date_picker_header', children=date_picker_default_header),
    html.Div(id='kmeans_map_container', children=[
        dcc.Graph(id='map', figure=px.scatter_mapbox(initial_df, lat=initial_df.geometry.y, lon=initial_df.geometry.x,
                                                     hover_name=initial_df.id, zoom=1, mapbox_style='open-street-map'))
    ]),
    html.Div(id='kmeans_table_container', children=[])
]


# noinspection DuplicatedCode
@app.long_callback(
    output=[
        Output('kmeans_map_container', component_property='children'),
        Output('kmeans_n_clusters_header', component_property='children'),
        Output('kmeans_date_picker_header', component_property='children')
    ],
    inputs=[
        Input('kmeans_n_clusters', component_property='value'),
        Input('kmeans_date_picker', component_property='date'),
    ],
    manager=long_callback_manager,
    prevent_initial_call=True
)
def load(n_clusters, date):
    config = read_json(filename=f'{file_config["config"]}/{n_clusters}.json')

    option_raw = datetime.strptime(date, '%Y-%m-%d')
    min_date = datetime.strptime(config['min_date'], '%Y-%m-%d')
    option = (option_raw - min_date).days

    layers = read_json(filename=f'{file_config["geojson"]}/{n_clusters}.json')
    dff = gpd.GeoDataFrame.from_features(layers[option]['features'])

    n_clusters_header = f'No. of Clusters: {n_clusters}'
    date_header = f'Date Selected: {datetime.strftime(option_raw, "%d-%m-%Y")}'

    if len(dff) == 0:
        return [html.Div('No clusters on this date')], n_clusters_header, date_header

    fig = px.scatter_mapbox(dff,
                            lat=dff.geometry.y,
                            lon=dff.geometry.x,
                            hover_name=dff.id,
                            zoom=1,
                            mapbox_style='open-street-map')
    return [dcc.Graph('kmeans_map', figure=fig)], n_clusters_header, date_header


@callback(
    Output('kmeans_table_container', component_property='children'),
    Input('kmeans_map', component_property='hoverData'),
    State('kmeans_n_clusters', component_property='value'),
)
def on_hover(hover_data, n_clusters):
    print('Updating on Hover')
    if hover_data is None:
        return []

    idx = str(hover_data['points'][0]['hovertext'])
    return build_table(idx, n_clusters)


# noinspection DuplicatedCode
def build_table(idx, n_clusters):
    data = read_json(f'{file_config["labelled"]}/{n_clusters}.json')[idx]
    df = pd.DataFrame(data)
    cols = [{'name': i, 'id': i} for i in df.columns]

    for col in cols:
        df[col['id']] = df[col['id']].astype(str)
    return html.Div(children=[
        html.Div(id=f'kmeans_table_id_header', children=f'Cluster ID: {idx}'),
        dash_table.DataTable(data=df.to_dict('records'), columns=cols, id='kmeans_table',
                             style_header={'backgroundColor': '#0D6EFD', 'color': 'white'},
                             style_data={'whiteSpace': 'normal', 'height': 'auto',
                                         'color': 'black'})])
