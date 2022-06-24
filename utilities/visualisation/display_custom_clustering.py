import glob
import pandas as pd
import geopandas as gpd
import plotly.express as px

from datetime import datetime
from utilities.utils import read_json
from utilities.visualisation.vis_utils import build_dropdown
from utilities.visualisation.app import long_callback_manager, app
from dash import dcc, html, dash_table, State, Input, Output, callback


def get_time_limit(f, i):
    return int(f[:-i].split('/')[-1].split('_')[-2])


def get_space_limit(f, i):
    return int(f[:-i].split('/')[-1].split('_')[-1])


file_config = {
    'geojson': './shared/custom_clusters/geojson',
    'config': './shared/custom_clusters/config',
    'labelled': './shared/custom_clusters/labelled',
    'labels': './output/clustering_labels'
}

def_time_limit = 9
def_space_limit = 10

initial_df = gpd.GeoDataFrame.from_features(
    read_json(f'{file_config["geojson"]}/{def_time_limit}_{def_space_limit}.json')[0]['features'])
initial_config = read_json(f'{file_config["config"]}/{def_time_limit}_{def_space_limit}.json')

custom_params = [{'time_limit': get_time_limit(f, 4), 'space_limit': get_space_limit(f, 4)} for f in
                 glob.glob(f'{file_config["labels"]}/labels_manual_*.txt')]

time_limit_default_header = f'Time Limit: {def_time_limit}'
space_limit_default_header = f'Space Limit: {def_space_limit}'
date_picker_default_header = f'Date \
Selected: {datetime.strftime(datetime.strptime(initial_config["min_date"], "%Y-%m-%d"), "%d-%m-%Y")}'

layout = [
    build_dropdown(
        label='Time Limit',
        idx='custom_time_limit',
        cols=sorted({x['time_limit'] for x in custom_params}),
        value=def_time_limit
    ),
    build_dropdown(
        label='Space Limit',
        idx='custom_space_limit',
        cols=sorted({x['space_limit'] for x in custom_params}),
        value=def_space_limit
    ),
    dcc.DatePickerSingle(
        id='custom_date_picker',
        min_date_allowed=initial_config['min_date'],
        max_date_allowed=initial_config['max_date'],
        initial_visible_month=initial_config['min_date'],
        date=initial_config['min_date']
    ),
    html.Div(id=f'custom_time_limit_header', children=time_limit_default_header),
    html.Div(id=f'custom_space_limit_header', children=space_limit_default_header),
    html.Div(id=f'custom_date_picker_header', children=date_picker_default_header),
    html.Div(id='custom_map_container', children=[
        dcc.Graph(id='custom_map',
                  figure=px.scatter_mapbox(initial_df, lat=initial_df.geometry.y, lon=initial_df.geometry.x,
                                           hover_name=initial_df.id, zoom=1, mapbox_style='open-street-map'))
    ]),
    html.Div(id='custom_table_container', children=[])
]


# noinspection DuplicatedCode
@app.long_callback(
    output=[
        Output('custom_map_container', component_property='children'),
        Output('custom_time_limit_header', component_property='children'),
        Output('custom_space_limit_header', component_property='children'),
        Output('custom_date_picker_header', component_property='children')
    ],
    inputs=[
        Input('custom_time_limit', component_property='value'),
        Input('custom_space_limit', component_property='value'),
        Input('custom_date_picker', component_property='date'),
    ],
    manager=long_callback_manager,
    prevent_initial_call=True
)
def load(time_limit, space_limit, date):
    config = read_json(filename=f'{file_config["config"]}/{time_limit}_{space_limit}.json')

    option_raw = datetime.strptime(date, '%Y-%m-%d')
    min_date = datetime.strptime(config['min_date'], '%Y-%m-%d')
    option = (option_raw - min_date).days

    layers = read_json(filename=f'{file_config["geojson"]}/{time_limit}_{space_limit}.json')
    dff = gpd.GeoDataFrame.from_features(layers[option]['features'])

    time_limit_header = f'Time Limit: {time_limit}'
    space_limit_header = f'Space Limit: {space_limit}'
    date_header = f'Date Selected: {datetime.strftime(option_raw, "%d-%m-%Y")}'

    if len(dff) == 0:
        return [html.Div('No clusters on this date')], time_limit_header, space_limit_header, date_header

    fig = px.scatter_mapbox(dff,
                            lat=dff.geometry.y,
                            lon=dff.geometry.x,
                            hover_name=dff.id,
                            zoom=1,
                            mapbox_style='open-street-map')
    return [dcc.Graph('custom_map', figure=fig)], time_limit_header, space_limit_header, date_header


@callback(
    Output('custom_table_container', component_property='children'),
    Input('custom_map', component_property='hoverData'),
    State('custom_time_limit', component_property='value'),
    State('custom_space_limit', component_property='value')
)
def on_hover(hover_data, time_limit, space_limit):
    print('Updating on Hover')
    if hover_data is None:
        return []

    idx = str(hover_data['points'][0]['hovertext'])
    return build_table(idx, time_limit, space_limit)


def build_table(idx, time_limit, space_limit):
    data = read_json(f'{file_config["labelled"]}/{time_limit}_{space_limit}.json')[idx]
    df = pd.DataFrame(data)
    cols = [{'name': i, 'id': i} for i in df.columns]

    for col in cols:
        df[col['id']] = df[col['id']].astype(str)
    return html.Div(children=[
        html.Div(id=f'custom_table_id_header', children=f'Cluster ID: {idx}'),
        dash_table.DataTable(data=df.to_dict('records'), columns=cols, id='custom_table',
                             style_header={'backgroundColor': '#0D6EFD', 'color': 'white'},
                             style_data={'whiteSpace': 'normal', 'height': 'auto',
                                         'color': 'black'})])
