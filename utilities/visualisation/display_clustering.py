import pandas as pd
import urllib.request
import dash_bootstrap_components as dbc

from plotly.io import from_json
from dash import html, callback, Output, Input, dcc
from utilities.visualisation.vis_utils import open_json, font, build_table, DIV_STYLE_PARENT, DIV_STYLE_RIGHT, \
    DIV_STYLE_LEFT, HOST, PORT
from utilities.utils import custom_tuning_dir, kmeans_tuning_dir
from utilities.visualisation.display_custom_clustering import layout as manual_layout
from utilities.visualisation.display_kmeans_clustering import layout as kmeans_layout

ID_PREFIX = 'dc'

file_config = {
    'custom_sl_dunn': f'{custom_tuning_dir}/manual_sl_dunn_graph.json',
    'custom_sl_silhouette': f'{custom_tuning_dir}/manual_sl_silhouette_graph.json',
    'custom_sl_wcss': f'{custom_tuning_dir}/manual_sl_wcss_graph.json',
    'custom_tl_dunn': f'{custom_tuning_dir}/manual_tl_dunn_graph.json',
    'custom_tl_silhouette': f'{custom_tuning_dir}/manual_tl_silhouette_graph.json',
    'custom_tl_wcss': f'{custom_tuning_dir}/manual_tl_wcss_graph.json',
    'kmeans_n_dunn': f'{kmeans_tuning_dir}/kmeans_n_dunn_graph.json',
    'kmeans_n_silhouette': f'{kmeans_tuning_dir}/kmeans_n_silhouette_graph.json',
    'kmeans_n_wcss': f'{kmeans_tuning_dir}/kmeans_n_wcss_graph.json',
    'tod_count': 'impact_prediction/data_exploration/time_of_day_manual_hist.json',
    'dow_count': 'impact_prediction/data_exploration/weekday_manual_hist.json',
    'tod_size': 'impact_prediction/data_exploration/tod_size_manual_hist.json',
    'dow_size': 'impact_prediction/data_exploration/dow_size_manual_hist.json'
}

# noinspection DuplicatedCode
layout = html.Div(id=f'{ID_PREFIX}_clustering_container', children=[
    dbc.Tabs(
        [
            dbc.Tab(label='Overview', tab_id='overview'),
            dbc.Tab(label='Custom', tab_id='custom'),
            dbc.Tab(label='KMeans', tab_id='kmeans'),
        ],
        id=f'{ID_PREFIX}_clustering_view',
        active_tab='overview',
    ),
    html.Br(),
    html.Div(id=f'{ID_PREFIX}_main_container', children=[])
])


# noinspection DuplicatedCode
def build_overview_figures():
    figs = [
        from_json(urllib.request.urlopen(url=f'{HOST}:{PORT}/{file_config["tod_count"]}').read()),
        from_json(urllib.request.urlopen(url=f'{HOST}:{PORT}/{file_config["dow_count"]}').read()),
        from_json(urllib.request.urlopen(url=f'{HOST}:{PORT}/{file_config["tod_size"]}').read()),
        from_json(urllib.request.urlopen(url=f'{HOST}:{PORT}/{file_config["dow_size"]}').read()),
    ]

    for fig in figs:
        fig.update_layout(template='plotly')

    return figs


overview_figures = build_overview_figures()


# noinspection DuplicatedCode
def build_overview():
    figs = [
        from_json(open_json(filename=file_config['custom_sl_dunn'])),
        from_json(open_json(filename=file_config['custom_sl_silhouette'])),
        from_json(open_json(filename=file_config['custom_sl_wcss'])),
        from_json(open_json(filename=file_config['custom_tl_dunn'])),
        from_json(open_json(filename=file_config['custom_tl_silhouette'])),
        from_json(open_json(filename=file_config['custom_tl_wcss'])),
        from_json(open_json(filename=file_config['kmeans_n_dunn'])),
        from_json(open_json(filename=file_config['kmeans_n_silhouette'])),
        from_json(open_json(filename=file_config['kmeans_n_wcss']))
    ]

    for fig in figs:
        fig.update_layout(font=font)

    df = pd.DataFrame({'metric': ['Rand Index', 'Adjusted Rand Index'], 'value': [0.9998, 0.6516]})

    return html.Div(id=f'{ID_PREFIX}_overview', children=[
        html.H3('Algorithmic Similarity'),
        build_table(df=df, id_='algorithmic_similarity'),
        html.Br(),
        html.H3('Results of custom clustering hyperparameter tuning'),
        html.H6('Dunn\'s Index - Space Limit'),
        dcc.Graph(figure=figs[0]),
        html.H6('Silhouette Coefficient - Space Limit'),
        dcc.Graph(figure=figs[1]),
        html.H6('Within Cluster Sum of Squares - Space Limit'),
        dcc.Graph(figure=figs[2]),
        html.H6('Dunn\'s Index - Time Limit'),
        dcc.Graph(figure=figs[3]),
        html.H6('Silhouette Coefficient - Time Limit'),
        dcc.Graph(figure=figs[4]),
        html.H6('Within Cluster Sum of Squares - Time Limit'),
        dcc.Graph(figure=figs[5]),
        html.H3('Results of K-Means clustering hyperparameter tuning'),
        html.H6('Dunn\'s Index - No. of Clusters'),
        dcc.Graph(figure=figs[6]),
        html.H6('Silhouette Coefficient - No. of Clusters'),
        dcc.Graph(figure=figs[7]),
        html.H6('Within Cluster Sum of Squares - No. of Clusters'),
        dcc.Graph(figure=figs[8]),
        html.Div(children=[
            html.H3('Distribution of custom clusters by time of day', style=DIV_STYLE_LEFT),
            html.H3('Distribution of custom clusters by day of the week', style=DIV_STYLE_RIGHT)
        ], style=DIV_STYLE_PARENT),
        html.Div(children=[
            dcc.Graph(figure=overview_figures[0], style=DIV_STYLE_LEFT),
            dcc.Graph(figure=overview_figures[1], style=DIV_STYLE_RIGHT)
        ], style=DIV_STYLE_PARENT),
        html.Div(children=[
            html.H3('Average size of custom clusters by time of day', style=DIV_STYLE_LEFT),
            html.H3('Average size of custom clusters by day of the week', style=DIV_STYLE_RIGHT)
        ], style=DIV_STYLE_PARENT),
        html.Div(children=[
            dcc.Graph(figure=overview_figures[2], style=DIV_STYLE_LEFT),
            dcc.Graph(figure=overview_figures[3], style=DIV_STYLE_RIGHT)
        ], style=DIV_STYLE_PARENT)
    ])


@callback(
    Output(f'{ID_PREFIX}_main_container', component_property='children'),
    Input(f'{ID_PREFIX}_clustering_view', component_property='active_tab')
)
def handle_tab(active):
    match active:
        case 'overview':
            return build_overview()
        case 'custom':
            return manual_layout
        case 'kmeans':
            return kmeans_layout
