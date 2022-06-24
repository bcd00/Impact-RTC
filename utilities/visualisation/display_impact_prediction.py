import pandas as pd
import urllib.request
import dash_bootstrap_components as dbc

from plotly.io import from_json
from dash import html, dcc, callback, Output, Input, dash_table
from utilities.visualisation.vis_utils import DIV_STYLE_LEFT, DIV_STYLE_RIGHT, DIV_STYLE_PARENT, HOST, PORT, font, \
    build_table as bt

ID_PREFIX = 'dip'

file_config = {
    'lr_eval_dow': 'impact_prediction/linear_regression/manual_eval_weekday.json',
    'lr_test_dow': 'impact_prediction/linear_regression/manual_test_weekday.json',
    'lr_eval_tod': 'impact_prediction/linear_regression/manual_eval_time_of_day.json',
    'lr_test_tod': 'impact_prediction/linear_regression/manual_test_time_of_day.json',
    'lr_eval_both': 'impact_prediction/linear_regression/manual_eval_weekday_time_of_day.json',
    'lr_test_both': 'impact_prediction/linear_regression/manual_test_weekday_time_of_day.json',
    'nn_dow': 'impact_prediction/nn_model/nn_model_weekday.json',
    'nn_tod': 'impact_prediction/nn_model/nn_model_tod.json',
    'nn_both': 'impact_prediction/nn_model/nn_model_both.json',
    'nn_loss': 'impact_prediction/nn_model/nn_loss_graph.json',
    'arima_all': 'impact_prediction/arima/arima_all.json',
    'arima_london': 'impact_prediction/arima/arima_london.json',
    'anova': './output/figures/visualisation/anova.pickle',
    'arima_tuning': './output/figures/visualisation/arima_tuning.pickle',
    'arima_results': './output/figures/visualisation/arima_results.pickle',
    'nn_results': './output/figures/visualisation/nn_results.pickle',
    'lr_results': './output/figures/visualisation/lr_results.pickle'
}


# noinspection DuplicatedCode
def build_manual_figures():
    figs = [
        from_json(urllib.request.urlopen(url=f'{HOST}:{PORT}/{file_config["lr_eval_dow"]}').read()),
        from_json(urllib.request.urlopen(url=f'{HOST}:{PORT}/{file_config["lr_eval_tod"]}').read()),
        from_json(urllib.request.urlopen(url=f'{HOST}:{PORT}/{file_config["lr_eval_both"]}').read()),
        from_json(urllib.request.urlopen(url=f'{HOST}:{PORT}/{file_config["lr_test_dow"]}').read()),
        from_json(urllib.request.urlopen(url=f'{HOST}:{PORT}/{file_config["lr_test_tod"]}').read()),
        from_json(urllib.request.urlopen(url=f'{HOST}:{PORT}/{file_config["lr_test_both"]}').read())
    ]

    for fig in figs:
        fig.update_layout(template='plotly', font=font)

    return figs


def build_nn_figures():
    figs = [
        from_json(urllib.request.urlopen(url=f'{HOST}:{PORT}/{file_config["nn_dow"]}').read()),
        from_json(urllib.request.urlopen(url=f'{HOST}:{PORT}/{file_config["nn_tod"]}').read()),
        from_json(urllib.request.urlopen(url=f'{HOST}:{PORT}/{file_config["nn_both"]}').read()),
        from_json(urllib.request.urlopen(url=f'{HOST}:{PORT}/{file_config["nn_loss"]}').read())
    ]

    for fig in figs:
        fig.update_layout(template='plotly', font=font)

    return figs


def build_arima_figures():
    figs = [
        from_json(urllib.request.urlopen(url=f'{HOST}:{PORT}/{file_config["arima_all"]}').read()),
        from_json(urllib.request.urlopen(url=f'{HOST}:{PORT}/{file_config["arima_london"]}').read())
    ]

    for fig in figs:
        fig.update_layout(template='plotly', font=font)

    return figs


def build_table(filename, label):
    df = pd.read_pickle(filename)
    df.reset_index(drop=False, inplace=True)
    cols = [{'name': i, 'id': i} for i in df.columns]

    for col in cols:
        df[col['id']] = df[col['id']].astype(str)
    return html.Div(children=[
        dash_table.DataTable(data=df.to_dict('records'), columns=cols, id=f'{ID_PREFIX}_{label}_table',
                             style_header={'backgroundColor': '#0D6EFD', 'color': 'white'},
                             style_data={'whiteSpace': 'normal', 'height': 'auto',
                                         'color': 'black'})])


lr_figures = build_manual_figures()
nn_figures = build_nn_figures()
arima_figures = build_arima_figures()

overview_layout = html.Div(id=f'{ID_PREFIX}_overview_container', children=[
    html.H3('Results of 2-way ANOVA test on custom clusters'),
    html.Br(),
    build_table(filename=file_config['anova'], label='manual'),
    html.Br(),
    html.H3('Results of linear regression model'),
    html.Br(),
    bt(df=pd.read_pickle(file_config['lr_results']), id_='linear_regression_results'),
    html.Br(),
    html.H3('Results of neural network model'),
    html.Br(),
    bt(df=pd.read_pickle(file_config['nn_results']), id_='neural_network_results'),
    html.Br(),
    html.H3('Results of ARIMA model'),
    html.Br(),
    bt(df=pd.read_pickle(file_config['arima_results']), id_='arima_results')
], style=DIV_STYLE_PARENT)

# noinspection DuplicatedCode
lr_layout = html.Div(id=f'{ID_PREFIX}_lr_container', children=[
    html.Div(children=[
        html.H3('Performance of linear regression on validation data using day of week variable', style=DIV_STYLE_LEFT),
        html.H3('Performance of linear regression on validation data using time of day variable', style=DIV_STYLE_RIGHT)
    ], style=DIV_STYLE_PARENT),
    html.Br(),
    html.Div(children=[
        dcc.Graph(figure=lr_figures[0], style=DIV_STYLE_LEFT),
        dcc.Graph(figure=lr_figures[1], style=DIV_STYLE_RIGHT)
    ]),
    html.Br(),
    html.H3('Performance of linear regression on validation data using both variables'),
    html.Br(),
    dcc.Graph(figure=lr_figures[2]),
    html.Br(),
    html.Div(children=[
        html.H3('Performance of linear regression on testing data using day of week variable', style=DIV_STYLE_LEFT),
        html.H3('Performance of linear regression on testing data using time of day variable', style=DIV_STYLE_RIGHT)
    ], style=DIV_STYLE_PARENT),
    html.Br(),
    html.Div(children=[
        dcc.Graph(figure=lr_figures[3], style=DIV_STYLE_LEFT),
        dcc.Graph(figure=lr_figures[4], style=DIV_STYLE_RIGHT)
    ], style=DIV_STYLE_PARENT),
    html.Br(),
    html.H3('Performance of linear regression on testing data using both variables'),
    html.Br(),
    dcc.Graph(figure=lr_figures[5])
])

nn_layout = html.Div(id=f'{ID_PREFIX}_nn_container', children=[
    html.Div(children=[
        html.H3('Performance of neural network on testing data using day of week variable', style=DIV_STYLE_LEFT),
        html.H3('Performance of neural network on testing data using time of day variable', style=DIV_STYLE_RIGHT)
    ], style=DIV_STYLE_PARENT),
    html.Br(),
    html.Div(children=[
        dcc.Graph(figure=nn_figures[0], style=DIV_STYLE_LEFT),
        dcc.Graph(figure=nn_figures[1], style=DIV_STYLE_RIGHT)
    ], style=DIV_STYLE_PARENT),
    html.Br(),
    html.H3('Performance of neural network on testing data using both variables'),
    html.Br(),
    dcc.Graph(figure=nn_figures[2]),
    html.Br(),
    html.H3('Loss graph for neural network'),
    html.Br(),
    dcc.Graph(figure=nn_figures[3])
])

arima_layout = html.Div(id=f'{ID_PREFIX}_arima_container', children=[
    html.H3('Hyperparameter tuning'),
    bt(df=pd.read_pickle(file_config['arima_tuning']),
       id_='arima_tuning_results'),
    html.Br(),
    html.H3('Performance of ARIMA model on all data'),
    html.Br(),
    dcc.Graph(figure=arima_figures[0]),
    html.Br(),
    html.H3('Performance of ARIMA model on data within London'),
    html.Br(),
    dcc.Graph(figure=arima_figures[1])
])

# noinspection DuplicatedCode
layout = html.Div(id=f'{ID_PREFIX}_density_prediction_container', children=[
    dbc.Tabs(
        [
            dbc.Tab(label='Overview', tab_id='overview'),
            dbc.Tab(label='Linear Regression', tab_id='linear_regression'),
            dbc.Tab(label='Neural Network Model', tab_id='neural_network'),
            dbc.Tab(label='ARIMA', tab_id='arima')
        ],
        id=f'{ID_PREFIX}_density_prediction_view',
        active_tab='overview',
    ),
    html.Br(),
    html.Div(id=f'{ID_PREFIX}_main_container', children=[])
])


@callback(
    Output(f'{ID_PREFIX}_main_container', component_property='children'),
    Input(f'{ID_PREFIX}_density_prediction_view', component_property='active_tab')
)
def load(tab):
    match tab:
        case 'overview':
            return overview_layout
        case 'linear_regression':
            return lr_layout
        case 'neural_network':
            return nn_layout
        case 'arima':
            return arima_layout
