import datetime
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc

from dash import html, dcc, callback, Input, Output
from utilities.classifier.nlp_model import NLPModel
from utilities.clustering.cluster_model import ClusterModel
from utilities.data_processing.preprocessing import PreProcessing
from utilities.utils import read_json
from utilities.visualisation.vis_utils import build_table

file_config = {
    'geospatial': './shared/geospatial.pickle',
    'ids': './output/figures/visualisation/ids.json',
    'model': './shared/model',
    'tokenizer': './shared/tokenizer'
}

xs = read_json(filename=file_config['ids'])
geospatial = pd.read_pickle(file_config['geospatial'])
original_data = geospatial[geospatial.index.isin([str(x) for x in xs])].reset_index(drop=False, inplace=False)
preprocessed_data = original_data.copy()
classified_data = original_data.copy()

trdf = geospatial.head().copy()
tvdf = geospatial.head().copy()
trdf['label'] = [i for i in range(5)]
tvdf['label'] = [i for i in range(5)]

model = NLPModel(
    training_data=trdf,
    validation_data=tvdf,
    device='cuda',
    use_downsampling=True,
    batch_size=32,
    gradient_accumulation_steps=1,
    epochs=2,
    scheduler_type='linear',
    model_name='roberta-base',
    learning_rate=1e-4,
    learning_rate_end=1e-6,
    model_filename=file_config['model'],
    tokenizer_filename=file_config['tokenizer']
)


def build_controls():
    preprocessing_tasks = ['Strip Newlines', 'Strip Links', 'Strip Emojis', 'Strip Hashtags', 'Convert Emojis',
                           'Convert Hashtags', 'Convert HTML', 'Strip Mentions']
    return html.Div(children=[
        dbc.Label('Pre-processing Task'),
        dcc.Dropdown(
            id='preprocessing_dropdown',
            options=[{'label': col, 'value': col} for col in preprocessing_tasks],
            value=None,
            multi=True
        )
    ])


def _preprocess(xs_):
    df = original_data.copy()
    for x in xs_:
        match x:
            case 'Strip Newlines':
                df = PreProcessing(df=df, silent=True).strip_newlines().df
            case 'Strip Links':
                df = PreProcessing(df=df, silent=True).strip_links().df
            case 'Strip Emojis':
                df = PreProcessing(df=df, silent=True).strip_emojis().df
            case 'Strip Hashtags':
                df = PreProcessing(df=df, silent=True).strip_hashtags().df
            case 'Convert Emojis':
                df = PreProcessing(df=df, silent=True).emojis().df
            case 'Convert Hashtags':
                df = PreProcessing(df=df, silent=True).contextualise_hashtags(
                    use_frequencies=True).df
            case 'Convert HTML':
                df = PreProcessing(df=df, silent=True).convert_html_entities().df
            case 'Strip Mentions':
                df = PreProcessing(df=df, silent=True).strip_mentions().df
    return df


@callback(
    Output(component_id='preprocessed_container', component_property='children'),
    Output(component_id='models_container', component_property='children'),
    Input(component_id='preprocessing_dropdown', component_property='value'),
)
def preprocess(xs_):
    global preprocessed_data
    default_models = [
        html.H3('Classifier Model'),
        html.Br(),
        html.Button('Run Classifier', id='run_classifier', n_clicks=0),
        html.Br(),
        html.Br(),
        html.Div(id='classifier_results', children=[]),
        html.Br(),
        html.Div(id='clustering_container', children=[])
    ]

    if not xs_:
        preprocessed_data = original_data.copy()
        return build_table(df=preprocessed_data['text'].to_frame(), id_='preprocessed_data'), default_models

    preprocessed_data = _preprocess(xs_)
    return build_table(df=preprocessed_data['text'].to_frame(), id_='preprocessed_data'), default_models


@callback(
    Output(component_id='classifier_results', component_property='children'),
    Output(component_id='clustering_container', component_property='children'),
    Input(component_id='run_classifier', component_property='n_clicks')
)
def run_model(n_clicks):
    global classified_data
    if n_clicks == 0:
        return [], []

    classified_data = preprocessed_data.copy()
    temp_data = model.fit(classified_data['text'].copy().to_frame(), silent=True)
    classified_data['label'] = temp_data['label'].tolist()

    return build_table(df=classified_data[['text', 'label']].copy(), id_='classified_data'), [
        html.H3('Clustering Model'),
        html.Br(),
        html.Button('Run Clustering', id='run_clustering', n_clicks=0),
        html.Br(),
        html.Br(),
        html.Div(id='clustering_results', children=[])
    ]


def _cluster():
    dfs = []
    cm = ClusterModel(df=classified_data[classified_data.label == 1].copy())

    cm.generate_data()
    cm.fit('custom', time_limit=9, space_limit=10)

    for k, v in cm.labelled_clusters.items():
        df = pd.DataFrame(v)
        df['cluster'] = k
        dfs.append(df)

    df = pd.concat(dfs)
    df['x'] = df['location'].apply(lambda x: float(x[0]))
    df['y'] = df['location'].apply(lambda x: float(x[1]))
    df['location'] = df['location'].apply(lambda x: f'({float(x[0])}, {float(x[1])})')
    df['time'] = df['time'].apply(
        lambda x: (datetime.datetime(year=2000, month=1, day=1) + datetime.timedelta(seconds=int(x))).strftime(
            '%d/%m/%Y, %H:%M:%S'))

    fig = px.scatter_mapbox(
        df,
        lat=df.x,
        lon=df.y,
        hover_name=df.cluster,
        zoom=1,
        mapbox_style='open-street-map'
    )
    return df, fig


@callback(
    Output(component_id='clustering_results', component_property='children'),
    Input(component_id='run_clustering', component_property='n_clicks')
)
def run_clustering(n_clicks):
    if n_clicks == 0:
        return []

    df, fig = _cluster()

    return html.Div(children=[
        build_table(df=df[['text', 'label', 'cluster', 'location', 'time']].copy(), id_='clustered_data'),
        html.Br(),
        dcc.Graph(figure=fig)
    ])


layout = html.Div(children=[
    html.H1('Interactive Model Pipeline'),
    html.Br(),
    html.H3('Data Processing'),
    build_controls(),
    html.Br(),
    html.Div(id='preprocessed_container',
             children=[]),
    html.Br(),
    html.Div(id='models_container', children=[
        html.H3('Classifier Model'),
        html.Br(),
        html.Button('Run Classifier', id='run_classifier', n_clicks=0),
        html.Br(),
        html.Br(),
        html.Div(id='classifier_results', children=[]),
        html.Br(),
        html.Div(id='clustering_container', children=[])
    ])
])
