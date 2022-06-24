import pandas as pd
from dash import html, dcc
from plotly.io import from_json

from utilities.visualisation.vis_utils import DIV_STYLE_LEFT, DIV_STYLE_RIGHT, HOST, PORT, DIV_STYLE_PARENT, \
    font, open_json, build_table, image_style

ID_PREFIX = 'co'

file_config = {
    'jaccard': 'data_processing/annotations/jaccard_similarity.png',
    'lr_end': './output/figures/classifier/tuning/lr_end/lr_end_eval_graph.json',
    'lr_start': './output/figures/classifier/tuning/lr_start/lr_start_eval_graph.json',
    'batch_size': './output/figures/classifier/tuning/batch_size/batch_size_eval_graph.json',
    'model_name': './output/figures/classifier/tuning/model_name/model_name_results.pickle',
    'epochs': './output/figures/classifier/tuning/epochs/epochs_eval_graph.json',
    'scheduler_type': './output/figures/classifier/tuning/scheduler_type/scheduler_type_results.pickle',
    'preprocessing': './output/figures/classifier/tuning/preprocessing/preprocessing_results.pickle',
    'downsampling': './output/figures/classifier/tuning/downsample/downsample_results.pickle',
    'augmentation': './output/figures/classifier/tuning/augmentation/augmentation_eval_graph.json',
    'classifier_params': './output/figures/visualisation/classifier_params.pickle',
    'classifier_results': './output/figures/visualisation/classifier_results.pickle'
}


def build_hyper_params(params, style):
    return html.Div(id=f'{ID_PREFIX}_hyper_params', children=[
        html.Ul(children=[
            html.Li(f'Initial Learning Rate: {params["lr_start"]}'),
            html.Li(f'Final Learning Rate: {params["lr_end"]}'),
            html.Li(f'Batch Size: {params["batch_size"]}'),
            html.Li(f'Model Name: {params["model_name"]}'),
            html.Li(f'No. Epochs: {params["epochs"]}'),
            html.Li(f'Scheduler Type: {params["scheduler_type"]}'),
            html.Li(f'Downsample Training Data: {params["downsample"]}')
        ])
    ], style=style)


def load_figs():
    figs = [
        from_json(open_json(filename=file_config['lr_start'])),
        from_json(open_json(filename=file_config['lr_end'])),
        from_json(open_json(filename=file_config['batch_size'])),
        from_json(open_json(filename=file_config['epochs'])),
        from_json(open_json(filename=file_config['augmentation']))
    ]

    for fig in figs:
        fig.update_layout(template='plotly', font=font)

    return figs


def build_hyperparameter_tuning():
    figs = load_figs()

    preprocessing_legend = ['Convert HTML Entities', 'Convert Emojis', 'Strip Emojis', 'Strip Mentions',
                            'Strip Hashtags',
                            'Strip Newlines', 'Strip Links', 'Contextualise Hashtags (10_000, non-freq)',
                            'Contextualise Hashtags (10_000, freq)', 'Contextualise Hashtags (50_000, non-freq)',
                            'Contextualise Hashtags (50_000, freq)']
    return html.Div(children=[
        html.Div(id=f'{ID_PREFIX}_layer_five', children=[
            html.H3('Impact of initial learning rate on model performance', style=DIV_STYLE_LEFT),
            html.H3('Impact of final learning rate on model performance', style=DIV_STYLE_RIGHT)
        ], style=DIV_STYLE_PARENT),
        html.Br(),
        html.Div(id=f'{ID_PREFIX}_layer_six', children=[
            dcc.Graph(figure=figs[0], style=DIV_STYLE_LEFT),
            dcc.Graph(figure=figs[1], style=DIV_STYLE_RIGHT)
        ], style=DIV_STYLE_PARENT),
        html.Br(),
        html.Div(id=f'{ID_PREFIX}_layer_seven', children=[
            html.H3('Impact of batch size on model performance', style=DIV_STYLE_LEFT),
            html.H3('Impact of the number of epochs on model performance', style=DIV_STYLE_RIGHT)
        ], style=DIV_STYLE_PARENT),
        html.Br(),
        html.Div(id=f'{ID_PREFIX}_layer_seven', children=[
            dcc.Graph(figure=figs[2], style=DIV_STYLE_LEFT),
            dcc.Graph(figure=figs[3], style=DIV_STYLE_RIGHT)
        ], style=DIV_STYLE_PARENT),
        html.Br(),
        html.H3('Impact of text augmentation on model performance'),
        html.Br(),
        dcc.Graph(figure=figs[4]),
        html.Br(),
        html.H3('Impact of model type on model performance'),
        html.Br(),
        build_table(
            df=load_tuning_df(file_config['model_name'], label='model_name'),
            id_='model_type_tuning',
            column_id='model_name'
        ),
        html.Br(),
        html.H3('Impact of downsampling on model performance'),
        html.Br(),
        build_table(
            df=load_tuning_df(file_config['downsampling'], label='downsample'),
            id_='downsampling_tuning',
            column_id='downsample'
        ),
        html.Br(),
        html.H3('Impact of scheduler type on model performance'),
        html.Br(),
        build_table(
            df=load_tuning_df(file_config['scheduler_type'], label='scheduler_type'),
            id_='scheduler_type_tuning',
            column_id='scheduler_type'
        ),
        html.Br(),
        html.H3('Impact of pre-processing tasks on model performance'),
        html.Br(),
        build_table(
            df=load_tuning_df(file_config['preprocessing'], legend=preprocessing_legend, label='pre-processing'),
            id_='preprocessing_tuning',
            column_id='pre-processing'
        )
    ])


def build_final_model_results():
    return html.Div(children=[
        html.Div(id=f'{ID_PREFIX}_layer_one', children=[
            html.H3('Final Model\'s Results', style=DIV_STYLE_LEFT),
            html.H3('Final Model\'s Hyper-Parameters', style=DIV_STYLE_RIGHT)
        ], style=DIV_STYLE_PARENT),
        html.Br(),
        html.Div(id=f'{ID_PREFIX}_layer_two', children=[
            html.Div(children=build_table(
                df=pd.read_pickle(file_config['classifier_results']),
                id_=f'final_results'), style=DIV_STYLE_LEFT),
            html.Div(children=build_table(
                df=pd.read_pickle(file_config['classifier_params']),
                id_=f'final_params'), style=DIV_STYLE_RIGHT)
        ], style=DIV_STYLE_PARENT)
    ])


def load_tuning_df(filename, label, legend=None):
    df = pd.read_pickle(filename)
    if legend is not None:
        df[label] = legend
    return df[[label, 'eval_accuracy', 'eval_negative_f1', 'eval_positive_f1', 'eval_cohen\'s_kappa']].round(3)


def build_classifier_overview():
    return [
        build_final_model_results(),
        html.Br(),
        html.H3('Similarity between the classes in the labelled and external testing data'),
        html.Br(),
        html.Img(src=f'{HOST}:{PORT}/{file_config["jaccard"]}',
                 style=image_style),
        html.Br(),
        build_hyperparameter_tuning()
    ]


layout = build_classifier_overview()
