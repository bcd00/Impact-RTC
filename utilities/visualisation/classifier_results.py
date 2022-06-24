import numpy as np
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc

from utilities.utils import read_json, get_max_key
from dash import Input, Output, dcc, html, callback
from utilities.visualisation.vis_utils import plot_cfm, build_dropdown, build_table, font, image_style, text_style

ID_PREFIX = 'cr'

file_config = {
    'validation_dataset': './shared/labelled.pickle',
    'testing_dataset': './shared/tedf.pickle',
    'validation_predictions': './output/figures/classifier/final_model/validation/predictions.txt',
    'testing_predictions': './output/figures/classifier/final_model/testing/predictions.txt',
    'keys': './output/config/keys.json',
    'values': './output/config/values.json'
}


def load_datasets():
    tvdf_ = pd.read_pickle(file_config['validation_dataset'])[['text', 'label']].copy()
    tedf_ = pd.read_pickle(file_config['testing_dataset'])

    tvdf_['label'] = tvdf_['label'].astype(int)

    with open(file_config['validation_predictions'], 'r', encoding='utf-8') as f:
        tvdf_preds = [int(x) for x in f.read().split(',')]
    with open(file_config['testing_predictions'], 'r', encoding='utf-8') as f:
        tedf_preds = [int(x) for x in f.read().split(',')]

    tvdf_['prediction'] = tvdf_preds
    tedf_['prediction'] = tedf_preds

    return tvdf_, tedf_


def build_controls(keys: dict, values: dict):
    max_key = get_max_key(keys, values)

    def get_value(label):
        return {k[label] for k in keys.values()}, keys[max_key][label]

    lr_starts, lr_starts_value = get_value(label='lr_start')
    lr_ends, lr_ends_value = get_value(label='lr_end')
    batch_sizes, batch_sizes_value = get_value(label='batch_size')
    model_names, model_names_value = get_value(label='model_name')
    epochs, epochs_value = get_value(label='epochs')
    scheduler_types, scheduler_types_value = get_value(label='scheduler_type')
    ds_value = str(keys[max_key]['downsample'])

    return dbc.Card([
        build_dropdown(
            label='Initial Learning Rate',
            idx='lr_start',
            cols=lr_starts,
            value=lr_starts_value
        ),
        build_dropdown(
            label='Final Learning Rate',
            idx='lr_end',
            cols=lr_ends,
            value=lr_ends_value
        ),
        build_dropdown(
            label='Batch Size',
            idx='batch_size',
            cols=batch_sizes,
            value=batch_sizes_value
        ),
        build_dropdown(
            label='Model Name',
            idx='model_name',
            cols=model_names,
            value=model_names_value
        ),
        build_dropdown(
            label='No. Epochs',
            idx='epochs',
            cols=epochs,
            value=epochs_value
        ),
        build_dropdown(
            label='Scheduler Type',
            idx='scheduler_type',
            cols=scheduler_types,
            value=scheduler_types_value
        ),
        build_dropdown(
            label='Downsample Training Data',
            idx='downsample',
            cols=['False', 'True'],
            value=ds_value
        )
    ], body=True, style={'flex': 1, 'marginRight': '20px'})


def build_samples(df: pd.DataFrame):
    return html.Div(children=[
        html.H5('Sample True Positives'),
        build_table(
            df=df,
            data=df[(df.label == '1') & (df.prediction == '1')].head().to_dict('records'),
            id_='tp_table'
        ),
        html.Br(),
        html.H5('Sample False Positives'),
        build_table(
            df=df,
            data=df[(df.label == '0') & (df.prediction == '1')].head().to_dict('records'),
            id_='fp_table'
        ),
        html.Br(),
        html.H5('Sample True Negatives'),
        build_table(
            df=df,
            data=df[(df.label == '0') & (df.prediction == '0')].head().to_dict('records'),
            id_='tn_table'
        ),
        html.Br(),
        html.H5('Sample False Negatives'),
        build_table(
            df=df,
            data=df[(df.label == '1') & (df.prediction == '0')].head().to_dict('records'),
            id_='fn_table'
        ),
    ])


def build_classifier_model_results():
    keys = read_json(file_config['keys'])
    values = read_json(file_config['values'])
    controls = build_controls(keys, values)
    return html.Div(children=[
        controls,
        html.Div(id='classifier_display', children=[], style={'flex': 3})
    ], style={'display': 'flex'})


tvdf, tedf = load_datasets()
layout = build_classifier_model_results()


@callback(
    Output(component_id='classifier_display', component_property='children'),
    Input(component_id='lr_start', component_property='value'),
    Input(component_id='lr_end', component_property='value'),
    Input(component_id='batch_size', component_property='value'),
    Input(component_id='model_name', component_property='value'),
    Input(component_id='epochs', component_property='value'),
    Input(component_id='scheduler_type', component_property='value'),
    Input(component_id='downsample', component_property='value'),
)
def build_classifier_results(lr_start, lr_end, batch_size, model_name, epoch, scheduler_type, use_ds):
    keys = read_json(filename=file_config['keys'])
    values = read_json(filename=file_config['values'])
    key = {
        'lr_start': lr_start,
        'lr_end': lr_end,
        'batch_size': batch_size,
        'model_name': model_name,
        'GAS': 1,
        'epochs': epoch,
        'scheduler_type': scheduler_type,
        'downsample': True if use_ds == 'True' else False
    }
    key = [k for k, v in keys.items() if v == key]

    if not key:
        return html.Div('No data is available for these hyperparameters')

    value = values[str(key[0])]

    losses_df = pd.DataFrame({'Iteration': list(range(len(value['losses']))), 'Loss': value['losses']})
    loss_fig = px.line(losses_df, x='Iteration', y='Loss', title='Loss Graph')
    loss_fig.update_layout(template='plotly', font=font)

    lrs_df = pd.DataFrame({'Iteration': list(range(len(value['lrs']))), 'Learning Rate': value['lrs']})
    lrs_fig = px.line(lrs_df, x='Iteration', y='Learning Rate', title='Learning Rate Graph')
    lrs_fig.update_layout(template='plotly', font=font)

    eval_cfm = plot_cfm(np.array(value['eval_confusion_matrix']))
    test_cfm = plot_cfm(np.array(value['test_confusion_matrix']))

    samples = html.Div(id='samples', children=[build_samples(tedf)])
    eval_report = value['eval_report']
    test_report = value['test_report']

    df = pd.DataFrame({'Dataset': ['Validation', 'Validation - Baseline', 'Testing', 'Testing - Baseline'],
                       'Accuracy': [eval_report['accuracy'], 0.957, test_report['accuracy'], 0.845],
                       'Negative F1-Score': [eval_report['negative']['f1-score'], 0.968,
                                             test_report['negative']['f1-score'], 0.834],
                       'Positive F1-Score': [eval_report['positive']['f1-score'], 0.935,
                                             test_report['positive']['f1-score'], 0.854]})

    return html.Div(children=[
        html.H3('Model Results', style=dict(**image_style, **{'text-align': 'center'})),
        html.Br(),
        build_table(df=df, id_=f'model_results_interactive'),
        html.Br(),
        html.H5('Confusion Matrix for Validation Data', style=text_style),
        html.Br(),
        html.Img(id='eval_confusion_matrix', src=eval_cfm, style=image_style),
        html.Br(),
        html.H5('Confusion Matrix for Testing Data', style=text_style),
        html.Br(),
        html.Img(id='test_confusion_matrix', src=test_cfm, style=image_style),
        html.Br(),
        dcc.RadioItems(['Validation', 'Testing'], 'Testing', id='sample_options'),
        samples,
        html.Br(),
        dcc.Graph(id='losses_graph', figure=loss_fig),
        html.Br(),
        dcc.Graph(id='lrs_graph', figure=lrs_fig),
    ])


@callback(
    Output('samples', 'children'),
    Input('sample_options', 'value')
)
def visualise_samples(option):
    match option:
        case 'Testing':
            return build_samples(tedf)
        case 'Validation':
            return build_samples(tvdf)
        case _:
            raise ValueError(f'Unknown sample option: {option}')
