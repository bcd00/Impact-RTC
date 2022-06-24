import io
import base64
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import dash_bootstrap_components as dbc

from dash import html, dcc, dash_table
from utilities.utils import config

SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '16rem',
    'padding': '2rem 1rem',
    'backgroundColor': '#f8f9fa',
}

CONTENT_STYLE = {
    'marginLeft': '18rem',
    'marginRight': '2rem',
    'padding': '2rem 1rem',
}

IMAGE_STYLE = {
    'display': 'block',
    'marginLeft': 'auto',
    'marginRight': 'auto',
    'width': '100%',
}

DIV_STYLE_LEFT = {
    'display': 'block',
    'marginLeft': '0em',
    'marginRight': '1em',
    'width': '45%',
    'height': 'auto',
    'float': 'left',
    'text-align': 'center'
}

DIV_STYLE_RIGHT = {
    'display': 'block',
    'marginLeft': '1em',
    'marginRight': '0em',
    'width': '45%',
    'height': 'auto',
    'float': 'right',
    'text-align': 'center'
}

DIV_STYLE_PARENT = {
    'overflow': 'overlay'
}

image_style = {'width': '70%', 'height': '70%', 'margin': 'auto', 'display': 'block'}
text_style = dict(**image_style, **{'text-align': 'center'})
style_header = {'backgroundColor': '#0D6EFD', 'color': 'white'}
style_data = {'whiteSpace': 'normal', 'height': 'auto', 'width': 'auto', 'color': 'black'}
font = {'family': 'verdana', 'size': 14, 'color': 'black'}

HOST = config['FILE_SERVER_HOST_DASH']
PORT = config['FILE_SERVER_PORT']


def build_table(df, id_, data=None, column_id='TESTING_EXAMPLE', width_perc=None):
    """
    Builds a data table
    :param df: data to display
    :param id_: id of component
    :param data: optional processed data
    :param column_id: column for special formatting
    :param width_perc: width of column with special formatting
    :return: table component
    """
    if data is None:
        data = df.to_dict('records')
    cols = [{'name': i, 'id': i} for i in df.columns]

    for col in cols:
        df[col['id']] = df[col['id']].astype(str)
    return dash_table.DataTable(
        data=data,
        columns=cols,
        id=id_,
        style_header=style_header,
        style_data=style_data,
        style_cell_conditional=[
            {
                'if': {'column_id': column_id},
                'width': '40%' if width_perc is None else width_perc
            },
        ]
    )


def open_json(filename):
    """
    Opens JSON file for reading
    :param filename: name of file to open
    :return: open file
    """
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()


def plot_cfm(confusion_matrix):
    """
    Plots confusion matrix for Dash
    :param confusion_matrix: array to visualise
    :return: decoded image
    """
    group_names = ['TN', 'FP', 'FN', 'TP']

    group_counts = ["{0:0.0f}".format(value) for value in confusion_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in confusion_matrix.flatten() / np.sum(confusion_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]

    labels = np.asarray(labels).reshape(2, 2)

    heatmap = sns.heatmap(confusion_matrix, annot=labels, fmt='', cmap='Blues')

    figure = heatmap.get_figure()
    with io.BytesIO() as bts:
        figure.savefig(bts, format='png', dpi=figure.dpi)
        bts.seek(0)
        b64 = base64.b64encode(bts.read())
        plt.clf()
    return f'data:image/png;base64,{b64.decode()}'


def build_dropdown(label, idx, cols, value=None):
    """
    Builds dropdown menu with label
    :param label: label of menu
    :param idx: id of component
    :param cols: options for dropdown
    :param value: default option
    :return: dropdown component
    """
    return html.Div(children=[
        dbc.Label(label),
        dcc.Dropdown(
            id=idx,
            options=[{'label': col, 'value': col} for col in cols],
            value=value
        )
    ])
