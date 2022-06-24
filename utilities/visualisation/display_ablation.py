import pandas as pd

from dash import dcc, html
from plotly.io import from_json
from utilities.utils import read_json
from utilities.visualisation.vis_utils import open_json, HOST, PORT, font, build_table

ID_PREFIX = 'da'

file_config = {
    'ablation_reports': './output/figures/ablation/ablating_%_report.json',
    'location_cfm': 'ablation/ablating_user_cfm.png',
    'coordinates_cfm': 'ablation/ablating_coordinates_cfm.png',
    'entities_cfm': 'ablation/ablating_entity_cfm.png',
    'ablation_graph': './output/figures/ablation/ablation_graph.json'
}


def build_ablation_results():
    reports = [(key, read_json(file_config['ablation_reports'].replace('%', key))) for key in
               ['user', 'coordinates', 'entity']]
    reports = [{'key': key, 'accuracy': value['accuracy'], 'negative_f1': value['negative']['f1-score'],
                'positive_f1': value['positive']['f1-score']} for key, value in reports]

    df = pd.DataFrame(reports).round(3)

    return html.Div(id=f'{ID_PREFIX}_ablation_results', children=[
        build_table(
            df=df,
            id_='ablation_results_table'
        ),
        build_cfs()
    ])


def build_cfs():
    return html.Div(className='row_', children=[
        html.Div(className='column_', children=[
            html.H6(f'Effect of ablating user location source'),
            html.Img(src=f'{HOST}:{PORT}/{file_config["location_cfm"]}', style={'width': '100%'})
        ]),
        html.Div(className='column_', children=[
            html.H6(f'Effect of ablating coordinate location source'),
            html.Img(src=f'{HOST}:{PORT}/{file_config["coordinates_cfm"]}', style={'width': '100%'})
        ]),
        html.Div(className='column_', children=[
            html.H6(f'Effect of ablating entity location source'),
            html.Img(src=f'{HOST}:{PORT}/{file_config["entities_cfm"]}', style={'width': '100%'})
        ])
    ])


def build_ablation_graph():
    fig = from_json(open_json(filename=file_config['ablation_graph']))
    fig.update_layout(font=font)
    return dcc.Graph(figure=fig)


layout = html.Div(id='processing_container', children=[
    build_ablation_results(),
    build_ablation_graph(),
])
