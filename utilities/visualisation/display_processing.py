from dash import dcc, html
from plotly.io import from_json
from utilities.visualisation.vis_utils import DIV_STYLE_LEFT, DIV_STYLE_RIGHT, open_json, HOST, PORT, \
    DIV_STYLE_PARENT, font

ID_PREFIX = 'dp'

file_config = {
    'annotations': './output/figures/data_processing/annotations/annotations.json',
    'location': './output/figures/data_processing/location_type.json',
    'positive_word_cloud': 'data_processing/word_clouds/positive_wordcloud.png',
    'negative_word_cloud': 'data_processing/word_clouds/negative_wordcloud.png',
    'rules_overlay': './output/figures/data_processing/rules/rules.json',
    'rules_bihistogram': 'data_processing/rules/rules_bihistogram.png'
}


def build_first_layer():
    return html.Div(id=f'{ID_PREFIX}_layer_one', children=[
        html.H3('Distribution of the labelled data by class', style=DIV_STYLE_LEFT),
        html.H3('Distribution of specific vs user-provided location in the geospatial data', style=DIV_STYLE_RIGHT)
    ], style=DIV_STYLE_PARENT)


# noinspection DuplicatedCode
def build_second_layer():
    fig_one = from_json(open_json(filename=file_config['annotations']))
    fig_two = from_json(open_json(filename=file_config['location']))
    fig_one.update_layout(template='plotly', font=font)
    fig_two.update_layout(template='plotly', font=font)
    return html.Div(id=f'{ID_PREFIX}_layer_two', children=[
        dcc.Graph(figure=fig_one, style=DIV_STYLE_LEFT),
        dcc.Graph(figure=fig_two, style=DIV_STYLE_RIGHT)
    ], style=DIV_STYLE_PARENT)


def build_third_layer():
    return html.Div(id=f'{ID_PREFIX}_layer_three', children=[
        html.H3('Wordcloud for the positive class', style=DIV_STYLE_LEFT),
        html.H3('Wordcloud for the negative class', style=DIV_STYLE_RIGHT)
    ], style=DIV_STYLE_PARENT)


def build_fourth_layer():
    return html.Div(id=f'{ID_PREFIX}_layer_four', children=[
        html.Img(src=f'{HOST}:{PORT}/{file_config["positive_word_cloud"]}', style=DIV_STYLE_LEFT),
        html.Img(src=f'{HOST}:{PORT}/{file_config["negative_word_cloud"]}', style=DIV_STYLE_RIGHT)
    ], style=DIV_STYLE_PARENT)


def build_fifth_layer():
    return html.Div(id=f'{ID_PREFIX}_layer_five', children=[
        html.H3('Overlay of the rule distribution for raw vs geospatial data', style=DIV_STYLE_LEFT),
        html.H3('Bihistogram showing the rule distribution for raw vs geospatial data', style=DIV_STYLE_RIGHT)
    ], style=DIV_STYLE_PARENT)


def build_sixth_layer():
    fig = from_json(open_json(filename=file_config["rules_overlay"]))
    img_style = dict(DIV_STYLE_RIGHT, **{'marginTop': '55px'})
    fig.update_layout(template='plotly', font=font)

    return html.Div(id=f'{ID_PREFIX}_layer_six', children=[
        dcc.Graph(figure=fig, style=DIV_STYLE_LEFT),
        html.Img(src=f'{HOST}:{PORT}/{file_config["rules_bihistogram"]}', style=img_style)
    ], style=DIV_STYLE_PARENT)


def build_data_processing():
    return html.Div(id='processing_container', children=[
        build_first_layer(),
        html.Br(),
        build_second_layer(),
        html.Br(),
        build_third_layer(),
        html.Br(),
        build_fourth_layer(),
        html.Br(),
        build_fifth_layer(),
        html.Br(),
        build_sixth_layer(),
        html.Br()
    ])


layout = build_data_processing()
