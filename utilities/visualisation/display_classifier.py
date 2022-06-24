import dash_bootstrap_components as dbc

from dash import Input, Output, html, callback
import utilities.visualisation.classifier_results as cr
import utilities.visualisation.classifier_overview as co

layout = html.Div(id='classifier_container', children=[
    dbc.Tabs(
        [
            dbc.Tab(label="Overview", tab_id="overview"),
            dbc.Tab(label="Model Results", tab_id="model_results"),
        ],
        id="classifier_display_options",
        active_tab="overview",
    ),
    html.Br(),
    html.Div(id='classifier_main_container', children=[])
])


@callback(
    Output(component_id='classifier_main_container', component_property='children'),
    Input(component_id='classifier_display_options', component_property='active_tab')
)
def build_main_classifier(display_option):
    match display_option:
        case 'overview':
            return co.layout
        case 'model_results':
            return cr.layout
        case _:
            raise ValueError(f'Unknown display options: {display_option}')
