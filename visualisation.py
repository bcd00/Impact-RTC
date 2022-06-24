import dash_bootstrap_components as dbc

from utilities.utils import config
from dash import html, Output, Input, dcc
from utilities.visualisation.app import app
from utilities.visualisation.display_interactive_models import layout
from utilities.visualisation.home_page import home_layout
from utilities.visualisation.vis_utils import SIDEBAR_STYLE, CONTENT_STYLE
from utilities.visualisation import display_ablation, display_classifier, display_processing, display_clustering, \
    display_impact_prediction
from utilities.visualisation.display_custom_clustering import layout as manual_layout
import utilities.visualisation.classifier_results as cr

sidebar = html.Div(
    [
        html.H2('RTC Impact', className='display-4'),
        html.Hr(),
        html.P('Select a page to view further information', className='lead'),
        dbc.Nav(
            [
                dbc.NavLink('Home', href='/', active='exact'),
                dbc.NavLink('Data Processing', href='/data_processing', active='exact'),
                dbc.NavLink('Tweet Classification', href='/classification', active='exact'),
                dbc.NavLink('Incident Clustering', href='/incident_clustering', active='exact'),
                dbc.NavLink('Impact Prediction', href='/impact_prediction', active='exact'),
                dbc.NavLink('Ablation Experiment', href='/ablation_experiment', active='exact'),
                dbc.NavLink('Interactive Model Pipeline', href='/model_pipeline', active='exact')
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id='page-content', style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id='url'), sidebar, content])


@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def render_page_content(pathname):
    match pathname:
        case '/':
            return home_layout
        case '/data_processing':
            return display_processing.layout
        case '/classification':
            return display_classifier.layout
        case '/incident_clustering':
            return display_clustering.layout
        case '/impact_prediction':
            return display_impact_prediction.layout
        case '/ablation_experiment':
            return display_ablation.layout
        case '/model_pipeline':
            return layout
        case '/ove_classifier_tuning':
            return html.Div(children=cr.layout, style={'height': '1080px', 'overflowY': 'auto'})
        case '/ove_clustering':
            return html.Div(children=manual_layout, style={'height': '1080px', 'overflowY': 'auto'})
        case '/ove_pipeline':
            return html.Div(children=layout, style={'height': '1080px', 'overflowY': 'auto'})
        case _:
            return [
                html.H1('404: Not found', className='text-danger'),
                html.Hr(),
                html.P(f'Unknown path: {pathname}'),
            ]


if __name__ == '__main__':
    app.run_server(
        host=config['DASH_HOST'],
        port=int(config['DASH_PORT']),
        debug=False,
        dev_tools_silence_routes_logging=False,
        dev_tools_ui=True,
        dev_tools_props_check=False
    )
