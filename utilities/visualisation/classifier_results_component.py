from dash import html


def display_report(title, report, style=None):
    if style is None:
        style = {}
    return html.Div(children=[
        html.H5(title),
        html.Ul(children=[
            html.Li(f'Accuracy: {report["accuracy"]}'),
            html.Li(f'Negative F1-Score: {report["negative"]["f1-score"]}'),
            html.Li(f'Positive F1-Score: {report["positive"]["f1-score"]}')
        ])
    ], style=style)


def display_model_results(container_id, eval_report, test_report, style):
    return html.Div(id=container_id, children=[
        display_report(title='Performance on Validation Dataset', report=eval_report),
        display_report(title='Performance on Testing Dataset', report=test_report)
    ], style=style)
