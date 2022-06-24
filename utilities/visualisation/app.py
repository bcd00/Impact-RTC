import diskcache
import dash_bootstrap_components as dbc

from dash import Dash
from utilities.utils import cache_dir
from dash.long_callback import DiskcacheLongCallbackManager

cache = diskcache.Cache(cache_dir)
long_callback_manager = DiskcacheLongCallbackManager(cache)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
