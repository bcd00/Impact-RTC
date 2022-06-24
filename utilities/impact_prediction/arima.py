import itertools
import pandas as pd
import plotly.graph_objects as go

from tqdm.auto import tqdm
from datetime import timedelta, datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from IPython.core.display_functions import display
from utilities.utils import write_json, read_json, arima_dir


# noinspection DuplicatedCode
class ArimaModel:
    def __init__(self, data):
        self.is_stationary(data)
        self.averages = {label: group['size'].mean() for label, group in data.groupby(['weekday', 'time_of_day'])}
        self.x_data, self.df = self.format_data(data.copy(), self.averages)
        self.fitted_model = None

    # noinspection PyUnresolvedReferences
    @staticmethod
    def is_stationary(data):
        rolling_mean = data['size'].rolling(window=12).mean().to_frame()
        rolling_std = data['size'].rolling(window=12).std().to_frame()

        dt_all = pd.date_range(start=data['local_time'].iloc[0], end=data['local_time'].iloc[-1])
        dt_obs = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(data['local_time'])]
        dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if d not in dt_obs]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=data.index.tolist(), y=data['size'].tolist(), name='Original', mode='lines'))
        fig.add_trace(go.Scatter(x=rolling_mean.index.tolist(), y=rolling_mean['size'].tolist(), name='Rolling Mean',
                                 mode='lines'))
        fig.add_trace(
            go.Scatter(x=rolling_std.index.tolist(), y=rolling_std['size'].tolist(), name='Rolling STD', mode='lines'))
        fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
        fig.update_layout(template='plotly')

        fig.write_json(f'{arima_dir}/stationary.json')
        fig.write_html(f'{arima_dir}/stationary.html')

        fig.show()

        result = adfuller(data['size'])
        print('ADF Statistic: {}'.format(result[0]))
        print('p-value: {}'.format(result[1]))
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t{}: {}'.format(key, value))

    @staticmethod
    def _get_times(data):
        groups = data.groupby([data.local_time.dt.date, data.local_time.dt.hour])
        groups = tuple(zip(*[
            (datetime.combine(label[0], datetime.min.time()) + timedelta(hours=label[1].tolist()), group['size'].mean())
            for label, group in groups]))
        return pd.DataFrame({'time': groups[0], 'size': groups[1]}).set_index('time', drop=False)

    @staticmethod
    def _get_missing_times(averages, times_df, default=None):
        dt_all = pd.date_range(start=times_df['time'].iloc[0], end=times_df['time'].iloc[-1], freq='H')
        dt_obs = [d.strftime("%Y-%m-%d %H:%M:%S") for d in pd.to_datetime(times_df['time'])]
        dt_breaks = [datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in dt_all.strftime("%Y-%m-%d %H:%M:%S").tolist() if
                     d not in dt_obs]
        alt_times_df = pd.DataFrame(dt_breaks, columns=['time'])

        alt_times_df['size'] = alt_times_df['time'].apply(
            lambda x: averages.get((x.weekday(), x.hour), 0) if default is None else default)
        return alt_times_df.set_index('time', drop=False)

    def format_data(self, data, averages, default=None):
        data['local_time'] = pd.to_datetime(data['local_time'])
        x = self._get_times(data)
        x_comp = self._get_missing_times(averages, x, default)
        x_data = pd.concat([x, x_comp]).sort_index()
        return x_data, x_data['size'].copy().asfreq(f'H')

    def tune(self, training_df, validation_df, ds, ps, qs, cluster_type, do_grid_search=False):
        def display_tuning(test_vs, baseline_vs, label):
            fig = go.Figure()
            xs = [i for i in range(len(test_vs))]

            fig.add_trace(go.Scatter(x=xs, y=test_vs, mode='lines'))
            fig.add_trace(go.Scatter(x=xs, y=baseline_vs, mode='lines'))

            fig.update_layout(template='plotly')

            fig.write_json(f'{arima_dir}/tuning/tuning_{label}.json')
            fig.write_html(f'{arima_dir}/tuning/tuning_{label}.html')

            fig.show()

        test_mses = read_json(f'{arima_dir}/tuning/tuning_results.json')
        baseline_mses = read_json(f'{arima_dir}/tuning/tuning_baseline.json')

        test_mses = {'d': [], 'p': [], 'q': [], 'grid_search': []} if test_mses == {} else test_mses
        baseline_mses = {'d': [], 'p': [], 'q': [], 'grid_search': []} if baseline_mses == {} else baseline_mses

        if do_grid_search:
            for p, q in tqdm(list(itertools.product(ps, qs))):
                test_mse, baseline_mse = self.run(training_df, validation_df, order=(p, 0, q), to_display=False)
                test_mses['grid_search'].append({'p': p, 'q': q, 'mse': test_mse})
                baseline_mses['grid_search'].append({'p': p, 'q': q, 'mse': baseline_mse})

            display(pd.DataFrame(test_mses['grid_search']))
            display(pd.DataFrame(baseline_mses['grid_search']))
        else:
            display(pd.DataFrame(test_mses['grid_search']))
            display(pd.DataFrame(baseline_mses['grid_search']))

        for d in tqdm(ds):
            test_mse, baseline_mse = self.run(training_df, validation_df, order=(1, d, 4), to_display=False)
            test_mses['d'].append(test_mse)
            baseline_mses['d'].append(baseline_mse)

        for p in tqdm(ps):
            test_mse, baseline_mse = self.run(training_df, validation_df, order=(p, 0, 4), to_display=False)
            test_mses['p'].append(test_mse)
            baseline_mses['p'].append(baseline_mse)

        for q in tqdm(qs):
            test_mse, baseline_mse = self.run(training_df, validation_df, order=(1, 0, q), to_display=False)
            test_mses['q'].append(test_mse)
            baseline_mses['q'].append(baseline_mse)

        display_tuning(test_mses['p'], baseline_mses['p'], label='p')
        display_tuning(test_mses['d'], baseline_mses['d'], label='d')
        display_tuning(test_mses['q'], baseline_mses['q'], label='q')

        write_json(test_mses, f'{arima_dir}/tuning/tuning_results.json')
        write_json(baseline_mses, f'{arima_dir}/tuning/tuning_baseline.json')

    def run(self, training_df, validation_df, order=None, to_display=True, filename=None):
        if order is None:
            order = (1, 0, 4)

        trdf = training_df.copy()
        tvdf = validation_df.copy()
        history = [x for x in self.df]

        tvdf = self.format_data(tvdf, self.averages, default=-1)[1].to_frame()

        # Get test values
        test = [x for x in tvdf['size'].tolist()]

        # Get baseline values
        tvdf.reset_index(drop=False, inplace=True)
        tvdf['weekday'] = tvdf['time'].apply(lambda x: x.weekday())
        tvdf['time_of_day'] = tvdf['time'].apply(lambda x: x.hour)
        vs_ = {label: group['size'].mean() for label, group in trdf.groupby(['weekday', 'time_of_day'])}
        baseline = tvdf.apply(lambda x: vs_.get((x['weekday'], x['time_of_day']), 0), axis=1).tolist()

        predictions = []
        skipped = []
        for t in tqdm(range(len(test)), disable=not to_display):
            if test[t] == -1:
                skipped.append(t)
                continue
            model = ARIMA(history, order=order)
            model_fit = model.fit()
            predictions.append(model_fit.forecast()[0])
            history.append(test[t])

        test = [x for i, x in enumerate(test) if i not in skipped]
        baseline = [x for i, x in enumerate(baseline) if i not in skipped]

        test_mse = mean_squared_error(test, predictions)
        baseline_mse = mean_squared_error(test, baseline)

        if not to_display:
            return test_mse, baseline_mse

        print(f'Test MSE: {test_mse: .3f}')
        print(f'Baseline MSE: {baseline_mse: .3f}')

        xs = [i for i in range(len(test))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=test))
        fig.add_trace(go.Scatter(x=xs, y=predictions))
        fig.add_trace(go.Scatter(x=xs, y=baseline))

        fig.update_layout(template='plotly', font={'family': 'verdana', 'size': 26, 'color': 'black'})

        if filename is not None:
            fig.write_json(f'{arima_dir}/{filename}.json')
            fig.write_html(f'{arima_dir}/{filename}.html')

        fig.show()
