import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn import linear_model
from utilities.utils import linear_regression_dir
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score


# noinspection PyPep8Naming
class LRModel:
    def __init__(self, trdf, tvdf, tedf, averages, cols, label):
        self.trdf = trdf
        self.tvdf = tvdf
        self.tedf = tedf
        self.averages = averages
        self.cols = cols
        self.type = label
        self.model = linear_model.LinearRegression()

        scaler = MinMaxScaler()
        self.trdf[cols] = scaler.fit_transform(self.trdf[cols])
        self.tvdf[cols] = scaler.fit_transform(self.tvdf[cols])
        self.tedf[cols] = scaler.fit_transform(self.tedf[cols])

    def get_x(self, df: pd.DataFrame):
        return np.array(list(zip(*[df[col].tolist() for col in self.cols])))

    @staticmethod
    def get_y(df: pd.DataFrame):
        return np.array(df['size'].tolist())

    def train(self):
        X, y = self.get_x(self.trdf), self.get_y(self.trdf)
        self.model.fit(X, y)

    def eval(self):
        X, y = self.get_x(self.tvdf), self.get_y(self.tvdf)
        y_pred = self.model.predict(X)
        vs = {label: group['size'].mean() for label, group in self.trdf.groupby(self.cols)}
        if len(self.cols) > 1:
            baseline = self.tvdf.apply(lambda x: vs.get(tuple([x[col] for col in self.cols]), 0), axis=1).tolist()
        else:
            baseline = self.tvdf.apply(lambda x: vs.get(x[self.cols[0]], 0), axis=1).tolist()

        self._print_results(X, y, y_pred, baseline, label='eval')

    def test(self):
        X, y = self.get_x(self.tedf), self.get_y(self.tedf)
        y_pred = self.model.predict(X)

        vs = {label: group['size'].mean() for label, group in self.trdf.groupby(self.cols)}

        if len(self.cols) > 1:
            baseline = self.tedf.apply(lambda x: vs.get(tuple(x[col] for col in self.cols), 0), axis=1).tolist()
        else:
            baseline = self.tedf.apply(lambda x: vs.get(x[self.cols[0]], 0), axis=1).tolist()

        self._print_results(X, y, y_pred, baseline, label='test')

    def _print_results(self, X, y, y_pred, baseline, label):
        print(f'Coefficients: {self.model.coef_}')
        print(f'MSE: {mean_squared_error(y, y_pred): .2f}')
        print(f'Baseline MSE: {mean_squared_error(y, baseline): .2f}')
        print(f'Coefficient of determination: {r2_score(y, y_pred): .2f}')

        if len(self.cols) == 2:
            self._print_2d(X, y, y_pred, baseline, label)
        else:
            self._print_1d(X, y, y_pred, label)

    def _print_1d(self, X: np.ndarray, y, y_pred, label):
        unique = len(self.averages)
        tx = [x[0] * (unique - 1) for x in X]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tx, y=y, mode='markers', marker={'size': 2}))
        fig.add_trace(go.Scatter(x=tx, y=y_pred, mode='lines'))
        fig.add_trace(go.Scatter(x=[i for i in range(unique)], y=list(self.averages.values()), mode='lines'))
        fig.update_layout(
            template='plotly',
            title='Model Results',
            yaxis_title='Size',
            xaxis_title=self.cols[0],
            font={'family': 'verdana', 'size': 26, 'color': 'black'}
        )
        fig.write_json(f'{linear_regression_dir}/{label}_{self.cols[0]}.json')
        fig.write_html(f'{linear_regression_dir}/{label}_{self.cols[0]}.html')
        fig.show()

    def _print_2d(self, X: np.ndarray, y, y_pred, baseline, label):
        tx = X.swapaxes(0, 1)
        data = {col: tx[i] for i, col in enumerate(self.cols)}
        data['y'] = y
        data['y_pred'] = y_pred
        df = pd.DataFrame(data)
        fig = go.Figure(data=go.Scatter3d(
            x=df[self.cols[0]], y=df[self.cols[1]], z=y, mode='markers', marker={'size': 2}))
        fig.add_trace(
            go.Scatter3d(x=df[self.cols[0]], z=y_pred, y=df[self.cols[1]], mode='markers', marker={'size': 2}))
        fig.add_trace(
            go.Scatter3d(x=df[self.cols[0]], z=baseline, y=df[self.cols[1]], mode='markers', marker={'size': 2}))
        fig.update_layout(template='plotly', font={'family': 'verdana', 'size': 26, 'color': 'black'})

        fig.write_json(f'{linear_regression_dir}/{label}_{self.cols[0]}_{self.cols[1]}.json')
        fig.write_html(f'{linear_regression_dir}/{label}_{self.cols[0]}_{self.cols[1]}.html')

        fig.show()
