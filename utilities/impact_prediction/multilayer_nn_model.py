import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from IPython.core.display_functions import display
from sklearn.metrics import mean_squared_error

from tqdm.auto import tqdm
from torch.optim import SGD
from torch.utils.data import DataLoader
from utilities.utils import nn_model_dir
from density_dataset import DensityDataset
from sklearn.preprocessing import MinMaxScaler


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = nn.Linear(2, 4)
        self.b1 = nn.BatchNorm1d(4)
        self.l2 = nn.Linear(4, 2)
        self.b2 = nn.BatchNorm1d(2)
        self.l3 = nn.Linear(2, 1)

    @staticmethod
    def _swish(x):
        return x * torch.relu(x)

    # noinspection PyPep8Naming
    def forward(self, X):
        X = self._swish(self.l1(X))
        X = self.b1(X)
        X = self._swish(self.l2(X))
        X = self.b2(X)
        return torch.relu(self.l3(X))


class NNModel:
    def __init__(self, trdf, batch_size, epochs, learning_rate, device):
        self.trdf = trdf
        self.cols = ['weekday', 'time_of_day']
        self.scaler = MinMaxScaler()
        self.trdf[self.cols] = self.scaler.fit_transform(self.trdf[self.cols])
        display(self.trdf.head())
        self.training_data = DataLoader(dataset=DensityDataset(self.trdf, cols=self.cols, device=device),
                                        batch_size=batch_size, shuffle=False)
        self.batch_size = batch_size
        self.device = device
        self.network = Network().to(device)
        self.criterion = nn.MSELoss()
        self.epochs = epochs
        self.optimiser = SGD(self.network.parameters(), lr=learning_rate)

    @staticmethod
    def _train(model_, x, y, optimiser_, criterion_):
        model_.zero_grad()
        output = model_(x)
        loss = criterion_(output, y)
        loss.backward()
        optimiser_.step()

        return loss.detach().cpu().numpy().tolist(), output

    # noinspection DuplicatedCode
    def train(self):
        losses = []
        for _ in tqdm(range(self.epochs), total=self.epochs):
            epoch_loss = 0

            for bidx, batch in enumerate(self.training_data):
                x_train, y_train = batch['input'], batch['output']
                epoch_loss += self._train(self.network, x_train, y_train, self.optimiser, self.criterion)[0]

            losses.append(epoch_loss)

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=list(range(self.epochs)), y=losses, mode='lines'))

        fig.update_layout(template='plotly', font={'family': 'verdana', 'size': 26, 'color': 'black'},
                          xaxis_title='epochs', yaxis_title='loss')

        fig.write_json(f'{nn_model_dir}/nn_loss_graph.json')
        fig.write_html(f'{nn_model_dir}/nn_loss_graph.html')

        fig.show()

    def validate(self, validation):
        predictions = []
        df = validation.copy()
        df[self.cols] = self.scaler.fit_transform(df[self.cols])
        validation_data = DataLoader(
            dataset=DensityDataset(df, cols=['weekday', 'time_of_day'], device=self.device),
            batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for batch in validation_data:
                predictions.extend(self.network(batch['input']).squeeze().detach().cpu().numpy().tolist())

        vs = {label: group['size'].mean() for label, group in self.trdf.groupby(['weekday', 'time_of_day'])}
        df['baseline'] = df.apply(lambda x: vs.get((x['weekday'], x['time_of_day']), 0), axis=1)

        df['prediction'] = predictions
        print(f'MSE: {mean_squared_error(np.array(df["size"].tolist()), np.array(df["prediction"].tolist())): .2f}')
        print(
            f'Baseline MSE: {mean_squared_error(np.array(df["size"].tolist()), np.array(df["baseline"].tolist())): .2f}'
        )

        self.display_2d(df, cols=['weekday', 'time_of_day'], y='size', y_pred='prediction', y_baseline='baseline',
                        filename=f'{nn_model_dir}/nn_model_both')

        self.display_1d(df, x='weekday', y='size', y_pred='prediction', y_baseline='baseline',
                        filename=f'{nn_model_dir}/nn_model_weekday')
        self.display_1d(df, x='time_of_day', y='size', y_pred='prediction', y_baseline='baseline',
                        filename=f'{nn_model_dir}/nn_model_tod')
        return vs

    @staticmethod
    def display_2d(df, cols, y, y_pred, y_baseline, filename=None):
        fig = go.Figure(
            data=go.Scatter3d(x=df[cols[0]], y=df[y], z=df[cols[1]], mode='markers', marker={'size': 2}))
        fig.add_trace(go.Scatter3d(x=df[cols[0]], y=df[y_pred], z=df[cols[1]], mode='markers', marker={'size': 2}))
        fig.add_trace(go.Scatter3d(x=df[cols[0]], y=df[y_baseline], z=df[cols[1]], mode='markers', marker={'size': 2}))
        fig.update_layout(template='plotly')

        if filename is not None:
            fig.write_json(f'{filename}.json')
            fig.write_html(f'{filename}.html')

        fig.show()

    @staticmethod
    def display_1d(df, x, y, y_pred, y_baseline, filename=None):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[x], y=df[y], mode='markers', marker={'size': 2}))
        fig.add_trace(go.Scatter(x=df[x], y=df[y_pred], mode='markers', marker={'size': 2}))
        fig.add_trace(go.Scatter(x=df[x], y=df[y_baseline], mode='markers', marker={'size': 2}))
        fig.update_layout(
            template='plotly',
            # title='Model Results',
            yaxis_title='size',
            xaxis_title=x,
            font={'family': 'verdana', 'size': 26, 'color': 'black'}
        )

        if filename is not None:
            fig.write_json(f'{filename}.json')
            fig.write_html(f'{filename}.html')

        fig.show()

    def test(self, test):
        self.validate(test)
