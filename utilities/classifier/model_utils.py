import gc
import math
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from tqdm.auto import tqdm
from nlp_model import NLPModel
from IPython.core.display_functions import display
from utilities.data_processing.preprocessing import PreProcessing
from utilities.utils import write_json, output_dir, read_json, split_by_length, hashtags_dir, \
    plot_confusion_matrix, plot_losses, plot_lrs, save_predictions, visualise_cf_samples, input_dir, final_model_dir, \
    classifier_tuning_dir

default_model = read_json(f'{input_dir}/default_model_hyperparameters.json')


def update_key(key, x):
    key['lr_start'] = x
    key['lr_end'] = x / 100
    return key


# noinspection PyTypeChecker
def _graph_train(k, xs, data, tedf, x_key, device, f=None, disable_tqdm=False, process_fn=None):
    """
    Helper function for training across multiple parameters.
    :param k: value of k for k-fold cross validation
    :param xs: parameters to run
    :param data: data for training and validation
    :param tedf: testing data
    :param x_key: x label
    :param device: device to run model on
    :param f: function for updating default key
    :param disable_tqdm: whether to disable tqdm
    :return: results, keys, values
    """

    def default_update(d_key, d_x):
        d_key[x_key] = d_x
        return d_key

    keys = []
    values = []
    results = []
    for i, x in tqdm(enumerate(xs), total=len(xs), disable=disable_tqdm):
        key = default_model.copy()
        key = f(key, x) if f is not None else default_update(key, x)

        value = run_cross_validation(
            k=k,
            data=data,
            tedf=tedf,
            device=device,
            key=key,
            to_display=False,
            process_fn=process_fn
        )

        results.append({
            x_key: x,
            'accuracy': value['test_report']['accuracy'],
            'negative_f1': value['test_report']['negative']['f1-score'],
            'positive_f1': value['test_report']['positive']['f1-score'],
            'cohen\'s_kappa': value['test_report']['kappa'],
            'eval_accuracy': value['eval_report']['accuracy'],
            'eval_negative_f1': value['eval_report']['negative']['f1-score'],
            'eval_positive_f1': value['eval_report']['positive']['f1-score'],
            'eval_cohen\'s_kappa': value['eval_report']['kappa'],
            'short_eval_accuracy': value['short_eval_report']['accuracy'],
            'short_eval_negative_f1': value['short_eval_report']['negative']['f1-score'],
            'short_eval_positive_f1': value['short_eval_report']['positive']['f1-score'],
            'short_eval_cohen\'s_kappa': value['short_eval_report']['kappa'],
            'long_eval_accuracy': value['long_eval_report']['accuracy'],
            'long_eval_negative_f1': value['long_eval_report']['negative']['f1-score'],
            'long_eval_positive_f1': value['long_eval_report']['positive']['f1-score'],
            'long_eval_cohen\'s_kappa': value['long_eval_report']['kappa'],
            'short_accuracy': value['short_report']['accuracy'],
            'short_negative_f1': value['short_report']['negative']['f1-score'],
            'short_positive_f1': value['short_report']['positive']['f1-score'],
            'short_cohen\'s_kappa': value['short_report']['kappa'],
            'long_accuracy': value['long_report']['accuracy'],
            'long_negative_f1': value['long_report']['negative']['f1-score'],
            'long_positive_f1': value['long_report']['positive']['f1-score'],
            'long_cohen\'s_kappa': value['long_report']['kappa'],
        })

        keys.append(key)
        values.append(value)
    return results, keys, values


# noinspection DuplicatedCode
def display_cross_validation(
        original_df: pd.DataFrame,
        result,
        label,
        cv_filename=None,
        cfm_filename=None,
        loss_filename=None,
        lr_filename=None,
        predictions_filename=None,
        display_training=False
):
    odf = original_df.copy()
    df = pd.DataFrame(result[label]['report'])

    display(df.head())

    fig = px.line(df, x='id', y=['accuracy', 'negative_f1', 'positive_f1', 'cohen\'s_kappa'])

    fig.update_layout(template='plotly', title='Negative F1-Score')

    if cv_filename is not None:
        fig.write_json(f'{final_model_dir}/{cv_filename}.json')
        fig.write_html(f'{final_model_dir}/{cv_filename}.html')

    fig.show()

    for k in ['accuracy', 'negative_f1', 'positive_f1', 'cohen\'s_kappa']:
        print(f'Average {k}: {np.mean([e[k] for e in result[label]["report"]])}')

    plot_confusion_matrix(np.mean(np.array(result[label]['cfm']), axis=0), filename=cfm_filename)

    predictions = [x for xs in result[label]['predictions'] for x in xs]
    if predictions_filename is not None:
        save_predictions(predictions, path=predictions_filename)

    odf = odf.head(len(predictions)).copy()
    odf['prediction'] = predictions

    visualise_cf_samples(odf)

    if loss_filename is not None and display_training:
        plot_losses(np.mean(np.array(result[label]['losses']), axis=0), filename=loss_filename, override_filename=True)
    if lr_filename is not None and display_training:
        plot_lrs(np.mean(np.array(result[label]['lrs']), axis=0), filename=lr_filename, override_filename=True)


# noinspection DuplicatedCode
def run_cross_validation(
        k,
        data,
        tedf,
        device,
        key=None,
        to_display=False,
        process_fn=None,
        cache=None
):
    if cache is not None:
        return cache

    if key is None:
        key = default_model

    result = {label: {'predictions': [], 'losses': [], 'lrs': [], 'cfm': [], 'report': []} for label in
              ['eval', 'test', 'short_eval', 'long_eval', 'short_test', 'long_test']}

    for i in tqdm(range(k)):
        result[i] = {'eval': {}, 'short_eval': {}, 'long_eval': {}, 'test': {}, 'short_test': {}, 'long_test': {}}
        train = pd.concat(
            [data.head(i * math.floor(len(data) / k)), data.tail(len(data) - (i * math.floor(len(data) / k)))])
        validate = data.tail(len(data) - (i * math.floor(len(data) / k))).head(math.floor(len(data) / k)).copy()
        test = tedf.tail(len(tedf) - (i * math.floor(len(tedf) / k))).head(math.floor(len(tedf) / k)).copy()

        if process_fn is None:
            preprocessing = PreProcessing(train, silent=True).augment_dataset(n=2, reset_index=False)
            train = preprocessing.df
        else:
            train = process_fn(train).df

        svdf_eval, lvdf_eval = split_by_length(validate)
        svdf, lvdf = split_by_length(test)

        special = ['roberta-large', 'microsoft/deberta-base']

        model_ = NLPModel(
            training_data=train,
            validation_data=validate,
            device=device,
            use_downsampling=key['downsample'],
            batch_size=key['batch_size'] if key['model_name'] not in special else 4,
            gradient_accumulation_steps=key['GAS'] if key['model_name'] not in special else 8,
            epochs=key['epochs'],
            scheduler_type=key['scheduler_type'],
            model_name=key['model_name'],
            learning_rate=key['lr_start'],
            learning_rate_end=key['lr_end']
        )

        model_.train(log_level='critical')

        model_.eval(to_display=False)

        for label, dataset in [('eval', validate), ('test', test), ('short_eval', svdf_eval), ('long_eval', lvdf_eval),
                               ('short_test', svdf), ('long_test', lvdf)]:
            model_.test(dataset, visualise_df=False, to_display=False)

            result[label]['predictions'].append(model_.predictions)
            result[label]['losses'].append(model_.get_losses())
            result[label]['lrs'].append(model_.lrs)
            result[label]['cfm'].append(model_.confusion_matrix)
            result[label]['report'].append({
                'id': i,
                'accuracy': model_.report['accuracy'],
                'negative_f1': model_.report['negative']['f1-score'],
                'positive_f1': model_.report['positive']['f1-score'],
                'cohen\'s_kappa': model_.report['kappa'][0]
            })

        del model_
        gc.collect()
        torch.cuda.empty_cache()

    if to_display:
        write_json(result, f'{final_model_dir}/cross_validation_results.json')
        return result
    else:
        return {
            'eval_report': {
                'accuracy': np.mean([e['accuracy'] for e in result['eval']['report']]),
                'negative': {'f1-score': np.mean([e['negative_f1'] for e in result['eval']['report']])},
                'positive': {'f1-score': np.mean([e['positive_f1'] for e in result['eval']['report']])},
                'kappa': np.mean([e['cohen\'s_kappa'] for e in result['eval']['report']])
            },
            'short_eval_report': {
                'accuracy': np.mean([e['accuracy'] for e in result['short_eval']['report']]),
                'negative': {'f1-score': np.mean([e['negative_f1'] for e in result['short_eval']['report']])},
                'positive': {'f1-score': np.mean([e['positive_f1'] for e in result['short_eval']['report']])},
                'kappa': np.mean([e['cohen\'s_kappa'] for e in result['short_eval']['report']])
            },
            'long_eval_report': {
                'accuracy': np.mean([e['accuracy'] for e in result['long_eval']['report']]),
                'negative': {'f1-score': np.mean([e['negative_f1'] for e in result['long_eval']['report']])},
                'positive': {'f1-score': np.mean([e['positive_f1'] for e in result['long_eval']['report']])},
                'kappa': np.mean([e['cohen\'s_kappa'] for e in result['long_eval']['report']])
            },
            'test_report': {
                'accuracy': np.mean([t['accuracy'] for t in result['test']['report']]),
                'negative': {'f1-score': np.mean([t['negative_f1'] for t in result['test']['report']])},
                'positive': {'f1-score': np.mean([t['positive_f1'] for t in result['test']['report']])},
                'kappa': np.mean([t['cohen\'s_kappa'] for t in result['test']['report']])
            },
            'short_report': {
                'accuracy': np.mean([t['accuracy'] for t in result['short_test']['report']]),
                'negative': {'f1-score': np.mean([t['negative_f1'] for t in result['short_test']['report']])},
                'positive': {'f1-score': np.mean([t['positive_f1'] for t in result['short_test']['report']])},
                'kappa': np.mean([t['cohen\'s_kappa'] for t in result['short_test']['report']])
            },
            'long_report': {
                'accuracy': np.mean([t['accuracy'] for t in result['long_test']['report']]),
                'negative': {'f1-score': np.mean([t['negative_f1'] for t in result['long_test']['report']])},
                'positive': {'f1-score': np.mean([t['positive_f1'] for t in result['long_test']['report']])},
                'kappa': np.mean([t['cohen\'s_kappa'] for t in result['long_test']['report']])
            }
        }


# noinspection DuplicatedCode
def run_graph(
        k,
        xs,
        data,
        tedf,
        x_key,
        x_title,
        log_x,
        device,
        update_key_=None,
        df=None,
        show_all=True
):
    """
    Graphs performance for hyperparameter tuning.
    :param k: value of k for k-fold cross validation
    :param xs: baseline x data
    :param data: data for training and testing
    :param tedf: testing data
    :param x_key: x label
    :param x_title: title of x-axis
    :param log_x: whether to use a logarithmic x-axis
    :param device: device to run model on
    :param update_key_: function for updating default key with new value
    :param df: cached results for faster graphing
    :param show_all: whether to show all graphs
    :return:
    """
    if df is None:
        results, keys, values = _graph_train(k=k, xs=xs, data=data, tedf=tedf, x_key=x_key, device=device,
                                             f=update_key_)

        write_json(keys, filename=f'{classifier_tuning_dir}/{x_key}/{x_key}_keys.json')
        write_json(values, filename=f'{classifier_tuning_dir}/{x_key}/{x_key}_values.json')

        all_keys: dict = read_json(filename=f'{output_dir}/config/keys.json')
        all_values: dict = read_json(filename=f'{output_dir}/config/values.json')
        offset = len(all_keys)

        duplicate_indices = [i for i, k in enumerate(keys) if k not in all_keys.values()]

        all_keys.update({offset + i: k for i, k in enumerate(keys) if i not in duplicate_indices})
        all_values.update({offset + i: v for i, v in enumerate(values) if i not in duplicate_indices})

        write_json(all_keys, filename=f'{output_dir}/config/keys.json')
        write_json(all_values, filename=f'{output_dir}/config/values.json')

        df = pd.DataFrame(results)
        df.to_pickle(f'{classifier_tuning_dir}/{x_key}/{x_key}_results.pickle')

    xs = list(range(len(df)))
    cur_labels = ['accuracy', 'negative_f1', 'positive_f1', 'cohen\'s_kappa']

    _display_fig(
        df=df,
        xs=xs,
        x_key=x_key,
        file_key=f'{x_key}/{x_key}_graph',
        x_title=x_title,
        title=f'Performance of {x_title} on All Testing Data',
        log_x=log_x,
        show=show_all
    )

    graphs = ['short', 'long', 'eval', 'short_eval', 'long_eval']
    labels = [[f'{x}_{label}' for label in cur_labels] for x in graphs]
    file_keys = [f'{x_key}/{x_key}_{x}_graph' for x in graphs]
    titles = [f'Performance of {x_title} on {x}' for x in
              ['Short Testing Data', 'Long Testing Data', 'All Evaluation Data', 'Short Evaluation Data',
               'Long Evaluation Data']]

    for i in range(len(graphs)):
        df = _update_df(df, cur_labels, labels[i])
        _display_fig(
            df=df,
            xs=xs,
            x_key=x_key,
            file_key=file_keys[i],
            x_title=x_title,
            title=titles[i],
            log_x=log_x,
            show=True if graphs[i] == 'eval' else show_all
        )


# noinspection DuplicatedCode
def augmentation_graph(
        k,
        augmentation,
        data,
        tedf,
        device,
        df=None,
        show_all=True
):
    """
    Graphs performance of data augmentation.
    :param k: value of k for k-fold cross validation
    :param augmentation: augmentation values to test
    :param data: data for training and validation
    :param tedf: testing data
    :param device: device to run model on
    :param df: cached results for faster graphing
    :param show_all: whether to show all graphs
    :return:
    """
    x_key = 'augmentation'
    x_title = 'Augmentation'

    if df is None:
        keys = []
        values = []
        results = []
        for i, x in tqdm(enumerate(augmentation), total=len(augmentation)):
            r, key, v = _graph_train(
                k=k,
                xs=[i],
                data=data,
                tedf=tedf,
                x_key='augmentation',
                device=device,
                disable_tqdm=True,
                process_fn=lambda trdf: PreProcessing(trdf, silent=True).augment_dataset(n=x, reset_index=False)
            )
            results.append(r[0])
            keys.append(key[0])
            values.append(v[0])

        write_json(keys, filename=f'{classifier_tuning_dir}/{x_key}/{x_key}_keys.json')
        write_json(values, filename=f'{classifier_tuning_dir}/{x_key}/{x_key}_values.json')

        df = pd.DataFrame(results)
        df.to_pickle(f'{classifier_tuning_dir}/{x_key}/{x_key}_results.pickle')

    xs = list(range(len(df)))
    cur_labels = ['accuracy', 'negative_f1', 'positive_f1', 'cohen\'s_kappa']

    _display_fig(
        df=df,
        xs=xs,
        x_key='augmentation',
        file_key=f'{x_key}/{x_key}_graph',
        x_title=x_title,
        title=f'Performance of {x_title} on All Testing Data',
        log_x=False,
        show=show_all
    )

    graphs = ['short', 'long', 'eval', 'short_eval', 'long_eval']
    labels = [[f'{x}_{label}' for label in cur_labels] for x in graphs]
    file_keys = [f'{x_key}/{x_key}_{x}_graph' for x in graphs]
    titles = [f'Performance of {x_title} on {x}' for x in
              ['Short Testing Data', 'Long Testing Data', 'All Evaluation Data', 'Short Evaluation Data',
               'Long Evaluation Data']]

    for i in range(len(graphs)):
        df = _update_df(df, cur_labels, labels[i])
        _display_fig(
            df=df,
            xs=xs,
            x_key=x_key,
            file_key=file_keys[i],
            x_title=x_title,
            title=titles[i],
            log_x=False,
            show=True if graphs[i] == 'eval' else show_all
        )


# noinspection DuplicatedCode
def baseline_graph(
        k,
        data,
        tedf,
        device,
        size=11
):
    """
    Gets baseline for comparison with no pre-processing.
    :param k: value of k for k-fold cross validation
    :param data: data for training and validation
    :param tedf: testing data
    :param device: device to run model on
    :param size: number of repetitions for building graph from baseline result
    :return: baseline data
    """
    result = _graph_train(k=k, xs=[0], data=data, tedf=tedf, x_key='baseline', device=device,
                          disable_tqdm=True)[0][0]

    df = pd.DataFrame([result for _ in range(size)])
    df.to_pickle(f'{classifier_tuning_dir}/baseline/baseline_results.pickle')

    return df


def _build_preprocessors(data, testing):
    """
    Builds pre-processors for use in graphing.
    :param data: training and validation data
    :param testing: testing data
    :return: pre-processed data
    """
    def get_hashtag_cache(use_frequency, i_):
        """
        Gets the corresponding cache for the hashtags
        :param use_frequency: whether to optimise for unigram frequency or word length
        :param i_: number in process. < 2 means 10_000 source, else 50_000 source
        :return: formatted cache filename
        """
        type_ = 'unigram' if use_frequency else 'length'
        n = '10_000' if i_ < 2 else '50_000'
        return f'{type_}_hashtags_{n}.json'

    preprocessors = [
        lambda x: x.convert_html_entities(),
        lambda x: x.emojis(),
        lambda x: x.strip_emojis(),
        lambda x: x.strip_mentions(),
        lambda x: x.strip_hashtags(),
        lambda x: x.strip_newlines(),
        lambda x: x.strip_links()
    ]

    hashtags_dirs = [
        f'{hashtags_dir}/10_000_words.txt',
        f'{hashtags_dir}/50_000_words.txt',
    ]

    dfs = []
    raw_dfs = [data, testing]

    with tqdm(total=(len(preprocessors) * len(raw_dfs)) + (len(hashtags_dirs) * len(raw_dfs) * 2)) as tq:
        for pp in preprocessors:
            for df in raw_dfs:
                dfs.append(pp(PreProcessing(df, silent=True)).df)
                tq.update(1)

        i = 0
        for word_source in hashtags_dirs:
            for use_freq in [True, False]:
                for df in raw_dfs:
                    pp = PreProcessing(df=df, word_source=word_source, silent=True).contextualise_hashtags(
                        cache_source=f'{hashtags_dir}/{get_hashtag_cache(use_freq, i)}', use_frequencies=use_freq)
                    dfs.append(pp.df)
                    del pp
                    tq.update(1)
                i += 1
    del raw_dfs, preprocessors
    print(len(dfs))
    return dfs


def _calculate_y_range(df):
    max_val = max(df['accuracy'].max(), df['negative_f1'].max(), df['positive_f1'].max(), df['cohen\'s_kappa'].max())
    min_val = min(df['accuracy'].min(), df['negative_f1'].min(), df['positive_f1'].min(), df['cohen\'s_kappa'].min())
    padding = abs(max_val - min_val) * 0.1
    return min_val - padding, max_val + padding


def _display_fig(df, xs, x_key, file_key, log_x, x_title, title, baseline=None, baseline_labels=None,
                 show=True):
    """
    Helper function for displaying figure.
    :param df: data
    :param xs: baseline plot xs values
    :param x_key: key for x-axis labels
    :param file_key: key for saving to file
    :param log_x: whether to use a logarithmic x-axis
    :param x_title: title for x-axis
    :param title: title for graph
    :param baseline: baseline statistics for comparison
    :param baseline_labels: labels for building baseline statistics
    :param show: whether to show graph
    :return:
    """
    if (baseline is None and baseline_labels is not None) or (baseline is not None and baseline_labels is None):
        raise ValueError('baseline and its labels must both be null or initialised')

    fig = px.line(df, x=x_key, y=['accuracy', 'negative_f1', 'positive_f1', 'cohen\'s_kappa'], log_x=log_x,
                  template='plotly')

    y_range = _calculate_y_range(df)

    if baseline is not None:
        fig.add_trace(go.Scatter(x=xs, y=baseline[baseline_labels[0]].tolist(), name='accuracy_baseline',
                                 line={'color': 'blue', 'dash': 'dash'}))
        fig.add_trace(go.Scatter(x=xs, y=baseline[baseline_labels[1]].tolist(), name='negative_f1_baseline',
                                 line={'color': 'red', 'dash': 'dash'}))
        fig.add_trace(go.Scatter(x=xs, y=baseline[baseline_labels[2]].tolist(), name='positive_f1_baseline',
                                 line={'color': 'green', 'dash': 'dash'}))
        fig.add_trace(go.Scatter(x=xs, y=baseline[baseline_labels[3]].tolist(), name='cohen\'s_kappa',
                                 line={'color': 'purple', 'dash': 'dash'}))
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title='Performance',
        template='plotly',
        font={'family': 'verdana', 'size': 26, 'color': 'black'}
    )
    fig.update_yaxes(range=y_range)
    fig.write_json(f'{classifier_tuning_dir}/{file_key}.json')
    fig.write_html(f'{classifier_tuning_dir}/{file_key}.html')
    if show:
        fig.show()


def _update_df(df: pd.DataFrame, cur_labels, new_labels):
    """
    Updates the labels of the data for the next graph to be built.
    :param df: data for updating
    :param cur_labels: current labels used in graph
    :param new_labels: labels needed to change for next graph
    :return: data
    """
    df.drop(cur_labels, axis=1, inplace=True)
    df.rename(columns={new_labels[i]: cur_label for i, cur_label in enumerate(cur_labels)}, inplace=True)
    return df


# noinspection DuplicatedCode,PyTypeChecker
def preprocessing_graph(
        k,
        data,
        testing,
        device,
        df=None,
        baseline: pd.DataFrame = None,
        show_all=True
):
    """
    Graphs performance of pre-processing tasks
    :param k: value of k for k-fold cross validation
    :param data: training and validation data
    :param testing: testing data
    :param device: device to run model on
    :param df: cached results for faster graphing
    :param baseline: baseline for comparison using no pre-processing
    :param show_all: whether to show all graphs
    :return:
    """
    x_key = 'pre-processing'
    x_title = 'Pre-Processing Task'
    if df is None:
        dfs = _build_preprocessors(data, testing)
        keys = []
        values = []
        results = []
        for i in tqdm(range(int(len(dfs) / 2))):
            r, key, v = _graph_train(k=k, xs=[i], data=dfs[i * 2], tedf=dfs[(i * 2) + 1],
                                     x_key=x_key, device=device, disable_tqdm=True,
                                     process_fn=lambda x: PreProcessing(x, silent=True).augment_dataset(n=2))
            results.append(r[0])
            keys.append(key[0])
            values.append(v[0])

        write_json(keys, filename=f'{classifier_tuning_dir}/preprocessing/{x_key}_keys.json')
        write_json(values, filename=f'{classifier_tuning_dir}/preprocessing/{x_key}_values.json')

        df = pd.DataFrame(results)
        df.to_pickle(f'{classifier_tuning_dir}/preprocessing/preprocessing_results.pickle')

    if baseline is None:
        baseline = baseline_graph(
            k=k,
            data=data,
            tedf=testing,
            device=device
        )

    xs = list(range(len(df)))
    cur_labels = ['accuracy', 'negative_f1', 'positive_f1', 'cohen\'s_kappa']

    _display_fig(
        df=df,
        xs=xs,
        x_key=x_key,
        file_key=f'preprocessing/{x_key}_graph',
        x_title=x_title,
        title=f'Performance of {x_title} on All Testing Data',
        baseline=baseline,
        baseline_labels=['accuracy', 'negative_f1', 'positive_f1', 'cohen\'s_kappa'],
        log_x=False,
        show=show_all
    )

    graphs = ['short', 'long', 'eval', 'short_eval', 'long_eval']
    labels = [[f'{x}_{label}' for label in cur_labels] for x in graphs]
    file_keys = [f'preprocessing/{x_key}_{x}_graph' for x in graphs]
    titles = [f'Performance of {x_title} on {x}' for x in
              ['Short Testing Data', 'Long Testing Data', 'All Evaluation Data', 'Short Evaluation Data',
               'Long Evaluation Data']]

    for i in range(len(graphs)):
        df = _update_df(df, cur_labels, labels[i])
        _display_fig(
            df=df,
            xs=xs,
            x_key=x_key,
            file_key=file_keys[i],
            x_title=x_title,
            title=titles[i],
            baseline=baseline,
            baseline_labels=labels[i],
            log_x=False,
            show=True if graphs[i] == 'eval' else show_all
        )
