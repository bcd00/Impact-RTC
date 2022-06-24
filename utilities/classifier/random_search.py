import gc
import torch
import logging
import numpy as np
import pandas as pd

from itertools import product
from utilities.classifier.model_utils import run_cross_validation
from utilities.utils import read_json, output_dir, shared_dir, write_json, get_cuda_availability, config, input_dir


def setup():
    logging.basicConfig(level=logging.CRITICAL)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.CRITICAL)

    return get_cuda_availability()


def get_hyperparameter_combinations(keys_):
    hps = read_json(f'{input_dir}/hyperparameters.json')
    combinations_ = product(
        hps['initial_lrs'],
        hps['final_lrs'],
        hps['batch_sizes'],
        hps['model_types'],
        hps['epochs'],
        hps['scheduler_types'],
        hps['downsampling']
    )

    return [(lr_start, lr_end, batch_size, model_name, epoch, st, ds) for
            lr_start, lr_end, batch_size, model_name, epoch, st, ds in
            combinations_ if lr_start > lr_end and {'lr_start': lr_start, 'lr_end': lr_end, 'batch_size': batch_size,
                                                    'model_name': model_name, 'GAS': 1, 'epochs': epoch,
                                                    'scheduler_type': st,
                                                    'downsample': ds} not in keys_.values()]


def run_random_search(labelled, tedf, keys_, values, combinations_, indices, offset, device):
    for i, idx in enumerate(indices):
        print(f'Processing: {i}/{len(indices)}')
        (lr_start, lr_end, batch_size, model_name, epoch, scheduler_type, use_ds) = combinations_[idx]
        j = offset + i

        key = {
            'lr_start': lr_start,
            'lr_end': lr_end,
            'batch_size': batch_size,
            'model_name': model_name,
            'GAS': 1,
            'epochs': epoch,
            'scheduler_type': scheduler_type,
            'downsample': use_ds
        }

        if lr_end > lr_start:
            keys_[j] = {'error': 'ERROR'}
            values[j] = {'error': 'ERROR'}

            write_json(keys_, f'{output_dir}/config/keys.json')
            write_json(values, f'{output_dir}/config/values.json')
            continue

        value = run_cross_validation(
            k=5,
            data=labelled,
            tedf=tedf,
            device=device,
            key=key,
            to_display=False,
            process_fn=None,
            cache=None
        )

        keys_[j] = key
        values[j] = value

        write_json(keys_, f'{output_dir}/config/keys.json')
        write_json(values, f'{output_dir}/config/values.json')

        del key, value
        gc.collect()
        torch.cuda.empty_cache()
        print('Finished Processing')


if __name__ == '__main__':
    keys = read_json(f'{output_dir}/config/keys.json')
    combinations = get_hyperparameter_combinations(keys)

    run_random_search(
        labelled=pd.read_pickle(f'{shared_dir}/labelled.pickle'),
        tedf=pd.read_pickle(f'{shared_dir}/tedf.pickle'),
        keys_=keys,
        values=read_json(f'{output_dir}/config/values.json'),
        combinations_=combinations,
        indices=np.random.default_rng(config['RANDOM_SEED']).choice(range(len(combinations)),
                                                                    size=config['RANDOM_SEARCH_SIZE']),
        offset=len(keys),
        device=setup()
    )
