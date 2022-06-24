import torch
import pandas as pd

from utilities.classifier.dataset import Dataset
from utilities.classifier.trainer import Trainer
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from transformers import TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from utilities.utils import downsample, checkpoints_dir, plot_confusion_matrix, plot_losses, \
    plot_lrs, print_report, kappa, visualise_cf_samples


class NLPModel:
    """
    Model for classifying tweets.
    """

    def __init__(
            self,
            training_data,
            validation_data,
            device,
            model_name,
            epochs=2,
            use_downsampling=True,
            learning_rate=1e-4,
            learning_rate_end=1e-5,
            scheduler_type='polynomial',
            batch_size=32,
            gradient_accumulation_steps=1,
            model_filename=None,
            tokenizer_filename=None
    ):
        """
        Basic model configuration.
        :param training_data: training data
        :param validation_data: validation data
        :param device: device to run model on
        :param model_name: name of model to run
        :param epochs: number of epochs to run for
        :param use_downsampling: whether to use downsampling on training data
        :param learning_rate: initial learning rate for model
        :param learning_rate_end: final learning rate for model
        :param scheduler_type: learning rate scheduler type
        :param batch_size: batch size for model
        :param gradient_accumulation_steps: number of gradient accumulation steps - used to mock batch size when
            too large for memory
        :param model_filename: filename for pre-trained model
        :param tokenizer_filename: filename for pre-trained tokenizer
        """
        if model_name in ['roberta-large', 'microsoft/deberta-base']:
            gradient_accumulation_steps = batch_size / 4
            batch_size = 4
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_end = learning_rate_end
        self.learning_rate = learning_rate
        self.output_tf = lambda x: x.logits
        self.scheduler_type = scheduler_type
        config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name if tokenizer_filename is None else tokenizer_filename)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name if model_filename is None else model_filename, config=config)

        if model_filename is not None:
            self.model.to(self.device)

        self.lrs = []
        self.ck = None
        self.losses = []
        self.results = {}
        self.report = None
        self.predictions = []
        self.model_outputs = []
        self.confusion_matrix = None
        self.use_downsampling = use_downsampling

        self.training_dataset = Dataset(self.tokenizer,
                                        downsample(training_data) if use_downsampling else training_data)
        self.validation_data = validation_data.copy()
        self.validation_dataset = Dataset(self.tokenizer, validation_data)
        self.validation_dataloader = DataLoader(self.validation_dataset)

    def train(self, log_level='passive'):
        """
        Trains model.
        :param log_level: for controlling logging
        :return:
        """
        if log_level != 'critical':
            print("--------------------Train model--------------------")

        # noinspection PyTypeChecker
        training_args = TrainingArguments(
            logging_steps=100,
            log_level=log_level,
            log_level_replica=log_level,
            logging_strategy='no',
            report_to=['none'],
            optim='adamw_torch',
            eval_accumulation_steps=100,
            save_strategy='no',
            num_train_epochs=self.epochs,
            learning_rate=self.learning_rate,
            output_dir=f'{checkpoints_dir}/main',
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )

        trainer = Trainer(
            self.device,
            self.output_tf,
            lr_end=self.lr_end,
            scheduler_type=self.scheduler_type,
            model=self.model,
            args=training_args,
            train_dataset=self.training_dataset,
            data_collator=self.training_dataset.collate_fn,
        )

        trainer.train()

        self.losses = trainer.losses
        self.lrs = trainer.lrs

    def eval(self, validation_data=None, to_display=True):
        """
        Evaluates model.
        :param validation_data: allows for evaluating on separate dataset
        :param to_display: whether to display information
        :return:
        """
        data_type = 'validation' if validation_data is None else 'test'
        if to_display:
            print(f"--------------------Evaluate on {data_type} data--------------------")

        validation_loader = DataLoader(
            Dataset(self.tokenizer, validation_data)) if validation_data is not None else self.validation_dataloader
        self.predictions = []
        tot_labels = []

        with torch.no_grad():
            for data in tqdm(validation_loader, disable=(not to_display)):
                self.predictions.extend(self._predict(data['text'])['prediction'].tolist())
                tot_labels.extend(data['labels'].tolist())

        self.report = classification_report(tot_labels, self.predictions, target_names=["negative", "positive"],
                                            output_dict=True, zero_division=0)
        self.confusion_matrix = confusion_matrix(tot_labels, self.predictions)
        zs = list(zip(tot_labels, self.predictions))

        self.ck = kappa(
            tp=len([z for z in zs if z[0] == 1 and z[1] == 1]),
            tn=len([z for z in zs if z[0] == 0 and z[1] == 0]),
            fp=len([z for z in zs if z[0] == 0 and z[1] == 1]),
            fn=len([z for z in zs if z[0] == 1 and z[1] == 0])
        )
        self.report['kappa'] = self.ck

    def print_stats(self, df, cfm_filename=None, training=False, visualise_df=False):
        """
        Prints model statistics.
        :param df: data for visualising confusion matrix samples
        :param cfm_filename: filename for saving confusion matrix to
        :param training: whether to print losses and learning rates
        :param visualise_df: whether to visualise confusion matrix samples
        :return:
        """
        print("--------------------Predict--------------------")
        print_report(self.report)
        plot_confusion_matrix(self.confusion_matrix, filename=cfm_filename)

        if visualise_df:
            visualise_cf_samples(df)

        if training:
            plot_losses(self.get_losses())
            plot_lrs(self.lrs)

    def _predict(self, dataset):
        """
        Helper function for predicting labels for a dataset
        :param dataset: dataset to label
        :return: predictions
        """
        self.model.eval()
        encodings = self.tokenizer(dataset, return_tensors='pt', padding=True, truncation=True, max_length=128)
        encodings.to(self.device)
        output = self.model(**encodings)
        predictions = torch.max(self.output_tf(output), 1)

        return {'prediction': predictions[1], 'confidence': predictions[0]}

    def get_losses(self):
        """
        Formats losses into a list.
        :return: losses
        """
        return [loss.cpu().detach().numpy().tolist() for loss in self.losses]

    def test(self, tedf, cfm_filename=None, to_display=True, visualise_df=False):
        """
        Tests model.
        :param tedf: testing data
        :param cfm_filename: filename for saving confusion matrix to
        :param to_display: whether to display information
        :param visualise_df: whether to visualise confusion matrix samples
        :return: negative f1, positive f1
        """
        ed = tedf.copy()
        self.eval(ed, to_display=to_display)
        ed['prediction'] = self.predictions
        if to_display:
            self.print_stats(ed, cfm_filename, visualise_df=visualise_df)
        return self.report['negative']['f1-score'], self.report['positive']['f1-score']

    def fit(self, data: pd.DataFrame, key='label', silent=False):
        """
        Labels data.
        :param data: data to label
        :param key: key for saving predictions
        :param silent: disables tqdm
        :return: labelled data
        """
        self.model.eval()
        vl = DataLoader(Dataset(self.tokenizer, data), batch_size=200)
        predictions = []

        with torch.no_grad():
            for data_ in tqdm(vl, disable=silent):
                predictions.extend(self._predict(data_['text'])['prediction'].tolist())
        data[key] = predictions
        return data

    def run_all(self):
        """
        Trains and evaluates model, then prints its statistics.
        :return:
        """
        self.train()
        self.eval()
        self.print_stats(df=self.validation_data, training=True,
                         visualise_df=False)
