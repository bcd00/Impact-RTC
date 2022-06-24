import torch.nn as nn

from transformers import Trainer as TransformerTrainer
from sklearn.metrics import f1_score, classification_report
from transformers.optimization import Optimizer, get_polynomial_decay_schedule_with_warmup, \
    get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup


class Trainer(TransformerTrainer):
    def __init__(self, device, output_tf, *rgs, lr_end=1e-5, scheduler_type='linear', **kwargs):
        super().__init__(*rgs, **kwargs)
        self.device = device
        self.output_tf = output_tf
        self.losses = []
        self.lrs = []
        self.lr_end = lr_end
        self.scheduler_type = scheduler_type

    def compute_loss(self, model, inputs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)

        outputs = self.output_tf(outputs)

        loss_task = nn.CrossEntropyLoss()
        loss = loss_task(outputs.view(-1, 2), labels.view(-1))
        self.losses.append(loss)
        self.lrs.append(self.lr_scheduler.get_last_lr()[0])
        return loss

    # noinspection PyTypeChecker
    @staticmethod
    def compute_metric_eval(arg):
        predictions, labels = arg[0], arg[1]
        return {'f1_macro': f1_score(labels, predictions, average='macro'),
                'rtc_f1': classification_report(labels, predictions, target_names=["Not RTC", "RTC"], output_dict=True)[
                    'RTC']['f1-score']}

    # noinspection PyAttributeOutsideInit
    def create_scheduler(self, num_training_steps: int, optimizer: Optimizer = None):
        if self.lr_scheduler is None:
            if self.scheduler_type == 'polynomial':
                self.lr_scheduler = get_polynomial_decay_schedule_with_warmup(
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=0,
                    num_training_steps=num_training_steps,
                    lr_end=self.lr_end,
                    power=2.0,
                )
            elif self.scheduler_type == 'linear':
                self.lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer=self.optimizer,
                    num_warmup_steps=0,
                    num_training_steps=num_training_steps,
                )
            elif self.scheduler_type == 'cosine':
                self.lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=self.optimizer,
                    num_warmup_steps=0,
                    num_training_steps=num_training_steps
                )
            else:
                self.lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=0)

        return self.lr_scheduler
