import torch
from torch import nn

class BaseModel(nn.Module):
    def __init__(self, h, input_size, hist_exog_list=None, stat_exog_list=None, futr_exog_list=None,
                 exclude_insample_y=False, loss=None, valid_loss=None, max_steps=0, learning_rate=0.001,
                 num_lr_decays=-1, early_stop_patience_steps=-1, val_check_steps=100, batch_size=32,
                 valid_batch_size=None, windows_batch_size=1, inference_windows_batch_size=1,
                 start_padding_enabled=False, step_size=1, scaler_type="identity", drop_last_loader=False,
                 alias=None, random_seed=1, optimizer=None, optimizer_kwargs=None, lr_scheduler=None,
                 lr_scheduler_kwargs=None, dataloader_kwargs=None, **trainer_kwargs):
        super().__init__()
        self.h = h
        self.input_size = input_size
        self.loss = loss
        self.valid_loss = valid_loss if valid_loss is not None else loss
        self.futr_exog_size = len(futr_exog_list or [])

    def forward(self, *args, **kwargs):
        raise NotImplementedError
