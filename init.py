import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict, OrderedDict
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, T5ForConditionalGeneration

from tensorboardX import SummaryWriter

# Utility functions
import util

k_name = "unifiedQA"
k_banner = "Designed by Mingzhe Du"
k_model = "allenai/unifiedqa-t5-small"
k_save_dir = "./output"
k_data_dir = "./input"
k_seed = 42
k_max_src_len = 512
k_max_tgt_len = 10
k_batch_size = 4
k_num_train = -1
k_num_val = -1
k_num_workers = 0
k_epochs = 5
k_lr = 5e-5
k_adam_eps = 1e-8
k_warmup_steps = 0
k_max_grad_norm = 1.0

record_dir = util.get_save_dir(k_save_dir, k_name)

# Logger
logger = util.get_logger(record_dir, "root")

# TensorBoard X
tbx = SummaryWriter(record_dir, flush_secs=5)

