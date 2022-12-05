from init import *

record_dir = util.get_save_dir(k_save_dir, k_name)

# Logger
logger = util.get_logger(record_dir, "root")

# TensorBoard X
tbx = SummaryWriter(record_dir, flush_secs=5)