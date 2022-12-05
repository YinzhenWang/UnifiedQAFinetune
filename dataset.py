from init import *

class T5DataSet(Dataset):
    def __init__(self, tokenizer, data_dir: str, type_path, max_examples=-1, max_src_len=256, max_tgt_len=256):
        """
        max_examples: if > 0 then will load only max_examples into the dataset; -1 means use all
        max_src_len and max_tgt_len refer to number of tokens in the input sequences. These are not randomized. If they were we might need to collate.
        """

        valid_type_paths = ["test", "train", "val"]
        assert type_path in valid_type_paths, f"Type path must be one of {valid_type_paths}"

        self.example_path = Path(data_dir) / type_path
        self.max_examples = max_examples
        self.tokenizer = tokenizer

        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        self.inputs = []  # List of dict
        self.targets = []  # List of dict
        self.input_text = []  # List of str
        self.target_text = []  # List of str

        self._build()  # Fill inputs, targets, max_lens

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # Might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # Might need to squeeze

        src_text = self.input_text[index]
        tgt_text = self.target_text[index]

        # These will be cast to torch.long in forward
        return {"source_ids": source_ids, "source_mask": src_mask,
                "target_ids": target_ids, "target_mask": target_mask,
                "source_text": src_text, "target_text": tgt_text}

    def _build(self):
        source_path = self.example_path.with_suffix(".source")
        target_path = self.example_path.with_suffix(".target")

        with open(source_path, 'r') as f_source, open(target_path, 'r') as f_target:
            source, target = f_source.readlines(), f_target.readlines()
            source_ct, target_ct = len(source), len(target)
            assert source_ct == target_ct, f"Lengths don't match"

            # Note we could batch encode
            logger.warning(f'Using max_src_len, max_tgt_len = ({self.max_src_len}, {self.max_tgt_len})')

            inputs_out = []  # Accumulate the output of batch_encode
            targets_out = []  # Same
            inputs_text = []  # Save the original text for evaluations
            targets_text = []  # Aame

            if self.max_examples > 0:
                source_ct = min(self.max_examples, source_ct)

            for idx in range(source_ct):
                src = source[idx].strip()
                tgt = target[idx].strip()

                inputs_text.append(src)
                targets_text.append(tgt)

                # Tokenize
                # padding="max_length" pads to max_len, otherwise (e.g. for batch), we could use padding=longest with truncation.
                # self.tokenizer returns a dict of input_ids and attention_masks (where attn masks corresponds to padding)
                # NOTE: don't need add_special_tokens since EOS added automatically and others are PAD
                # NOTE: padding could also be done via collate in dataloader
                # TODO: we could actually batch encode these (i.e. multiple per)
                tokenized_inputs = self.tokenizer([src], max_length=self.max_src_len, padding="max_length",
                                                  return_tensors="pt", truncation=True)
                tokenized_targets = self.tokenizer([tgt], max_length=self.max_tgt_len, padding="max_length",
                                                   return_tensors="pt", truncation=True)

                inputs_out.append(tokenized_inputs)
                targets_out.append(tokenized_targets)

            self.inputs = inputs_out
            self.targets = targets_out
            self.input_text = inputs_text
            self.target_text = targets_text


def get_dataloaders(tokenizer, batch_size, num_train, num_val, data_dir, num_workers, shuffle_train=True,
                    shuffle_dev=False):
    """
    Returns: Tuple[train_loader : DataLoader, dev_loader : DataLoader]
    # NOTE: we default to not shuffling the dev set
    """
    # todo: should pass max src and max tgt len in as arguments
    train_data_set = T5DataSet(tokenizer, type_path="train", data_dir=data_dir, max_examples=num_train,
                               max_src_len=k_max_src_len, max_tgt_len=k_max_tgt_len)
    eval_data_set = T5DataSet(tokenizer, type_path="val", data_dir=data_dir, max_examples=num_val,
                              max_src_len=k_max_src_len, max_tgt_len=k_max_tgt_len)

    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    eval_loader = DataLoader(eval_data_set, batch_size=batch_size, shuffle=shuffle_dev, num_workers=num_workers)

    logger.info(f'Datasets loaded with sizes: train: {len(train_data_set)}, dev: {len(eval_data_set)}')

    return train_loader, eval_loader