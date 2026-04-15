import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output.
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.bos_token_id = self.tokenizer.pad_token_id  # T5 uses pad_token as decoder_start_token
        self.eos_token_id = self.tokenizer.eos_token_id
        self.encoder_inputs, self.decoder_inputs, self.decoder_targets = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_lines = load_lines(nl_path)

        encoder_inputs = []
        for line in nl_lines:
            prefix = "translate English to SQL: " + line
            ids = tokenizer.encode(prefix, add_special_tokens=True)
            encoder_inputs.append(torch.tensor(ids, dtype=torch.long))

        decoder_inputs = []
        decoder_targets = []
        if split != 'test':
            sql_path = os.path.join(data_folder, f'{split}.sql')
            sql_lines = load_lines(sql_path)
            for line in sql_lines:
                ids = tokenizer.encode(line, add_special_tokens=True)
                dec_in = [self.bos_token_id] + ids[:-1]
                dec_tgt = ids
                decoder_inputs.append(torch.tensor(dec_in, dtype=torch.long))
                decoder_targets.append(torch.tensor(dec_tgt, dtype=torch.long))

        return encoder_inputs, decoder_inputs, decoder_targets

    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        if self.split == 'test':
            return self.encoder_inputs[idx], torch.tensor([self.bos_token_id], dtype=torch.long)
        return self.encoder_inputs[idx], self.decoder_inputs[idx], self.decoder_targets[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.
    '''
    encoder_inputs, decoder_inputs, decoder_targets = zip(*batch)

    encoder_ids = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    decoder_inputs_padded = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_targets_padded = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = decoder_inputs_padded[:, :1]

    return encoder_ids, encoder_mask, decoder_inputs_padded, decoder_targets_padded, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.
    '''
    encoder_inputs, initial_tokens = zip(*batch)

    encoder_ids = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.stack(initial_tokens, dim=0)

    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    return train_x, train_y, dev_x, dev_y, test_x