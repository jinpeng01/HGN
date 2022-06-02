from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
#import truecase
import re
import numpy as np
import torch
import torch.nn.functional as F

import torch.nn.functional as F
# from pytorch_transformers import (WEIGHTS_NAME, AdamW, BertConfig,
#                                   BertForTokenClassification, BertTokenizer,
#                                   WarmupLinearSchedule)
from  transformers import WEIGHTS_NAME, AdamW, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup
from transformers import (PreTrainedModel, AutoModel,AutoConfig)

from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler



from tqdm import tqdm, trange

from seqeval.metrics import classification_report

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

from HGN import HGNER

    

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, domain_label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.domain_label = domain_label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None, domain_label=None, seq_len=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.domain_label = domain_label
        self.seq_len = seq_len

def readfile(filename, type_=None):
    '''
    read file
    '''
    f = open(filename)
    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                assert len(sentence) == len(label)
                data.append((sentence,label))
                sentence = []
                label = []
            continue
            
        #TODO
        if type_!='predict':
            splits = line.split()
            #print(splits)
            sentence.append(splits[0])
            label.append(splits[-1])
            #domain_l = eval(splits[2])
            
        else:
            splits = line.strip().split()
            sentence.append(splits[0])
            label.append('O')
            
        
    if len(sentence) >0:
        data.append((sentence,label))
        assert len(sentence) == len(label)
        sentence = []
        label = []
    return data
    
def readfile_label(train_file, test_file, dev_file):
    '''
    read file
    '''
    label_dict = {}
    f_train = open(train_file)
    f_test = open(test_file)
    f_dev = open(dev_file)
    
    for line in f_train:
          temp = line.strip()
          if temp!='':
              
              splits = line.strip().split()
              if splits[-1] != 'O':
                  label_dict[splits[-1]] = 0
    
    for line in f_test:
          temp = line.strip()
          if temp!='':
              
              splits = line.strip().split()
              if splits[-1] != 'O':
                  label_dict[splits[-1]] = 0
    
    for line in f_dev:
          temp = line.strip()
          if temp!='':
              
              splits = line.strip().split()
              if splits[-1] != 'O':
                  label_dict[splits[-1]] = 0
    return label_dict

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
        
    def get_predict_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self, label_list):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
    
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None, type_=None):
        """Reads a tab separated value file."""
        return readfile(input_file, type_)
    
    
class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_dir,type_='train'), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_dir, type_='dev'), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_dir, type_='test'), "test")
    
    def get_predict_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(data_dir, type_='predict'), "predict")
    

    def get_labels(self, label_list):
        #return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]
        
        f_label_list=[]
        f_label_list.append("O")
        for i in label_list:
            f_label_list.append("B-"+i)
            f_label_list.append("I-"+i)
            f_label_list.append("E-"+i)
            f_label_list.append("S-"+i)
        f_label_list.append("[CLS]")
        f_label_list.append("[SEP]")
        return f_label_list
        
        '''
        final_label_list = []
        
        final_label_list.append("O")
        for keys, values in label_list.items():
            final_label_list.append('B-'+keys)
            final_label_list.append('I-'+keys)
            f_label_list.append("E-"+i)
            f_label_list.append("S-"+i)
        final_label_list.append("[CLS]")
        final_label_list.append("[SEP]")
        
        return final_label_list
        '''
        
        
    def _create_examples(self,lines,set_type):
        examples = []
        for i,(sentence,label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = '\t\t'.join(sentence)
            text_b = None
            label = label
            #domain_label = domain_label
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples
        
        
            
            


def truecase_sentence(tokens):
   word_lst = [(w, idx) for idx, w in enumerate(tokens) if all(c.isalpha() for c in w)]
   lst = [w for w, _ in word_lst if re.match(r'\b[A-Z\.\-]+\b', w)]

   if len(lst) and len(lst) == len(word_lst):
       parts = truecase.get_true_case(' '.join(lst)).split()

       # the trucaser have its own tokenization ...
       # skip if the number of word dosen't match
       if len(parts) != len(word_lst): return tokens

       for (w, idx), nw in zip(word_lst, parts):
           tokens[idx] = nw
   return tokens



def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list,1)}

    
    
    features = []
    ori_sents = []
    #seq_lens = []
    for (ex_index,example) in enumerate(examples):
        textlist = example.text_a.split('\t\t')
        #textlist = truecase_sentence(textlist)
        ori_sents.append(textlist)
        labellist = example.label
        #domain_label = []
        #domain_label.append(example.domain_label)
        
        tokens = []
        labels = []
        valid = []
        label_mask = []
        #
        seq_len = []
        seq_len.append(len(textlist))
        
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        
        #assert len(tokens) == len(labels)    
        
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0,1)
        label_mask.insert(0,1)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                try:
                  label_ids.append(label_map[labels[i]])
                except:
                  print(tokens)
                  print(labels)
                  #print(len(tokens))
                  #print(len(labels))
                  time.sleep(100)
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        #if ex_index < 5:
        #    logger.info("*** Example ***")
        #    logger.info("guid: %s" % (example.guid))
        #    logger.info("tokens: %s" % " ".join(
        #            [str(x) for x in tokens]))
        #    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #    logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #    logger.info(
        #            "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask,
                              seq_len=seq_len))
    return features, ori_sents


def setup_seed(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmard = False
	torch.random.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    
    parser.add_argument("--dev_data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    
    parser.add_argument("--test_data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    
    parser.add_argument("--gpu_id",
                        default=None,
                        nargs='+',
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    
    parser.add_argument("--use_crf",
                        action='store_true',
                        help="Whether use crf")
    
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")


    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
                        
    parser.add_argument("--label_list",
                        default=["O"],
                        type=str,
                        nargs='+',
                        help="Where do you want to store the pre-trained models downloaded from s3")
                        
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval or not.")
    
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run eval or not.")
    
    parser.add_argument("--eval_on",
                        default="dev",
                        help="Whether to run eval on the dev set or test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")


    parser.add_argument("--hidden_dropout_prob",
                        default=0.1,
                        type=float,
                        help="hidden_dropout_prob")

    parser.add_argument("--window_size",
                        default=-1,
                        type=int,
                        help="window_size")


    parser.add_argument("--d_model",
                        default=1024,
                        type=int,
                        help="pre-trained model size")

    
    #####
    parser.add_argument("--use_bilstm",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")


    parser.add_argument("--use_single_window",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--use_multiple_window",
                        action='store_true',
                        help="Set this flag if you are using an multiple.")

    parser.add_argument("--use_global_lstm",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--use_n_gram",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--windows_list', type=str, default='', help="window list")
    parser.add_argument('--connect_type', type=str, default='add', help="window list")


    args = parser.parse_args()
    
    
    # if not args.do_train and not args.do_eval:
    #     raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    if args.do_train:
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    handler = logging.FileHandler(args.output_dir+'/log.txt', encoding='UTF-8')
    logger.addHandler(handler)
    
    
    gpu_ids = ''
    for ids in args.gpu_id:
      gpu_ids = gpu_ids + str(ids) +','

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids


    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {"ner":NerProcessor}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))
    
    
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    #random.seed(args.seed)
    #np.random.seed(args.seed)
    #torch.manual_seed(args.seed)
    #torch.cuda.manual_seed(args.seed)
    #torch.cuda.manual_seed_all(args.seed)
    setup_seed(args.seed)
    
    

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()



    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = 0
    if args.do_train:
        train_examples = processor.get_train_examples(args.train_data_dir)
        #num_train_optimization_steps = int(
        #    len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    label_dict = readfile_label(args.train_data_dir, args.test_data_dir, args.dev_data_dir)
    #label_list = processor.get_labels(args.label_list)
    label_list = []
    label_list.append('O')
    for keys,_ in label_dict.items():
        label_list.append(keys)
    label_list.append("[CLS]")
    label_list.append("[SEP]")

    num_labels = len(label_list) + 1
    logger.info(args)
    # Prepare model
    model = HGNER(args,
                  hidden_dropout_prob=args.hidden_dropout_prob,
                  num_labels=num_labels,
                  windows_list = [int(k) for k in args.windows_list.split('qq')] if args.windows_list else args.window_size,
                  )

    n_params = sum([p.nelement() for p in model.parameters()])
    print('n_params',n_params)


    #my_lstm = LSTMLayer(config.hidden_size)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab


    #for name, param in model.named_parameters():
    #    if name.startswith('embeddings'):
    #        param.requires_grad = False

    #param_optimizer = list(model.named_parameters())
    #param_optimizer_new = []
    #for i in param_optimizer:
    #    if i[0].startswith('bert.embeddings'):
    #        continue
    #    param_optimizer_new.append(i)


    model.to(device)

    #param_optimizer = param_optimizer_new
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)


    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    label_map = {i : label for i, label in enumerate(label_list,1)}
    
    
    logger.info("*** Label map ***")
    logger.info(label_map)
    logger.info("*******************************************")
    

    best_epoch = -1
    best_p = -1
    best_r = -1
    best_f = -1
    best_test_f = -1
    best_eval_f = -1
    if args.do_train:
        #load train data
        train_features,_ = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
        #all_domain_l = torch.tensor([f.domain_label for f in train_features], dtype=torch.long)
        all_seq_lens = torch.tensor([f.seq_len for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_valid_ids,all_lmask_ids, all_seq_lens)

        #load valid data

        eval_examples = processor.get_dev_examples(args.dev_data_dir)
        eval_features,_ = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
        #all_domain_l = torch.tensor([f.domain_label for f in eval_features], dtype=torch.long)
        all_seq_lens = torch.tensor([f.seq_len for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_valid_ids,all_lmask_ids, all_seq_lens)

        #load test data
        test_examples = processor.get_test_examples(args.test_data_dir)
        test_features,_ = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer)
        all_input_ids_dev = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask_dev = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids_dev = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids_dev = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        all_valid_ids_dev = torch.tensor([f.valid_ids for f in test_features], dtype=torch.long)
        all_lmask_ids_dev = torch.tensor([f.label_mask for f in test_features], dtype=torch.long)
        #all_domain_l = torch.tensor([f.domain_label for f in test_features], dtype=torch.long)
        all_seq_lens_dev = torch.tensor([f.seq_len for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids_dev, all_input_mask_dev, all_segment_ids_dev, all_label_ids_dev, all_valid_ids_dev, all_lmask_ids_dev, all_seq_lens_dev)

        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)



        test_f1 = []
        dev_f1 = []

        for epoch_ in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                # begin_time = time.time()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, valid_ids,l_mask, seq_len = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids,valid_ids,l_mask)#, seq_len=seq_len)

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
                # end_time = time.time()
                # print('one step时间',end_time-begin_time)
            # eval in each epoch.
            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            y_true = []
            y_pred = []
            label_map = {i : label for i, label in enumerate(label_list,1)}
            for input_ids, input_mask, segment_ids, label_ids,valid_ids,l_mask, seq_len in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                valid_ids = valid_ids.to(device)
                label_ids = label_ids.to(device)
                l_mask = l_mask.to(device)
                seq_len = seq_len.to(device)
                #domain_l = domain_l.to(device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask)#, seq_len=seq_len)


                if not args.use_crf:
                    logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                input_mask = input_mask.to('cpu').numpy()


                for i, label in enumerate(label_ids):
                    temp_1 = []
                    temp_2 = []
                    for j,m in enumerate(label):
                        if j == 0:
                            continue
                        elif label_ids[i][j] == len(label_map):
                            y_true.append(temp_1)
                            y_pred.append(temp_2)
                            break
                        else:

                            temp_1.append(label_map[label_ids[i][j]])
                            try:
                                temp_2.append(label_map[logits[i][j]])
                            except:
                                temp_2.append('O')
                            #temp_2.append(label_map[logits[i][j]])

            report = classification_report(y_true, y_pred,digits=4)
            logger.info("\n******evaluate on the dev data*******")
            logger.info("\n%s", report)
            temp = report.split('\n')[-3]
            f_eval = eval(temp.split()[-2])
            dev_f1.append(f_eval)
            
            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")


            #if os.path.exists(output_eval_file):
            with open(output_eval_file, "a") as writer:
                #logger.info("***** Eval results *****")
                #logger.info("=======token level========")
                #logger.info("\n%s", report)
                #logger.info("=======token level========")
                writer.write('*******************epoch*******'+str(epoch_)+'\n')
                writer.write(report+'\n')
            

            y_true = []
            y_pred = []
            label_map = {i : label for i, label in enumerate(label_list,1)}
            for input_ids, input_mask, segment_ids, label_ids,valid_ids,l_mask, seq_len in test_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                valid_ids = valid_ids.to(device)
                label_ids = label_ids.to(device)
                l_mask = l_mask.to(device)
                seq_len = seq_len
                #domain_l = domain_l.to(device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask)
                    shape = logits.shape
                    if len(shape) < 3:
                        logits = logits.unsqueeze(dim=0)

                try:
                    if not args.use_crf:
                        logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    input_mask = input_mask.to('cpu').numpy()
                except:
                    import pdb
                    pdb.set_trace()
    
                for i, label in enumerate(label_ids):
                    temp_1 = []
                    temp_2 = []
                    for j,m in enumerate(label):
                        if j == 0:
                            continue
                        elif label_ids[i][j] == len(label_map):
                            y_true.append(temp_1)
                            y_pred.append(temp_2)
                            #print(temp_2)
                            #time.sleep(5)
                            break
                        else:
                            temp_1.append(label_map[label_ids[i][j]])
                            try:
                                temp_2.append(label_map[logits[i][j]])
                            except:
                                temp_2.append('O')
                            #temp_2.append(label_map[logits[i][j]])
    
            report = classification_report(y_true, y_pred,digits=4)
            
            logger.info("\n******evaluate on the test data*******")
            logger.info("\n%s", report)
            temp = report.split('\n')[-3]
            f_test = eval(temp.split()[-2])
            test_f1.append(f_test)
            
            
            
            output_eval_file_t = os.path.join(args.output_dir, "test_results.txt")


            #if os.path.exists(output_eval_file):
            with open(output_eval_file_t, "a") as writer2:
                #logger.info("***** Eval results *****")
                #logger.info("=======token level========")
                #logger.info("\n%s", report)
                #logger.info("=======token level========")
                writer2.write('*******************epoch*******'+str(epoch_)+'\n')
                writer2.write(report+'\n')
            

                
        # Load a trained model and config that you have fine-tuned
        output_f1_test = os.path.join(args.output_dir, "f1_score_epoch.txt")
        with open(output_f1_test, "w") as writer1:
            for i, j in zip(test_f1, dev_f1):
                writer1.write(str(i) + '\t' + str(j) + '\n')
            writer1.write('\n')
            writer1.write(str(best_test_f))

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        
        model = Ner.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(device)
        #if args.eval_on == "dev":
        #    eval_examples = processor.get_dev_examples(args.data_dir)
        #elif args.eval_on == "test":
        eval_examples = processor.get_test_examples(args.test_data_dir)
        #else:
        #    raise ValueError("eval on dev or test set only")
        eval_features,_ = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_valid_ids,all_lmask_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        y_true = []
        y_pred = []
        label_map = {i : label for i, label in enumerate(label_list,1)}
        for input_ids, input_mask, segment_ids, label_ids,valid_ids,l_mask in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            valid_ids = valid_ids.to(device)
            label_ids = label_ids.to(device)
            l_mask = l_mask.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask)

            logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()

            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                for j,m in enumerate(label):
                    if j == 0:
                        continue
                    elif label_ids[i][j] == len(label_map):
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        break
                    else:
                        temp_1.append(label_map[label_ids[i][j]])
                        temp_2.append(label_map[logits[i][j]])

        report = classification_report(y_true, y_pred,digits=4)
        logger.info("\n%s", report)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            logger.info("\n%s", report)
            writer.write(report)
        
        output_result_file = os.path.join(args.output_dir, "text_results.txt")
        with open(output_result_file, "w") as writer:
            for i,j in zip(y_true, y_pred):
                for n,m in zip(i, j):
                  writer.write(n+'\t'+m+'\n')
                writer.write('\n')
        
    if args.do_predict:
        model_best = torch.load(args.output_dir+'/model.pt')
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        model_best.to(device)

        #

        test_examples = processor.get_test_examples(args.test_data_dir)
        test_features, ori_sents = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer)
        all_input_ids_dev = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask_dev = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids_dev = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids_dev = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        all_valid_ids_dev = torch.tensor([f.valid_ids for f in test_features], dtype=torch.long)
        all_lmask_ids_dev = torch.tensor([f.label_mask for f in test_features], dtype=torch.long)
        # all_domain_l = torch.tensor([f.domain_label for f in test_features], dtype=torch.long)
        all_seq_lens_dev = torch.tensor([f.seq_len for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids_dev, all_input_mask_dev, all_segment_ids_dev, all_label_ids_dev,
                                  all_valid_ids_dev, all_lmask_ids_dev, all_seq_lens_dev)

        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
        model_best.eval()

        y_true = []
        y_pred = []
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask, seq_len in test_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            valid_ids = valid_ids.to(device)
            label_ids = label_ids.to(device)
            l_mask = l_mask.to(device)
            seq_len = seq_len
            # domain_l = domain_l.to(device)

            with torch.no_grad():
                logits = model_best(input_ids, segment_ids, input_mask, valid_ids=valid_ids,
                                    attention_mask_label=l_mask)
                shape = logits.shape
                if len(shape) < 3:
                    logits = logits.unsqueeze(dim=0)

            if not args.use_crf:
                logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()

            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                for j, m in enumerate(label):
                    if j == 0:
                        continue
                    elif label_ids[i][j] == len(label_map):
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        # print(temp_2)
                        # time.sleep(5)
                        break
                    else:
                        temp_1.append(label_map[label_ids[i][j]])
                        try:
                            temp_2.append(label_map[logits[i][j]])
                        except:
                            temp_2.append('O')
                        # temp_2.append(label_map[logits[i][j]])

        report = classification_report(y_true, y_pred, digits=4)

        logger.info("\n******evaluate on the test data*******")
        logger.info("\n%s", report)

        # for i, sents in enumerate(ori_sents):
        #    temp = []
        #    temp2 = []
        #    for j,m in enumerate(sents):
        #       try:
        #         temp.append(label_map[logits[i][j]])
        #       except:
        #         temp.append('O')
        #       temp2.append(label_map[label_ids[i][j]])
        # y_pred.append(temp)
        # y_true.append(temp2)

        output_result_file = os.path.join(args.output_dir, "predict_results.txt")
        with open(output_result_file, "w") as writer:
            for i, j, k in zip(ori_sents, y_true, y_pred):
                for n, m, l in zip(i, j, k):
                    writer.write(n + ' ' + m + ' ' + l + '\n')
                writer.write('\n')
        
              


if __name__ == "__main__":
    main()
