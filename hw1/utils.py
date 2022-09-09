import re
import torch
import numpy as np
from collections import Counter


def get_device(force_cpu, status=True):
    # if not force_cpu and torch.backends.mps.is_available():
    # 	device = torch.device('mps')
    # 	if status:
    # 		print("Using MPS")
    # elif not force_cpu and torch.cuda.is_available():
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if status:
            print("Using CUDA")
    else:
        device = torch.device("cpu")
        if status:
            print("Using CPU")
    return device


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", "", s)
    # Replace all runs of whitespaces with one space
    s = re.sub(r"\s+", " ", s)
    # replace digits with no space
    s = re.sub(r"\d", "", s)
    return s


"""count number of occurences as each words. 
return word to num and num to word dictionaries. labels are actions and targets i.e. pick up ,waterbottle
input = sentence, output  = (action, target)"""
def build_tokenizer_table(train, vocab_size=1000):
    word_list = []
    padded_lens = []
    inst_count = 0
    for episode in train: # episode is a tuple
        for inst, _ in episode: # inst is sentence _ is labels i guess
            inst = preprocess_string(inst)
            padded_len = 2  # start/end
            for word in inst.lower().split():
                if len(word) > 0:
                    word_list.append(word)
                    padded_len += 1
            padded_lens.append(padded_len)
    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[
        : vocab_size - 4
    ]  # save room for <pad>, <start>, <end>, and <unk>
    vocab_to_index = {w: i + 4 for i, w in enumerate(corpus_)}
    vocab_to_index["<pad>"] = 0
    vocab_to_index["<start>"] = 1
    vocab_to_index["<end>"] = 2
    vocab_to_index["<unk>"] = 3
    index_to_vocab = {vocab_to_index[w]: w for w in vocab_to_index}
    return (
        vocab_to_index,
        index_to_vocab,
        int(np.average(padded_lens) + np.std(padded_lens) * 2 + 0.5),
    )

# TRAIN FORMAT [(sentence, (action, target))]
def build_output_tables(train):
    actions = set()
    targets = set()
    for episode in train:
        for _, outseq in episode: # _ is sentence, outseq is labels
            a, t = outseq
            actions.add(a)
            targets.add(t)
    actions_to_index = {a: i for i, a in enumerate(actions)}
    targets_to_index = {t: i for i, t in enumerate(targets)}
    index_to_actions = {actions_to_index[a]: a for a in actions_to_index}
    index_to_targets = {targets_to_index[t]: t for t in targets_to_index}
    return actions_to_index, index_to_actions, targets_to_index, index_to_targets

def create_train_val_splits(all_lines, prop_train=0.8):
    train_lines = []
    val_lines = []
    # get all lines in a given book, [1] is the book classification
    # lines = [all_lines[idx] for idx in range(len(all_lines)) if all_lines[idx][1] == b]
    # generate a random sample of size 80% of the total lines in this book to use for training
    val_idxs = np.random.choice(list(range(len(all_lines))), size=int(len(all_lines)*prop_train + 0.5), replace=False)
    # assign train/test lines accordingly for whether we chose them above
    # train_lines.extend([lines[idx] for idx in range(len(lines)) if idx not in val_idxs])
    # val_lines.extend([lines[idx] for idx in range(len(lines)) if idx in val_idxs])
    val_lines.extend([all_lines[idx] for idx in range(len(all_lines)) if idx in val_idxs])
    train_lines.extend([all_lines[idx] for idx in range(len(all_lines)) if idx not in val_idxs])
    # split is 80-20
    return train_lines, val_lines

# actions, targets to indices
def encode_data(data, v2i, seq_len, a2i, t2i):
    # v2i: vocab to index
    n_lines = len(data)
    #b2i: books to indices?
    n_a = len(a2i)
    n_t = len(t2i)
    # x = np.zeros((n_lines, seq_len), dtype=np.int32)
    # y = np.zeros((n_lines), dtype=np.int32)
    temp_x = []
    temp_y = []
    idx = 0
    n_early_cutoff = 0
    n_unks = 0
    n_tks = 0
    for episode in data:
        for txt, label in episode:
            a = label[0]
            t = label[1]
            txt = preprocess_string(txt)
            temp = np.zeros(seq_len)
            # temp.append(v2i["<start>"])
            temp[0] = v2i["<start>"]
            jdx = 1
            for word in txt.split():
                if len(word) > 0:
                    temp[jdx] = v2i[word] if word in v2i else v2i["<unk>"]
                    n_unks += 1 if temp[jdx] == v2i["<unk>"] else 0
                    n_tks += 1
                    jdx += 1
                    if jdx == seq_len - 1:
                        n_early_cutoff += 1
                        break
            temp[jdx] = v2i["<end>"]
            # get actions and targets
            labels = np.array([a2i[a], t2i[t]])
            temp_x.append(temp)
            temp_y.append(labels)
            idx += 1
    print(
        "INFO: had to represent %d/%d (%.4f) tokens as unk with vocab limit %d"
        % (n_unks, n_tks, n_unks / n_tks, len(v2i))
    )
    print(
        "INFO: cut off %d instances at len %d before true ending"
        % (n_early_cutoff, seq_len)
    )
    print("INFO: encoded %d instances without regard to order" % idx)
    y = np.array(temp_y)
    x = np.array(temp_x, dtype=np.int32)
    return x, y