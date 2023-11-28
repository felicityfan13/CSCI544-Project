import sys

sys.path.insert(0, '/mnt/c/users/felic/good-translation-wrong-in-context/') # insert your local path to the repo

import pickle
import numpy as np

DATA_PATH = "/mnt/c/users/felic/good-translation-wrong-in-context/scripts/data/bpe/" # insert your datadir
VOC_PATH =  "/mnt/c/users/felic/good-translation-wrong-in-context/scripts/build/"# insert your path

inp_voc = pickle.load(open(VOC_PATH + 'src.voc', 'rb'))
out_voc = pickle.load(open(VOC_PATH + 'dst.voc', 'rb'))

import lib as lib
import tensorflow as tf
import lib.task.seq2seq.models.transformer as tr

tf.reset_default_graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99, allow_growth=True)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

hp = {
    "num_layers": 6,
    "num_heads": 8,
    "ff_size": 2048,
    "ffn_type": "conv_relu",
    "hid_size": 512,
    "emb_size": 512,
    "res_steps": "nlda",

    "rescale_emb": True,
    "inp_emb_bias": True,
    "normalize_out": True,
    "share_emb": False,
    "replace": 0,

    "relu_dropout": 0.1,
    "res_dropout": 0.1,
    "attn_dropout": 0.1,
    "label_smoothing": 0.1,

    "translator": "ingraph",
    "beam_size": 4,
    "beam_spread": 3,
    "len_alpha": 0.6,  
    "attn_beta": 0,
}

model = tr.Model('mod', inp_voc, out_voc, inference_mode='fast', **hp)

path_to_ckpt = "/mnt/c/users/felic/good-translation-wrong-in-context/scripts/build/checkpoint/model-latest.npz" # insert path to the final checkpoint
var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
lib.train.saveload.load(path_to_ckpt, var_list)

path_to_testset = "/mnt/c/users/felic/good-translation-wrong-in-context/scripts/test/"
test_src = open(path_to_testset + 'deixis_dev.src').readlines()
test_dst = open(path_to_testset + 'deixis_dev.dst').readlines()
test_src2 = open(path_to_testset + 'ellipsis_infl.src').readlines()
test_dst2 = open(path_to_testset + 'ellipsis_infl.dst').readlines()
test_src3 = open(path_to_testset + 'lex_cohesion_dev.src').readlines()
test_dst3 = open(path_to_testset + 'lex_cohesion_dev.dst').readlines()

print(test_src[:4])
print(test_dst[:4])
print(test_src2[:4])
print(test_dst2[:4])
print(test_src3[:4])
print(test_dst3[:4])

from lib.task.seq2seq.problems.default import DefaultProblem
from lib.task.seq2seq.data import make_batch_placeholder
problem = DefaultProblem({'mod': model})
batch_ph = make_batch_placeholder(model.make_feed_dict(model._get_batch_sample()))
loss_values = problem.loss_values(batch=batch_ph, is_train=False)


def num_sents(text):
    return len(text.split(' _eos '))


def make_baseline_batch_data(src_lines, dst_lines):
    """
    src_lines contain groups of N sentences, last of which is to be translated (' _eos '-separated)
    dst_lines contain translations of sentences in src_lines (' _eos '-separated)

    returns: list of pairs (src, dst) which one can give to a model
    """
    assert len(src_lines) == len(dst_lines), "Different number of text fragments"
    batch_src = []
    batch_dst = []
    for src, dst in zip(src_lines, dst_lines):
        assert num_sents(src) == num_sents(dst)
        batch_src.append(src.split(' _eos ')[-1])
        batch_dst.append(dst.split(' _eos ')[-1])
    return list(zip(batch_src, batch_dst))


def score_batch(src_lines, dst_lines, name):
    feed_dict = model.make_feed_dict(make_baseline_batch_data(src_lines, dst_lines))
    feed = {batch_ph[k]: feed_dict[k] for k in batch_ph}
    scores = sess.run(loss_values, feed)
    print(name + ":")
    return scores

score_batch(test_src[:4], test_dst[:4], "deixis")
score_batch(test_src2[:4], test_dst2[:4], "ellipsis")
score_batch(test_src3[:4], test_dst3[:4], "lexical_cohesion")
