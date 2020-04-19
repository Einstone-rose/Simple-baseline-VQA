import json
import pickle
from collections import Counter
from torchtext import data
import torchtext
from os.path import join


def _create_tsv(data_dir, quesfile, ansfile, outfile, ansid=None):
    quesfile = join(data_dir, quesfile)
    ques_json = json.load(open(quesfile))

    ques = [q['question'] for q in ques_json['questions']] # 按照顺序取出所有的问题
    quesid = [q['question_id'] for q in ques_json['questions']]
    imgid = [q['image_id'] for q in ques_json['questions']]

    if ansfile is not None:
        ansfile = join(data_dir, ansfile)
        ans_json = json.load(open(ansfile))
        ans = [a['multiple_choice_answer'] for a in ans_json['annotations']]
        k = 1000
        if ansid is None:
                c = Counter(ans) #这个函数的功能可以直接统计单词的数量
                topk = c.most_common(n=k) #找到最频繁的1000个出现的答案
                # topk: [('yes',82590),('no',87420),('kp',52331),...]
                ansid = dict((a[0], i) for i, a in enumerate(topk))
                # ansid: {'yes': 0, 'no': 1, '1': 2, '2': 3, 'white': 4, '3': 5, 'blue': 6}   ans-to-id

                ans_itos_file = join(data_dir, 'ans_itos.tsv')
                with open(ans_itos_file, 'w') as fout:
                    for i, (a, freq) in enumerate(topk):
                        fout.write('{}\t{}\t{}'.format(i, a, freq) + '\n')
                    fout.write('{}\t{}\t{}'.format(k, '<unk>', 'rest') + '\n')

        ans = [ansid[a] if a in ansid else k for a in ans] # 将每一个答案映射为一个数字，频率比较少的就直接映射为1000
    else:
        ans = [0 for q in ques] # [0,0,0,0,0,0]
    outfile = join(data_dir, outfile)
    with open(outfile, 'w') as out:
            for q, qid, i, a in zip(ques, quesid, imgid, ans):
                    out.write('\t'.join([str(qid), q, str(i), str(a)]) + '\n')
    return ansid

def _create_loaders(path, traintsv, valtsv):
    def parse_int(tok, *args):
        return int(tok)
    # 对应.tsv文件中的几列,通过quesid.data[0]可以获得数据,ques.data[0]可以获得数据
    quesid = data.Field(sequential=False, use_vocab=False, postprocessing=data.Pipeline(parse_int))
    ques = data.Field(include_lengths=True)
    imgid = data.Field(sequential=False, use_vocab=False, postprocessing=data.Pipeline(parse_int))
    ans = data.Field(sequential=False, use_vocab=False, postprocessing=data.Pipeline(parse_int))

    train_data, val_data = data.TabularDataset.splits(path=path, train=traintsv, validation=valtsv,
                                                      fields=[('quesid', quesid), ('ques', ques), ('imgid', imgid), ('ans', ans)],
                                                      format='tsv')
    batch_sizes = (1, 1)
    train_loader, val_loader = data.BucketIterator.splits((train_data, val_data), batch_sizes=batch_sizes, repeat=False, sort_key=lambda x: len(x.ques))

    ques.build_vocab(train_data)
    print('vocabulary size: {}'.format(len(ques.vocab.stoi)))
    return ques, train_loader, val_loader


def _dump_datasets(loader, outfile, sorted=False):
    examples = []
    for ex in loader:
        examples.append((
            ex.quesid.data[0],
            ex.ques[0].data.squeeze().cpu().numpy(),        # squeeze the batch dim (as batch_size=1)
            ex.ques[1][0],
            ex.imgid.data[0],
            ex.ans.data[0]))
    # examples[0]:
    # (tensor(335624003), array([177, 6, 50, 106, 8, 9, 600], dtype=int64), tensor(7), tensor(335624), tensor(1))
    if not sorted:
        # required only for train_loader. Other loaders give examples in sorted order.
        examples.sort(key=lambda ex: ex[2])
    with open(outfile, 'wb') as trainf:
        pickle.dump(examples, trainf)



def _dump_vocab(vocab, outfile):
    with open(outfile, 'w') as fout:
        #  vocab在这里其实是一个字典
        for tok, idx in vocab:
            fout.write('{}\t{}'.format(tok, idx) + '\n')


def preprocess(data_dir, train_ques_file, train_ans_file, val_ques_file, val_ans_file):

    train_tsv_file, val_tsv_file = 'train.tsv', 'val.tsv'
    print('Creating tsv datasets: {}, {}'.format(train_tsv_file, val_tsv_file))
    # 将训练集的问题和答案都储存在train_tsv_file文件中
    ansid = _create_tsv(data_dir=data_dir, quesfile=train_ques_file, ansfile=train_ans_file, outfile=train_tsv_file)
    # create val tsv using ans-to-idx mapping made on train data
    # 将训练集的问题和答案都储存在val_tsv_file文件中
    _create_tsv(data_dir=data_dir, quesfile=val_ques_file, ansfile=val_ans_file, outfile=val_tsv_file, ansid=ansid)

    print('Creating loaders...')
    ques, train_loader, val_loader = _create_loaders(data_dir, train_tsv_file, val_tsv_file)
    print('loaders have been created...')

    ques_stoi_file = join(data_dir, 'ques_stoi.tsv')
    print('Dumping vocabulary to {}'.format(ques_stoi_file))
    # ques一共有三个属性:
    # 1> freqs   Counter({'I':1,'This':2,'a':1,...})
    # 2> itos    ['<unk>', '<pad>', 'This', 'movie', 'I', 'a',..,]
    # 3> stoi    ['I': 4,'love':5,'This':2,..,'to':12]
    _dump_vocab(ques.vocab.stoi.items(), ques_stoi_file)

    train_data_file, val_data_file = join(data_dir, 'train.pkl'), join(data_dir, 'val.pkl')
    print('Dumping train dataset to {}'.format(train_data_file))
    _dump_datasets(train_loader, outfile=train_data_file)
    print('Dumping val dataset to {}'.format(val_data_file))
    _dump_datasets(val_loader, outfile=val_data_file, sorted=True)
    print("dumping are finished...")
