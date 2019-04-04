import re
import pickle
import numpy as np

poem_path = r"./poem/poems.txt"
poem_pick = r"./poem/"


def get_data():
    result = dict()
    word = set()
    pat_replace = re.compile('（.*）')
    pat_del = re.compile('】|【|卷\d+|□')
    with open(poem_path, encoding='utf-8') as poem_file:
        for poem in poem_file.readlines():
            poem = poem.strip()
            if poem.count(':') != 1:
                continue
            if len(pat_del.findall(poem)) > 0:
                continue
            poem = poem.replace('_', '')
            poem = poem.replace('…', '-')
            [name, con] = poem.split(':')
            name = pat_replace.sub('', name)
            con = pat_replace.sub('', con)
            if len(con) < 10:
                continue
            result[name] = con
            word.update(set(name))
            word.update(set(con))
    return result, word


res, word = get_data()

word2num = dict(zip(word, range(1, len(word) + 1)))
num2word = dict(zip(word2num.values(), word2num.keys()))

res = list(map(lambda x: [np.array(list(map(lambda xx: word2num[xx], x))),
                          np.array(list(map(lambda xx: word2num[xx], res[x])))], res))
res = np.array(res)

res = {'text': res, 'len': len(word), 'word2num': word2num, 'num2word': num2word}
with open(poem_pick + 'poems.plk', mode='wb') as plk:
    pickle.dump(res, plk)
