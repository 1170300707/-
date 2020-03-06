# -*- coding: utf-8 -*-

import pandas as pd

def seriesToSupervisedLearning(data, columns, n_in = 1, n_out = 1, dropnan = True):

    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols = list()
    names = list()

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s%d(t-%d)' % (columns[j], j+1, i)) for j in range(n_vars)]

    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('%s%d(t)' % (columns[j], j + 1)) for j in range(n_vars)]
        else:
            names += [('%s%d(t+%d)' % (columns[j], j + 1, i)) for j in range(n_vars)]

    agg = pd.concat(cols, axis=1)
    agg.columns = names

    if dropnan:
        clean_agg = agg.dropna()
    return clean_agg

if __name__ == '__main__':
    values = [x for x in range(10)]
    data = seriesToSupervisedLearning(values, ['A'], 5)
    print(data)