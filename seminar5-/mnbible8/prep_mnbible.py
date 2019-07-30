#!/usr/bin/env python
# coding=utf-8

import os
import sys
import zipfile

if os.path.exists('train.txt'):
    print('Tokenized enwik8 already exists - skipping processing')
    sys.exit()

data = zipfile.ZipFile('mnbible.zip').read('mnbible')

print('Length of mnbible: {}'.format(len(data)))

num_test_chars = int(len(data) * 0.1)

train_data = data[: -2 * num_test_chars]
valid_data = data[-2 * num_test_chars: -num_test_chars]
test_data = data[-num_test_chars:]


for fn, part in [('train.txt', train_data), ('valid.txt', valid_data), ('test.txt', test_data)]:
    print('{} will have {} bytes'.format(fn, len(part)))
    print('- Tokenizing...')
    part_str = ' '.join([str(c) if c != ord('\n') else '\n' for c in part])
    print('- Writing...')
    f = open(fn, 'w').write(part_str)
    f = open(fn + '.raw', 'wb').write(part)
