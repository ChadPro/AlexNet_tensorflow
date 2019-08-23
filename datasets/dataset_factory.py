# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

from datasets import flowers17_dataset
from datasets import cifar10
from datasets import cifar100

datasets_map = {
    'flowers17_dataset' : flowers17_dataset,
    'cifar10' : cifar10,
    'cifar100' : cifar100
}

def get_dataset(name):
    if name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % name)
    return datasets_map[name]