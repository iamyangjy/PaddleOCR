#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pickle


def get_label_dict():
    f = open('./chinese_labels', 'rb')
    label_dict = pickle.load(f)
    f.close()
    return label_dict


if __name__ == "__main__":
    d = get_label_dict()
    print(d)
