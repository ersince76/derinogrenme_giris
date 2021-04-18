# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 12:29:17 2021

@author: eaksoy
"""

import re 
import numpy as np
import os

import locale

locale.setlocale(locale.LC_ALL, 'tr_TR.utf8')


def read_lines(file_name):
    result = []
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()

        for line in lines:
            if (len(line) > 15):
                result.append(line)
    return result


def read_from_dir(dir_path):
    result = []
    for k_file in os.listdir(dir_path):
        lines = read_lines(os.path.join(dir_path, k_file))
        result.extend(lines)
    return result


def split_to_sentences(input):
    result = []
    for line in input:
        for s in line.split("."):
            if len(s) > 15:
                result.append(s)
    return result


def format_text(text):
    text = text.lower().strip()
    result = re.sub("[^a-züğşçöıi]+", " ", text)
    return result


def format_text_list(list):
    result = []
    for line in list:
        result.append(format_text(line))
    return result


def prepare_triple_dict(list):
    result = dict()
    for line in list:
        triple_list = [line[i:i + 3] for i in range(0, len(line))]
        for t in triple_list:
            if t not in result:
                result[t] = len(result) + 1
    return result


def convert_triple_index(text, triple_dict):
    triple_list = [text[i:i + 3] for i in range(0, len(text))]
    result = []
    for t in triple_list:
        if t in triple_dict:
            result.append(triple_dict[t])
        else:
            result.append(0)
    return result


def list_convert_triple_index(lines, triple_dict):
    result = []
    for line in lines:
        result.append(convert_triple_index(line, triple_dict))
    return result

def one_hot_encoding(triple_line, triple_size):
    result = np.zeros(triple_size + 1)
    for t in triple_line:
        result[t-1] += 1
    result = result / triple_size
    result = result.reshape(1,len(result))
    return result

def list_one_hot_encoding(triple_line, triple_size):
    result = []
    for line in triple_line:
        result.append( one_hot_encoding(line, triple_size))
    return result


def load_trigram_dataset():
    k_list = read_from_dir("./data/karar")
    ks_list = split_to_sentences(k_list)
    fk_list = format_text_list(ks_list)

    b_list = read_from_dir("./data/birgun")
    bs_list = split_to_sentences(b_list)
    fb_list = format_text_list(bs_list)

    context = []
    context.extend(fb_list)
    context.extend(fk_list)

    triple_dictionary = prepare_triple_dict(context)

    karar_trp_list = list_convert_triple_index(fk_list,triple_dictionary)
    birgun_trp_list = list_convert_triple_index(fb_list,triple_dictionary)

    karar_ohe_list = list_one_hot_encoding(karar_trp_list,len(triple_dictionary))
    birgun_ohe_list = list_one_hot_encoding(birgun_trp_list, len(triple_dictionary))
    
    text = []
    text.extend(birgun_ohe_list)
    text.extend(karar_ohe_list)
    text = np.array( text )
    

    blabel =['birgun'] * len(birgun_ohe_list)
    klabel =['karar'] * len(karar_ohe_list)
    
    
    
    label = []
    label.extend(blabel)
    label.extend(klabel)
    
    label = np.array( label )

    return (triple_dictionary, text, label)

def prepare_triple_input(text,triple_dict):
    f_text = format_text(text)
    trp_line = convert_triple_index(f_text, triple_dict)
    result = one_hot_encoding(trp_line, len(triple_dict))
    #formatliyoruz
    result = np.array( result ).reshape( 1,len(triple_dict)+1)
    return result

def main():
    k_dir = "./data/karar"
    k_list = read_from_dir(k_dir)
    print("k cumle sayisi:" , len(k_list))

    print(k_list[0])
    ks_list = split_to_sentences(k_list)
    print(ks_list[0])

    fk_list = format_text_list(ks_list)
    print(fk_list[0])
    #print(fk_list)

    b_dir = "./data/birgun"
    b_list = read_from_dir(b_dir)
    print("b cumle sayisi:",len(b_list))

    print(b_list[0])
    bs_list = split_to_sentences(b_list)
    print(bs_list[0])

    fb_list = format_text_list(bs_list)
    print(fb_list[0])
    #print(fb_list)

    context = []
    context.extend(fb_list)
    context.extend(fk_list)
    print(context[0])

    triple_dictionary = prepare_triple_dict(context)
    print(triple_dictionary)

    print("fb_list[0]", fb_list[0])
    print(convert_triple_index(fb_list[0], triple_dictionary))
    print(convert_triple_index("işte gidiyorum çeşmi siyahımmmm", triple_dictionary))

    karar_trp_list = list_convert_triple_index(fk_list,triple_dictionary) 

    print(len(triple_dictionary))
    print(fk_list[0], " ", karar_trp_list[0])
    ohe = one_hot_encoding(karar_trp_list[0], len(triple_dictionary))
    print(ohe[0][615:620])

    ohe_list = list_one_hot_encoding([karar_trp_list[0]],len(triple_dictionary))
    print(ohe_list[0][0][615:620])



if __name__ == '__main__':
    main()
