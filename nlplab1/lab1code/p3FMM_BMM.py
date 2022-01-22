import re
import numpy as np
from p1_DICT import *
import time
dictory={}
#制作词典
def get_dict(dict_path):
    with open(dict_path,'r',encoding='utf-8') as f:
        for line in f:
            dictory[line[:len(line)-1]]=0

def isword(word):
    return word in dictory

def fmm(max_len,txt_path='../io_file/train_test/test.txt', fmm_path='../io_file/seg_FMM.txt'):
    '''
    :param max_len: 最大词长度
    :param txt_path: 需要分词的测试文件的地址
    :param fmm_path: 分词后的结果文件的地址
    :return:
    '''
    seg_result = ''
    file = open(txt_path, 'r', encoding='utf-8')
    for line in file:
        if line == '\n':
            continue
        seg_line, line = '', line[:len(line) - 1]  # 去掉读取的换行符
        while len(line) > 0:
            count =max_len
            terminal_word = line[0]  # 单个词
            while count >1: #找多字词
                if isword(line[:count if count<len(line) else len(line)]):
                    terminal_word=line[:count]
                    break
                else:
                    count=count-1
            line = line[len(terminal_word):]
            seg_line += terminal_word + '/'
        seg_line = pre_line(seg_line)  # 将数字、字母、字符之类的连起来
        seg_result += seg_line + '\n'
    open(fmm_path, 'w', encoding='UTF-8').write(seg_result)


#最大反向匹配
def bmm(max_len,txt_path='../io_file/train_test/test.txt', bmm_path='../io_file/seg_BMM.txt'):
    '''

    :param max_len:
    :param txt_path:
    :param fmm_path:
    :return:
    '''
    seg_result = ''
    file = open(txt_path, 'r', encoding='utf-8')
    for line in file:
        if line == '\n':
            continue
        seg_line, line = '', line[:len(line) - 1]  # 去掉读取的换行符
        while len(line) > 0:
            count = max_len
            terminal_word = line[len(line)-1]  # 单个词
            while count > 1:  # 找多字词
                if isword(line[- ( count if count < len(line) else len(line)):]):
                    terminal_word = line[-count:]
                    break
                else:
                    count = count - 1
            line = line[:len(line)-count]
            seg_line = terminal_word + '/'+seg_line
        seg_line = pre_line(seg_line)  # 将数字、字母、字符之类的连起来
        seg_result += seg_line + '\n'
    open(bmm_path, 'w', encoding='UTF-8').write(seg_result)


