import re
import numpy as np
sent_199801= '../io_file/199801_sent.txt'   #未分词的中文
SegPos_199801= '../io_file/199801_seg&pos.txt' #人工分词好的结果
train_SegPosPath='../io_file/train_test/train_seg&pos.txt' #训练集，前9/10
test='../io_file/train_test/test.txt' #测试集
test_compare='../io_file/train_test/test_compare.txt' #测试集需要对比的结果

#区分训练集和测试集
def createdict(K,segPosPath=SegPos_199801,
               sentpath=sent_199801,
               segpath_for_dict=train_SegPosPath,
               sentpath_for_dict=test,
               test_compare=test_compare):
    '''
    生成9/10为训练集和1/10为测试集
    :param segPosPath: 原标记文本
    :param sentpath: 原测试文本
    :param segpath_for_dict: 生成的训练集——标记文本
    :param sentpath_for_dict: 生成的测试集——未标记文本
    :param test_compare:生成的测试集——标记文本
    :param K :比例，生成的训练集和测试集的数量比为K：1
    :return:
    '''

    with open(segPosPath,'r',encoding='GBK')as f1:
        segPos=f1.readlines()
    train=[]
    testSegPos=[]
    for idx,line in enumerate(segPos):
        if idx%K==0:
            if line=='\n':
                continue
            else:
                testSegPos.append(line)
        else:train.append(line)
    open(segpath_for_dict,'w',encoding='utf-8').write(''.join(train))
    open(test_compare,'w',encoding='utf-8').write(''.join(testSegPos))
    test=[]
    with open(sentpath,'r',encoding='gbk') as f2:
        sent=f2.readlines()
    for idx,line in enumerate(sent):
        if idx%K==0:
            if line=='\n':
                continue
            else:test.append(line)
    open(sentpath_for_dict,'w',encoding='utf-8').write(''.join(test))

#制作词典
def load_dic_from_file(train_path,dic_path):
    '''
    制作词典
    :param train_path: 训练集
    :param dic_path: 词典位置
    :return:
    word_list:词表
    max_len:最大词长度
    '''
    max_len, word_set = 0, set()  # 保存最大词长，保存所有的词，要求具有唯一性且可排序
    with open(train_path, 'r', encoding='utf-8') as seg_file:
         lines = seg_file.readlines()  # 读取训练文本
    with open(dic_path, 'w', encoding='utf-8') as dic_file:
        for line in lines:
            for word in line.split():
                if '/m' in word:  # 不考虑将将量词加入
                    continue
                word = word[1 if word[0] == '[' else 0:word.index('/')]  # 去掉两个空格之间的非词字符
                word_set.add(word)  # 将词加入词典
                max_len = len(word) if len(word) > max_len else max_len  # 更新最大词长
        word_list = list(word_set)
        word_list.sort()  # 对此列表进行排序
        dic_file.write('\n'.join(word_list))  # 一个词一行
    return word_list, max_len


#可视化比较
def compare_differ(f1,f2,compare_path):
    '''
    比较两个文件不同的地方，按行比较
    :param f1:
    :param f2:
    :param compare_path:
    :return:
    '''
    temp=[]
    with open(f1, 'r', encoding='utf-8') as file1:
        lines1 = file1.readlines()
    with open(f2, 'r', encoding='utf-8') as file2:
        lines2 = file2.readlines()
    for idx,line1 in enumerate(lines1):
        line2=lines2[idx]
        if line1==line2:
            continue
        else:
            temp.append(line1)
            temp.append(line2)
    open(compare_path,'w',encoding='utf-8').write(''.join(temp))

def pre_line(line):
    punc='/-.'
    buffer=''
    result=''
    word_list=line.split('/')
    word_list=word_list[:len(word_list)-1]
    for idx,word in enumerate(word_list):
        if word.isascii() or word in punc:
            buffer +=word
            if idx==len(word_list)-1:
                result +=buffer+'/'
        else:
            if buffer:
                result +=buffer+'/'
                buffer=''
            result +=word+'/'
    return result


def temp(filepath,path='../io_file/train_test/test_compare.txt'):
    file=open(filepath,'w',encoding='utf-8')
    with open(path, 'r', encoding='utf-8') as seg_file:
        lines = seg_file.readlines()  # 读取训练文本
    result=''
    for line in lines:
        line=line[:len(line)-1]
        for word in line.split():
            word = word[1 if word[0] == '[' else 0:word.index('/')]  # 去掉两个空格之间的非词字符
            result +=word+'/'
        result +='\n'
    file.write(result)  # 一个词一行

