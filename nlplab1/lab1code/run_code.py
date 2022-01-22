
from p2_SCORE import Score


from p4HMM import HMM
# import numpy as np
import re

sent_199801= '../io_file/199801_sent.txt'   #未分词的中文
SegPos_199801= '../io_file/199801_seg&pos.txt' #人工分词好的结果
train_SegPosPath='../io_file/train_test/train_seg&pos.txt' #训练集，前9/10
test='../io_file/test.txt' #测试集
test_compare='../io_file/train_test/test_compare.txt' #测试集需要对比的结果
dic_path= '../io_file/train_test/dict.txt'  #词典
result='../io_file/seg.txt'
FMMresult='../io_file/segment/seg_FMM.txt' #FMM分词后的结果
BMMresult='../io_file/segment/seg_BMM.txt' #BMM分词后的结果
score_path= '../io_file/score.txt'  #分数词典
answerpath='../io_file/HMMResult.txt'# hmm分词后的结果
k=10 #训练集与测试集大小只比

def test_p1code():
    '''
    测试划分训练集，测试集
    测试生成词典
    :return:
    '''
    from p1_DICT import createdict,load_dic_from_file
    createdict(k)
    wordlist,max_len=load_dic_from_file(train_SegPosPath, dic_path='../io_file/train_test/dict.txt')
    print(max_len, wordlist)

def test_p2code():
    '''
    测试最小代码的fmm，bmm算法，
    如果想要尽快得到答案，可以应用dict加速,取消代码注释即可
    :return:
    '''
    from p3FMM_BMM import get_dict,fmm,bmm
    get_dict('../io_file/train_test/dict.txt')
    fmm(max_len=26)
    precisionfmm, recallfmm, Ffmm = Score('../io_file/segment/seg_FMM.txt', '../io_file/train_test/test_compare.txt',
                                          '../io_file/score/score_FMM.txt',k)
    bmm(max_len=26)
    precisionbmm, recallbmm, Fbmm = Score('../io_file/segment/seg_BMM.txt', '../io_file/train_test/test_compare.txt',
                                          '../io_file/score/score_BMM.txt',k)

def test_p3code():
    '''
    测试进行优化性能后的fmm，bmm性能
    :return:
    '''
    from p3search import DicAction,strMarch
    import time
    root = DicAction.get_fmm_dic(dic_path='../io_file/train_test/dict.txt')
    start = time.time()
    strMarch.fmm(root, txt_path='../io_file/train_test/test.txt')
    end = time.time()
    print('fmm优化后的运行时间为：' + str(end - start))
    # 0.357957983016968
    precisionfmm, recallfmm, Ffmm = Score('../io_file/segment/seg_FMM_optimize.txt', '../io_file/train_test/test_compare.txt',
                                 '../io_file/score/score_FMM_optimize.txt',k)

    root = DicAction.get_bmm_dic()
    start = time.time()
    strMarch.bmm(root)
    end = time.time()
    print('bmm优化后的运行时间为：' + str(end - start))
    # 0.592636799812317
    precisionbmm, recallbmm, Fbmm = Score('../io_file/segment/seg_BMM_optimize.txt', '../io_file/train_test/test_compare.txt',
                                 '../io_file/score/score_BMM_optimize.txt',k)

def test_p4_unigram():
    '''
    测试一元文法分词
    :return:
    '''
    from p4_unigram import createDict
    createDict.gen_uni_dict()
    createDict.unigram()
    precisionbmm, recallbmm, Fbmm = Score('../io_file/segment/seg_Unigram.txt',
                                          '../io_file/train_test/test_compare.txt',
                                          '../io_file/score/score_Unigram.txt',k)

def test_p4_bigram():
    '''
    测试二元文法分词
    :return:
    '''
    from p4_bigram import createDict
    createDict.gen_bi_dict()
    createDict.bigram()
    precisionbmm, recallbmm, Fbmm = Score('../io_file/segment/seg_bigram.txt',
                                          '../io_file/train_test/test_compare.txt',
                                          '../io_file/score/score_bigram.txt',k)


def test_p4_hmm():
    '''
    测试纯hmm分词结果
    :return:
    '''
    from p4HMM import HMM
    h = HMM()
    h.run(test_compare, answerpath)
    # h.precision()
    precisionbmm, recallbmm, Fbmm = Score('../io_file/segment/hmm.txt',
                                          '../io_file/train_test/test_compare.txt',
                                          '../io_file/score/score_hmm.txt', k)

if __name__ == '__main__':
    # test_p1code()
    # test_p2code()
    # test_p3code()
     # test_p4_unigram()
     test_p4_bigram()
     #test_p4_hmm()