from math import log
import Part_1
from Part_5_3 import HMM

Train_File = '../io_file/hmm/train.txt'  # 用于训练参数的文本文件
Word_Freq = {}  # 用于保存词典中的词和词频
Word_Num_Count = 0  # 记录总词数


class DicAction:

    # 构建离线词典并获得必要的数据结构，按照既定格式规约
    @staticmethod
    def gene_uni_dic(train_path=Train_File, dic_path='../io_file/dic/uni_dic.txt'):
        global Word_Freq  # 保存到全局变量中
        with open(train_path, 'r', encoding='utf-8') as seg_file:
            lines = seg_file.readlines()
        for line in lines:
            for word in line.split():
                word = word[1 if word[0] == '[' else 0:word.index('/')]
                Word_Freq[word] = Word_Freq.get(word, 0) + 1
        Word_Freq = {k: Word_Freq[k] for k in sorted(Word_Freq.keys())}  # 对词典排序
        with open(dic_path, 'w', encoding='utf-8') as dic_file:
            for word in Word_Freq.keys():
                dic_file.write(word + ' ' + str(Word_Freq[word]) + '\n')
        Word_Freq = {}
        DicAction.get_uni_dic(dic_path)

    # 从离线词典构建其数据结构，前提是离线词典已经按照既定格式组织好
    @staticmethod
    def get_uni_dic(dic_path='../io_file/dic/uni_dic.txt'):
        global Word_Num_Count
        with open(dic_path, encoding='utf-8') as dic_file:  # 读取离线词典
            lines = dic_file.readlines()
        for line in lines:
            word, freq = line.split()[0:2]  # 离线词典每行的属性通过空格分隔
            Word_Freq[word] = int(freq)  # 将该词存入到词典中
            Word_Num_Count += int(freq)
            for count in range(1, len(word)):  # 获取离线词典中每个词的前缀词
                prefix_word = word[:count]
                if prefix_word not in Word_Freq:  # 前缀不在word_freq中
                    Word_Freq[prefix_word] = 0  # 则存入并置词频为0

    # 通过构建的在线数据结构词典获得有向无环图DAG
    @staticmethod
    def get_dag(line):
        dag = {}  # 用于储存最终的DAG
        n = len(line)  # 句子长度
        for k in range(n):  # 遍历句子中的每一个字
            i = k
            dag[k] = []  # 开始保存处于第k个位置上的字的路径情况
            word_fragment = line[k]
            while i < n and word_fragment in Word_Freq:  # 以k位置开始的词的所在片段在词典中
                if Word_Freq[word_fragment] > 0:  # 若离线词典中存在该词
                    dag[k].append(i)  # 将该片段加入到临时的列表中
                i += 1
                word_fragment = line[k:i + 1]
            dag[k].append(k) if not dag[k] else dag[k]  # 未找到片段，则将单字加入
        return dag

    # 最大概率分词，用于概率最大路径计算
    @staticmethod
    def calc_line_dag(line, dag):
        n = len(line)
        route = {n: (0, 0)}
        log_total = log(Word_Num_Count)
        for idx in range(n - 1, -1, -1):  # 动态规划求最大路径
            route[idx] = max((log(Word_Freq.get(line[idx:x + 1], 0) or 1) - log_total +
                              route[x + 1][0], x) for x in dag[idx])
        return route

    # 对输入文本文件进行最大概率分词：maximum word frequency segmentation
    @staticmethod
    def mwf(txt_path=Part_1.Test_File, mwf_path='../io_file/seg/seg_mwf.txt'):
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        with open(mwf_path, 'w', encoding='utf-8') as mwf_file:
            for line in lines:
                line = line[:len(line) - 1]
                line_route = DicAction.calc_line_dag(line, DicAction.get_dag(line))
                old_start = 0
                seg_line = ''
                while old_start < len(line):
                    new_start = line_route[old_start][1] + 1
                    seg_line += line[old_start:new_start] + '/ '
                    old_start = new_start
                seg_line = HMM.oov_line(seg_line) if seg_line else ''  # 未登录词识别
                mwf_file.write(seg_line + '\n')


## bigram
from math import log
import Part_5_1, Part_1
from Part_5_3 import HMM

Train_File = '../io_file/hmm/train.txt'  # 用于训练参数的文本文件


class DicAction:
    words_dic = {}  # 格式：{'玩':'北京':1,'玩':'BOS':'3'}表示'北京玩'和'玩的不错'

    @staticmethod  # 训练文本为hmm文件夹下的train.txt，用于生成二元文法的词典
    def gene_bi_dic(train_path=Train_File, dic_path='../io_file/dic/bigram_dic.txt'):
        with open(train_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            if line == '\n':
                continue
            words = line.split()  # 一行初步处理得到的词语列表
            #插入头尾标记
            words.append('EOS/ ')
            words.insert(0, 'BOS')
            for idx in range(1, len(words)):
                words[idx] = words[idx][1 if words[idx][0] == '[' else 0:words[idx].index('/')]
                if words[idx] not in DicAction.words_dic.keys():
                    DicAction.words_dic[words[idx]] = {}
                if words[idx - 1] not in DicAction.words_dic[words[idx]].keys():
                    DicAction.words_dic[words[idx]][words[idx - 1]] = 0
                DicAction.words_dic[words[idx]][words[idx - 1]] += 1  # 更新词频
        DicAction.words_dic = {k: DicAction.words_dic[k] for k in
                               sorted(DicAction.words_dic.keys())}
        with open(dic_path, 'w', encoding='utf-8') as f:
            for word in DicAction.words_dic:
                DicAction.words_dic[word] = {k: DicAction.words_dic[word][k] for k in
                                             sorted(DicAction.words_dic[word].keys())}
                for pre in DicAction.words_dic[word]:
                    f.write(word + ' ' + pre + ' ' + str(DicAction.words_dic[word][pre]) + '\n')

    @staticmethod  # 从离线词典构建其数据结构，前提是离线词典已经按照既定格式组织好
    def get_bi_dic(dic_path='../io_file/dic/bigram_dic.txt'):
        with open(dic_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            word, pre_word, freq = line.split()[0:3]
            if word not in DicAction.words_dic.keys():
                DicAction.words_dic[word] = {pre_word: int(freq)}
            else:
                DicAction.words_dic[word][pre_word] = int(freq)

    @staticmethod  # 用于计算一个已知上一个词的词log概率
    def get_log_pos(pre_word, word):
        #这里的word_freq是每个词出现的词频
        #wordnumcount是一共出现的总次数
        pre_word_freq = Part_5_1.Word_Freq.get(pre_word, 0)  # 前词词频
        condition_word_freq = DicAction.words_dic.get(word, {}).get(pre_word, 0)  # 组合词频
        #这里计算的概率，已经进行了参数平滑了！
        return log(condition_word_freq + 1) - log(pre_word_freq + Part_5_1.Word_Num_Count)

    # 最大概率分词，用于概率最大路径计算
    @staticmethod
    def calc_line_dag(line, dag):
        n = len(line) - 3  # 减去<EOS>的长度
        start = 3  # 跳过<BOS>从第一个字开始
        pre_graph = {'BOS': {}}  # 关键字为前词，值为对应的词和对数概率
        word_graph = {}  # 每个词节点存有上一个相连词的词图
        for x in dag[3]:  # 初始化前词为BOS的情况
            pre_graph['BOS'][(3, x + 1)] = DicAction.get_log_pos('BOS', line[3:x + 1])
        print(pre_graph)
        while start < n:  # 对每一个字可能的词生成下一个词的词典
            for idx in dag[start]:  # 遍历dag[start]中的每一个结束节点
                pre_word = line[start:idx + 1]  # 这个词是前一个词比如，'去北京'中的去
                temp = {}
                for next_end in dag[idx + 1]:
                    last_word = line[idx + 1:next_end + 1]
                    if line[idx + 1:next_end + 3] == 'EOS':  # 判断是否到达末尾
                        temp['EOS'] = DicAction.get_log_pos(pre_word, 'EOS')
                    else:
                        temp[(idx + 1, next_end + 1)] = DicAction.get_log_pos(pre_word, last_word)
                pre_graph[(start, idx + 1)] = temp  # 每一个以start开始的词都建立一个关于下一个词的词典
            start += 1
        pre_words = list(pre_graph.keys())  # 表示所有的前面的一个词
        for pre_word in pre_words:  # word_graph表示关键字对应的值为关键字的前词列表
            for word in pre_graph[pre_word].keys():  # 遍历pre_word词的后一个词word
                word_graph[word] = word_graph.get(word, list())
                word_graph[word].append(pre_word)
        pre_words.append('EOS')
        route = {}
        for word in pre_words:
            if word == 'BOS':
                route[word] = (0.0, 'BOS')
            else:
                pre_list = word_graph.get(word, list())  # 取得该词对应的前词列表
                route[word] = (-65507, 'BOS') if not pre_list else max(
                    (pre_graph[pre][word] + route[pre][0], pre) for pre in pre_list)
        return route

    @staticmethod
    def bigram(txt_path=Part_1.Test_File, bigram_path='../io_file/seg/seg_bigram.txt'):
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        with open(bigram_path, 'w', encoding='utf-8') as bigram_file:
            for line in lines:
                #写文字
                line = 'BOS' + line[:len(line) - 1] + 'EOS'
                #这里的dag可以借用part5、1的函数去做
                dag = Part_5_1.DicAction.get_dag(line)
                line_route = DicAction.calc_line_dag(line, dag)
                seg_line = ''
                position = 'EOS'
                while True:
                    position = line_route[position][1]
                    if position == 'BOS':
                        break
                    seg_line = line[position[0]:position[1]] + '/ ' + seg_line
                seg_line = HMM.oov_line(seg_line) if seg_line else ''  # 未登录词处理
                bigram_file.write(seg_line + '\n')  # 写入分词文件中
DicAction.gene_bi_dic()
DicAction.bigram()

##hmm
from math import log

import Part_1

Min = -3.14e+100  # 表示最小值
Pre_State = {'B': 'ES', 'M': 'MB', 'S': 'SE', 'E': 'BM'}  # 表示一个标记的前一个可能的标记
Pi = {}  # 初始状态集Π
A = {}  # 用于状态转移概率
B = {}  # 用于发射概率计算
States = ['B', 'M', 'E', 'S']  # 状态列表
State_Count = {}  # 保存状态出现次数
Word_Count = 0  # 用于计算总的词数
Word_Dic = set()  # 保存所有的词
Train_File = '../io_file/hmm/train.txt'  # 用于训练参数的文本文件


class TRAIN:
    '''
    总而言之，是计算pi，A,B，生成人工标注的bsme文本
    '''
    @staticmethod  # 初始化待统计的参数λ并配置所有的词
    def init():
        global Word_Count, Word_Dic
        Word_Count = 0
        Word_Dic = set()
        for state in States:
            Pi[state] = 0.0
            State_Count[state] = 0
            B[state], A[state] = {}, {}
            for state_1 in States:
                A[state][state_1] = 0.0  # 由state转换为state_1概率初始化

    @staticmethod  # 将训练得到的参数写入文本文件中，便于以后分析
    def gene_hmm_dic(pi_path='../io_file/hmm/pi.txt', a_path='../io_file/hmm/a.txt',
                     b_path='../io_file/hmm/b.txt'):
        pi_file = open(pi_path, 'w', encoding='utf-8')
        a_file = open(a_path, 'w', encoding='utf-8')
        b_file = open(b_path, 'w', encoding='utf-8')
        for state in States:  # 将参数写入文本文件中
            # pi_file.write(state + ' ' + str(Pi[state]) + '\n')
            # a_file.write(state + '\n')
            # b_file.write(state + '\n')
            # for state_1 in States:
            #     a_file.write(' ' + state_1 + ' ' + str(A[state][state_1]) + '\n')
            # for word in B[state].keys():
            #     b_file.write(' ' + word + ' ' + str(B[state][word]) + '\n')
            pi_file.write(state + '/' + str(Pi[state]) + '\n')
            for state_1 in States:
                a_file.write(state+'/' + state_1 + '/' + str(A[state][state_1]) + '/\n')
            for word in B[state].keys():
                b_file.write(state+'/' + word + '/' + str(B[state][word]) + '/\n')


    @staticmethod  # 标注读取的一行，返回一行对应的状态[B,M,E,S]
    def tag_line(line):
        global Word_Count
        line_word, line_tag = [], []  # 保存每一行的所有单字和状态[B,M,E,S]
        for word in line.split():
            word = word[1 if word[0] == '[' else 0:word.index('/')]  # 取出一个词
            line_word.extend(list(word))  # 将这一个词的每个字加到该行的字列表中
            Word_Dic.add(word)  # 将该词保存在Word_Dic中
            Word_Count += 1
            if len(word) == 1:
                line_tag.append('S')
                Pi['S'] += 1
            else:
                line_tag.append('B')
                line_tag.extend(['M'] * (len(word) - 2))
                line_tag.append('E')
                Pi['B'] += 1
        return line_word, line_tag

    @staticmethod  # 通过标记整个文本来训练参数，并将参数写入文本文件中
    def tag_txt(train_txt=Train_File):
        TRAIN.init()  # 初始化参数
        with open(train_txt, 'r', encoding='utf-8') as txt_f:
            lines = txt_f.readlines()
        for line in lines:
            if line == '\n':
                continue
            line_word, line_tag = TRAIN.tag_line(line)  # 得到一行标注
            print(line_word)
            for i in range(len(line_tag)):
                State_Count[line_tag[i]] += 1  # 计算状态总出现次数
                B[line_tag[i]][line_word[i]] = B[line_tag[i]].get(line_word[i], 0) + 1
                if i > 0:
                    A[line_tag[i - 1]][line_tag[i]] += 1  # 转移概率变化
        for state in States:
            Pi[state] = Min if Pi[state] == 0 else log(Pi[state] / Word_Count)
            for state_1 in States:  # 计算状态转移概率
                A[state][state_1] = Min if A[state][state_1] == 0 else log(
                    A[state][state_1] / State_Count[state])
            for word in B[state].keys():  # 计算发射概率
                B[state][word] = log(B[state][word] / State_Count[state])
        TRAIN.gene_hmm_dic()  # 将参数写入文本文件中

    @staticmethod  # 用于从文件中读取训练好的参数并按既定格式解析文本内容（必须要求格式一致）
    def get_para(pi_path='../io_file/hmm/pi.txt', a_path='../io_file/hmm/a.txt',
                 b_path='../io_file/hmm/b.txt', word_path='../io_file/dic/uni_dic.txt'):
        TRAIN.init()
        pi_lines = open(pi_path, 'r', encoding='utf-8').readlines()
        a_lines = open(a_path, 'r', encoding='utf-8').readlines()
        b_lines = open(b_path, 'r', encoding='utf-8').readlines()
        word_lines = open(word_path, 'r', encoding='utf-8').readlines()  # 从Unigram词典中读取所有的词
        for word in word_lines:
            Word_Dic.add(word.split()[0])
        for idx in range(4):  # 配置Pi参数
            pi_state, pi_pos = pi_lines[idx].split()[0:2]
            Pi[pi_state] = float(pi_pos)
        for idx in range(20):  # 配置A参数
            if idx % 5 != 0:
                A[States[int(idx / 5)]][States[idx % 5 - 1]] = float(a_lines[idx].split()[1])
        state = 'B'
        for idx in range(len(b_lines) - 1):  # 配置B参数(最后一行为空行)
            if b_lines[idx][0] != ' ':  # 开始读取一个状态对应的单字
                state = b_lines[idx][0]  # 记录该状态
            else:
                word, pos = b_lines[idx].split()[0:2]
                B[state][word] = float(pos)


class HMM:
    '''
    这是比较硬核的，有关hmm算法的实现
    '''
    @staticmethod  # 处理一行，输入为w1/ w2 /w3 /w4w5/ ，输出形如w1w2/ w3/ w4w5/ .
    def oov_line(seg_line, choice=True):
        word_list = seg_line[:len(seg_line) - 2].split('/ ')  # 得到所有的词语列表
        # 储存连续的单字，用于进一步使用HMM进行未登录词识别
        seg_line, to_seg_word = '', ''
        for idx in range(len(word_list)):
            # 遇到一个单字，将其加入到待处理序列中
            if len(word_list[idx]) == 1:
                # 该单字是词典中的词
                if choice and word_list[idx] in Word_Dic:
                    if to_seg_word:  # 判断是否该词前面为单字
                        seg_line += HMM.oov_word(to_seg_word)
                        to_seg_word = ''
                    seg_line += word_list[idx] + '/ '
                # 该单字不是词典中的词
                else:
                    to_seg_word += word_list[idx]
                    if idx + 1 == len(word_list):
                        seg_line += HMM.oov_word(to_seg_word)
            # 遇到非单字情况
            else:
                # 判断是否该词前面为单字
                if to_seg_word:
                    seg_line += HMM.oov_word(to_seg_word)
                    to_seg_word = ''
                seg_line += word_list[idx] + '/ '
        return seg_line

    @staticmethod  # 处理几个连续的单字输入形如：w1w2w3，对应结果形如：w1/ w2w3/ .

    def oov_word(to_seg_word):
        # 若传输待处理词仅为一个字，直接返回
        if len(to_seg_word) == 1:
            return to_seg_word + '/ '

        tag_list = HMM.viterbi(to_seg_word)[1]
        begin, next_i = 0, 0
        res_word = ''  # 输入的to_seg_word分词结果
        for idx, char in enumerate(to_seg_word):
            tag = tag_list[idx]  # 取一个tag
            if tag == 'B':  # 表示开始
                begin = idx
            elif tag == 'E':  # 表示结束
                res_word += to_seg_word[begin:idx + 1] + '/ '
                next_i = idx + 1
            elif tag == 'S':
                res_word += char + '/ '
                next_i = idx + 1
        if next_i < len(to_seg_word):
            res_word += to_seg_word[next_i:] + '/ '
        return res_word

    @staticmethod  # 输入为w1w2w3
    def viterbi(to_seg_word):
        v = [{}]
        path = {}
        for state in States:  # 初始化
            v[0][state] = Pi[state] + B[state].get(to_seg_word[0], Min)
            path[state] = [state]
        for idx in range(1, len(to_seg_word)):
            v.append({})
            new_path = {}
            for state_1 in States:
                em_p = B[state_1].get(to_seg_word[idx], Min)
                (prob, state) = max([(v[idx - 1][y0] + A[y0].get(state_1, Min) + em_p, y0) for y0 in
                                     Pre_State[state_1]])
                v[idx][state_1] = prob
                new_path[state_1] = path[state] + [state_1]
            path = new_path
        (prob, state) = max((v[len(to_seg_word) - 1][y], y) for y in 'ES')
        return prob, path[state]

    @staticmethod  # 仅仅使用HMM分词
    def hmm(txt_path=Part_1.Test_File, hmm_path='../io_file/seg/seg_hmm.txt'):
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        with open(hmm_path, 'w', encoding='utf-8') as hmm_file:
            for line in lines:
                new_line = ''
                for word in line[0:len(line) - 1]:
                    new_line += word + '/ '
                new_line = HMM.oov_line(new_line, False) if new_line else ''
                hmm_file.write(new_line + '\n')
TRAIN.tag_txt()