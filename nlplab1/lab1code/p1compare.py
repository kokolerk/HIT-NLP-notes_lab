from math import log
import Part_1
from Part_5_3 import HMM

Train_File = '../io_file/hmm/train.txt'  # ����ѵ���������ı��ļ�
Word_Freq = {}  # ���ڱ���ʵ��еĴʺʹ�Ƶ
Word_Num_Count = 0  # ��¼�ܴ���


class DicAction:

    # �������ߴʵ䲢��ñ�Ҫ�����ݽṹ�����ռȶ���ʽ��Լ
    @staticmethod
    def gene_uni_dic(train_path=Train_File, dic_path='../io_file/dic/uni_dic.txt'):
        global Word_Freq  # ���浽ȫ�ֱ�����
        with open(train_path, 'r', encoding='utf-8') as seg_file:
            lines = seg_file.readlines()
        for line in lines:
            for word in line.split():
                word = word[1 if word[0] == '[' else 0:word.index('/')]
                Word_Freq[word] = Word_Freq.get(word, 0) + 1
        Word_Freq = {k: Word_Freq[k] for k in sorted(Word_Freq.keys())}  # �Դʵ�����
        with open(dic_path, 'w', encoding='utf-8') as dic_file:
            for word in Word_Freq.keys():
                dic_file.write(word + ' ' + str(Word_Freq[word]) + '\n')
        Word_Freq = {}
        DicAction.get_uni_dic(dic_path)

    # �����ߴʵ乹�������ݽṹ��ǰ�������ߴʵ��Ѿ����ռȶ���ʽ��֯��
    @staticmethod
    def get_uni_dic(dic_path='../io_file/dic/uni_dic.txt'):
        global Word_Num_Count
        with open(dic_path, encoding='utf-8') as dic_file:  # ��ȡ���ߴʵ�
            lines = dic_file.readlines()
        for line in lines:
            word, freq = line.split()[0:2]  # ���ߴʵ�ÿ�е�����ͨ���ո�ָ�
            Word_Freq[word] = int(freq)  # ���ôʴ��뵽�ʵ���
            Word_Num_Count += int(freq)
            for count in range(1, len(word)):  # ��ȡ���ߴʵ���ÿ���ʵ�ǰ׺��
                prefix_word = word[:count]
                if prefix_word not in Word_Freq:  # ǰ׺����word_freq��
                    Word_Freq[prefix_word] = 0  # ����벢�ô�ƵΪ0

    # ͨ���������������ݽṹ�ʵ��������޻�ͼDAG
    @staticmethod
    def get_dag(line):
        dag = {}  # ���ڴ������յ�DAG
        n = len(line)  # ���ӳ���
        for k in range(n):  # ���������е�ÿһ����
            i = k
            dag[k] = []  # ��ʼ���洦�ڵ�k��λ���ϵ��ֵ�·�����
            word_fragment = line[k]
            while i < n and word_fragment in Word_Freq:  # ��kλ�ÿ�ʼ�Ĵʵ�����Ƭ���ڴʵ���
                if Word_Freq[word_fragment] > 0:  # �����ߴʵ��д��ڸô�
                    dag[k].append(i)  # ����Ƭ�μ��뵽��ʱ���б���
                i += 1
                word_fragment = line[k:i + 1]
            dag[k].append(k) if not dag[k] else dag[k]  # δ�ҵ�Ƭ�Σ��򽫵��ּ���
        return dag

    # �����ʷִʣ����ڸ������·������
    @staticmethod
    def calc_line_dag(line, dag):
        n = len(line)
        route = {n: (0, 0)}
        log_total = log(Word_Num_Count)
        for idx in range(n - 1, -1, -1):  # ��̬�滮�����·��
            route[idx] = max((log(Word_Freq.get(line[idx:x + 1], 0) or 1) - log_total +
                              route[x + 1][0], x) for x in dag[idx])
        return route

    # �������ı��ļ����������ʷִʣ�maximum word frequency segmentation
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
                seg_line = HMM.oov_line(seg_line) if seg_line else ''  # δ��¼��ʶ��
                mwf_file.write(seg_line + '\n')


## bigram
from math import log
import Part_5_1, Part_1
from Part_5_3 import HMM

Train_File = '../io_file/hmm/train.txt'  # ����ѵ���������ı��ļ�


class DicAction:
    words_dic = {}  # ��ʽ��{'��':'����':1,'��':'BOS':'3'}��ʾ'������'��'��Ĳ���'

    @staticmethod  # ѵ���ı�Ϊhmm�ļ����µ�train.txt���������ɶ�Ԫ�ķ��Ĵʵ�
    def gene_bi_dic(train_path=Train_File, dic_path='../io_file/dic/bigram_dic.txt'):
        with open(train_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            if line == '\n':
                continue
            words = line.split()  # һ�г�������õ��Ĵ����б�
            #����ͷβ���
            words.append('EOS/ ')
            words.insert(0, 'BOS')
            for idx in range(1, len(words)):
                words[idx] = words[idx][1 if words[idx][0] == '[' else 0:words[idx].index('/')]
                if words[idx] not in DicAction.words_dic.keys():
                    DicAction.words_dic[words[idx]] = {}
                if words[idx - 1] not in DicAction.words_dic[words[idx]].keys():
                    DicAction.words_dic[words[idx]][words[idx - 1]] = 0
                DicAction.words_dic[words[idx]][words[idx - 1]] += 1  # ���´�Ƶ
        DicAction.words_dic = {k: DicAction.words_dic[k] for k in
                               sorted(DicAction.words_dic.keys())}
        with open(dic_path, 'w', encoding='utf-8') as f:
            for word in DicAction.words_dic:
                DicAction.words_dic[word] = {k: DicAction.words_dic[word][k] for k in
                                             sorted(DicAction.words_dic[word].keys())}
                for pre in DicAction.words_dic[word]:
                    f.write(word + ' ' + pre + ' ' + str(DicAction.words_dic[word][pre]) + '\n')

    @staticmethod  # �����ߴʵ乹�������ݽṹ��ǰ�������ߴʵ��Ѿ����ռȶ���ʽ��֯��
    def get_bi_dic(dic_path='../io_file/dic/bigram_dic.txt'):
        with open(dic_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            word, pre_word, freq = line.split()[0:3]
            if word not in DicAction.words_dic.keys():
                DicAction.words_dic[word] = {pre_word: int(freq)}
            else:
                DicAction.words_dic[word][pre_word] = int(freq)

    @staticmethod  # ���ڼ���һ����֪��һ���ʵĴ�log����
    def get_log_pos(pre_word, word):
        #�����word_freq��ÿ���ʳ��ֵĴ�Ƶ
        #wordnumcount��һ�����ֵ��ܴ���
        pre_word_freq = Part_5_1.Word_Freq.get(pre_word, 0)  # ǰ�ʴ�Ƶ
        condition_word_freq = DicAction.words_dic.get(word, {}).get(pre_word, 0)  # ��ϴ�Ƶ
        #�������ĸ��ʣ��Ѿ������˲���ƽ���ˣ�
        return log(condition_word_freq + 1) - log(pre_word_freq + Part_5_1.Word_Num_Count)

    # �����ʷִʣ����ڸ������·������
    @staticmethod
    def calc_line_dag(line, dag):
        n = len(line) - 3  # ��ȥ<EOS>�ĳ���
        start = 3  # ����<BOS>�ӵ�һ���ֿ�ʼ
        pre_graph = {'BOS': {}}  # �ؼ���Ϊǰ�ʣ�ֵΪ��Ӧ�ĴʺͶ�������
        word_graph = {}  # ÿ���ʽڵ������һ�������ʵĴ�ͼ
        for x in dag[3]:  # ��ʼ��ǰ��ΪBOS�����
            pre_graph['BOS'][(3, x + 1)] = DicAction.get_log_pos('BOS', line[3:x + 1])
        print(pre_graph)
        while start < n:  # ��ÿһ���ֿ��ܵĴ�������һ���ʵĴʵ�
            for idx in dag[start]:  # ����dag[start]�е�ÿһ�������ڵ�
                pre_word = line[start:idx + 1]  # �������ǰһ���ʱ��磬'ȥ����'�е�ȥ
                temp = {}
                for next_end in dag[idx + 1]:
                    last_word = line[idx + 1:next_end + 1]
                    if line[idx + 1:next_end + 3] == 'EOS':  # �ж��Ƿ񵽴�ĩβ
                        temp['EOS'] = DicAction.get_log_pos(pre_word, 'EOS')
                    else:
                        temp[(idx + 1, next_end + 1)] = DicAction.get_log_pos(pre_word, last_word)
                pre_graph[(start, idx + 1)] = temp  # ÿһ����start��ʼ�Ĵʶ�����һ��������һ���ʵĴʵ�
            start += 1
        pre_words = list(pre_graph.keys())  # ��ʾ���е�ǰ���һ����
        for pre_word in pre_words:  # word_graph��ʾ�ؼ��ֶ�Ӧ��ֵΪ�ؼ��ֵ�ǰ���б�
            for word in pre_graph[pre_word].keys():  # ����pre_word�ʵĺ�һ����word
                word_graph[word] = word_graph.get(word, list())
                word_graph[word].append(pre_word)
        pre_words.append('EOS')
        route = {}
        for word in pre_words:
            if word == 'BOS':
                route[word] = (0.0, 'BOS')
            else:
                pre_list = word_graph.get(word, list())  # ȡ�øôʶ�Ӧ��ǰ���б�
                route[word] = (-65507, 'BOS') if not pre_list else max(
                    (pre_graph[pre][word] + route[pre][0], pre) for pre in pre_list)
        return route

    @staticmethod
    def bigram(txt_path=Part_1.Test_File, bigram_path='../io_file/seg/seg_bigram.txt'):
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        with open(bigram_path, 'w', encoding='utf-8') as bigram_file:
            for line in lines:
                #д����
                line = 'BOS' + line[:len(line) - 1] + 'EOS'
                #�����dag���Խ���part5��1�ĺ���ȥ��
                dag = Part_5_1.DicAction.get_dag(line)
                line_route = DicAction.calc_line_dag(line, dag)
                seg_line = ''
                position = 'EOS'
                while True:
                    position = line_route[position][1]
                    if position == 'BOS':
                        break
                    seg_line = line[position[0]:position[1]] + '/ ' + seg_line
                seg_line = HMM.oov_line(seg_line) if seg_line else ''  # δ��¼�ʴ���
                bigram_file.write(seg_line + '\n')  # д��ִ��ļ���
DicAction.gene_bi_dic()
DicAction.bigram()

##hmm
from math import log

import Part_1

Min = -3.14e+100  # ��ʾ��Сֵ
Pre_State = {'B': 'ES', 'M': 'MB', 'S': 'SE', 'E': 'BM'}  # ��ʾһ����ǵ�ǰһ�����ܵı��
Pi = {}  # ��ʼ״̬����
A = {}  # ����״̬ת�Ƹ���
B = {}  # ���ڷ�����ʼ���
States = ['B', 'M', 'E', 'S']  # ״̬�б�
State_Count = {}  # ����״̬���ִ���
Word_Count = 0  # ���ڼ����ܵĴ���
Word_Dic = set()  # �������еĴ�
Train_File = '../io_file/hmm/train.txt'  # ����ѵ���������ı��ļ�


class TRAIN:
    '''
    �ܶ���֮���Ǽ���pi��A,B�������˹���ע��bsme�ı�
    '''
    @staticmethod  # ��ʼ����ͳ�ƵĲ����˲��������еĴ�
    def init():
        global Word_Count, Word_Dic
        Word_Count = 0
        Word_Dic = set()
        for state in States:
            Pi[state] = 0.0
            State_Count[state] = 0
            B[state], A[state] = {}, {}
            for state_1 in States:
                A[state][state_1] = 0.0  # ��stateת��Ϊstate_1���ʳ�ʼ��

    @staticmethod  # ��ѵ���õ��Ĳ���д���ı��ļ��У������Ժ����
    def gene_hmm_dic(pi_path='../io_file/hmm/pi.txt', a_path='../io_file/hmm/a.txt',
                     b_path='../io_file/hmm/b.txt'):
        pi_file = open(pi_path, 'w', encoding='utf-8')
        a_file = open(a_path, 'w', encoding='utf-8')
        b_file = open(b_path, 'w', encoding='utf-8')
        for state in States:  # ������д���ı��ļ���
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


    @staticmethod  # ��ע��ȡ��һ�У�����һ�ж�Ӧ��״̬[B,M,E,S]
    def tag_line(line):
        global Word_Count
        line_word, line_tag = [], []  # ����ÿһ�е����е��ֺ�״̬[B,M,E,S]
        for word in line.split():
            word = word[1 if word[0] == '[' else 0:word.index('/')]  # ȡ��һ����
            line_word.extend(list(word))  # ����һ���ʵ�ÿ���ּӵ����е����б���
            Word_Dic.add(word)  # ���ôʱ�����Word_Dic��
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

    @staticmethod  # ͨ����������ı���ѵ����������������д���ı��ļ���
    def tag_txt(train_txt=Train_File):
        TRAIN.init()  # ��ʼ������
        with open(train_txt, 'r', encoding='utf-8') as txt_f:
            lines = txt_f.readlines()
        for line in lines:
            if line == '\n':
                continue
            line_word, line_tag = TRAIN.tag_line(line)  # �õ�һ�б�ע
            print(line_word)
            for i in range(len(line_tag)):
                State_Count[line_tag[i]] += 1  # ����״̬�ܳ��ִ���
                B[line_tag[i]][line_word[i]] = B[line_tag[i]].get(line_word[i], 0) + 1
                if i > 0:
                    A[line_tag[i - 1]][line_tag[i]] += 1  # ת�Ƹ��ʱ仯
        for state in States:
            Pi[state] = Min if Pi[state] == 0 else log(Pi[state] / Word_Count)
            for state_1 in States:  # ����״̬ת�Ƹ���
                A[state][state_1] = Min if A[state][state_1] == 0 else log(
                    A[state][state_1] / State_Count[state])
            for word in B[state].keys():  # ���㷢�����
                B[state][word] = log(B[state][word] / State_Count[state])
        TRAIN.gene_hmm_dic()  # ������д���ı��ļ���

    @staticmethod  # ���ڴ��ļ��ж�ȡѵ���õĲ��������ȶ���ʽ�����ı����ݣ�����Ҫ���ʽһ�£�
    def get_para(pi_path='../io_file/hmm/pi.txt', a_path='../io_file/hmm/a.txt',
                 b_path='../io_file/hmm/b.txt', word_path='../io_file/dic/uni_dic.txt'):
        TRAIN.init()
        pi_lines = open(pi_path, 'r', encoding='utf-8').readlines()
        a_lines = open(a_path, 'r', encoding='utf-8').readlines()
        b_lines = open(b_path, 'r', encoding='utf-8').readlines()
        word_lines = open(word_path, 'r', encoding='utf-8').readlines()  # ��Unigram�ʵ��ж�ȡ���еĴ�
        for word in word_lines:
            Word_Dic.add(word.split()[0])
        for idx in range(4):  # ����Pi����
            pi_state, pi_pos = pi_lines[idx].split()[0:2]
            Pi[pi_state] = float(pi_pos)
        for idx in range(20):  # ����A����
            if idx % 5 != 0:
                A[States[int(idx / 5)]][States[idx % 5 - 1]] = float(a_lines[idx].split()[1])
        state = 'B'
        for idx in range(len(b_lines) - 1):  # ����B����(���һ��Ϊ����)
            if b_lines[idx][0] != ' ':  # ��ʼ��ȡһ��״̬��Ӧ�ĵ���
                state = b_lines[idx][0]  # ��¼��״̬
            else:
                word, pos = b_lines[idx].split()[0:2]
                B[state][word] = float(pos)


class HMM:
    '''
    ���ǱȽ�Ӳ�˵ģ��й�hmm�㷨��ʵ��
    '''
    @staticmethod  # ����һ�У�����Ϊw1/ w2 /w3 /w4w5/ ���������w1w2/ w3/ w4w5/ .
    def oov_line(seg_line, choice=True):
        word_list = seg_line[:len(seg_line) - 2].split('/ ')  # �õ����еĴ����б�
        # ���������ĵ��֣����ڽ�һ��ʹ��HMM����δ��¼��ʶ��
        seg_line, to_seg_word = '', ''
        for idx in range(len(word_list)):
            # ����һ�����֣�������뵽������������
            if len(word_list[idx]) == 1:
                # �õ����Ǵʵ��еĴ�
                if choice and word_list[idx] in Word_Dic:
                    if to_seg_word:  # �ж��Ƿ�ô�ǰ��Ϊ����
                        seg_line += HMM.oov_word(to_seg_word)
                        to_seg_word = ''
                    seg_line += word_list[idx] + '/ '
                # �õ��ֲ��Ǵʵ��еĴ�
                else:
                    to_seg_word += word_list[idx]
                    if idx + 1 == len(word_list):
                        seg_line += HMM.oov_word(to_seg_word)
            # �����ǵ������
            else:
                # �ж��Ƿ�ô�ǰ��Ϊ����
                if to_seg_word:
                    seg_line += HMM.oov_word(to_seg_word)
                    to_seg_word = ''
                seg_line += word_list[idx] + '/ '
        return seg_line

    @staticmethod  # �����������ĵ����������磺w1w2w3����Ӧ������磺w1/ w2w3/ .

    def oov_word(to_seg_word):
        # �����������ʽ�Ϊһ���֣�ֱ�ӷ���
        if len(to_seg_word) == 1:
            return to_seg_word + '/ '

        tag_list = HMM.viterbi(to_seg_word)[1]
        begin, next_i = 0, 0
        res_word = ''  # �����to_seg_word�ִʽ��
        for idx, char in enumerate(to_seg_word):
            tag = tag_list[idx]  # ȡһ��tag
            if tag == 'B':  # ��ʾ��ʼ
                begin = idx
            elif tag == 'E':  # ��ʾ����
                res_word += to_seg_word[begin:idx + 1] + '/ '
                next_i = idx + 1
            elif tag == 'S':
                res_word += char + '/ '
                next_i = idx + 1
        if next_i < len(to_seg_word):
            res_word += to_seg_word[next_i:] + '/ '
        return res_word

    @staticmethod  # ����Ϊw1w2w3
    def viterbi(to_seg_word):
        v = [{}]
        path = {}
        for state in States:  # ��ʼ��
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

    @staticmethod  # ����ʹ��HMM�ִ�
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