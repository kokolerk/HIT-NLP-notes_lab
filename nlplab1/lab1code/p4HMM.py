import re
from math import log
train_path='../io_file/199801_seg&pos.txt.txt' #训练集
HMM_train_tag_path='../io_file/hmm/HMM_train_tag.txt' #训练集标记
answerpath='../io_file/hmm/HMM_train.txt'  #hmm结果
test= '../io_file/train_test/test.txt'   #测试集
test_compare='../io_file/train_test/test_compare.txt' #测试集对比答案
test_tag_path='../io_file/hmm/HMM_test_tag.txt'  #测试集人工标注的结果
hmm_pi='../io_file/hmm/pi.txt'
hmm_A='../io_file/hmm/A.txt'
hmm_B='../io_file/hmm/B.txt'
status=['B','M','E','S']    #开始，中间，结尾，单字
array_A = {}    #状态转移概率矩阵，从某个状态转移到另一个状态的概率，比如B到S，概率为0.1
array_B = {}    #发射概率矩阵
array_E = {}    #测试集存在的字符，但在训练集中不存在，发射概率矩阵
array_Pi = {}   #初始状态分布，初始化概率,比如一开始是B的概率为0.5
word_set = set()    #训练数据集中所有字的集合
count_dic = {}  #‘B,M,E,S’每个状态在训练集中出现的次数
line_num = 0    #训练集语句数量


class HMM:
    #初始化pi，A , B, Pi
    def __init__(self):
        '''
        初始化A,B,PI矩阵
        '''
        for s0 in status:
            array_A[s0]={}
            array_B[s0]={}
            array_Pi[s0]=0.0
            count_dic[s0]=0
            for s1 in status:
                array_A[s0][s1]=0.0

    #计算A , B, Pi的概率转移矩阵,并保存到文件里面
    def write_prob_array_to_file(self):
        '''
        将参数估计的概率取对数，对概率0取无穷小-3.14e+100
        :return:
        '''
        # 初始状态
        pi=open(hmm_pi,'w',encoding='utf-8')
        B=open(hmm_B,'w',encoding='utf-8')
        A=open(hmm_A,'w',encoding='utf-8')
        Min = -3.14e+100
        for state in status:
            array_Pi[state] = Min if array_Pi[state] == 0 else log(array_Pi[state] / line_num)
            pi.write(state + '/' + str(array_Pi[state]) + '/' + '\n')
            for state_1 in status:  # 计算状态转移概率
                array_A[state][state_1] = Min if array_A[state][state_1] == 0 else log(
                    array_A[state][state_1] / count_dic[state])
                A.write(state + '/' + state_1 + '/' + str(array_A[state][state_1]) + '/' + '\n')
            for word in array_B[state].keys():  # 计算发射概率
                array_B[state][word] = log(array_B[state][word] / count_dic[state])
                B.write(state + '/' + word + '/' + str(array_B[state][word]) + '/' + '\n')


    def get_prob_array_from_file(self):
        pi = open(hmm_pi, 'r', encoding='utf-8')
        B = open(hmm_B, 'r', encoding='utf-8')
        A = open(hmm_A, 'r', encoding='utf-8')
        for line in pi:
            lines=line.split('/')
            lines=lines[:len(lines)-1] #去除 \n
            array_Pi[lines[0]] = float(lines[1])
        for line in A:
            lines=line.split('/')
            lines=lines[:len(lines)-1]
            # print(lines)
            array_A[lines[0]][lines[1]] =float(lines[2])
        for line in B:
            lines = line.split('/')
            lines = lines[:len(lines) - 1]
            array_B[lines[0]][lines[1]] = float(lines[2])

    #标记一句话的tag
    def tag(self,wordline):
        '''
        标注一行的每个字的状态，返回字的list和状态的list
        status'BEMS'累计计算出现次数
        如果本行不为空，那么训练的行数加一
        :param wordline: 输入的是一行标注的文字
        :return: word，每个字分割后的list
                wordtag，对应的每个字的状态标注list
        '''
        word = []
        wordtag = []
        if wordline=='\n':
            return word,wordtag
        global line_num,count_dic
        line_num=line_num+1
        l = re.split(r'[\s]\s*', wordline.strip())  # r表示不转义，s表示空格，多个空格
        l = list(filter(None, l))  # 只能过滤空字符和None
        # 处理词
        for s in l:
            # 词+词性
            s = s.split('/')
            s1 = s[0].split('[')
            s1 = "".join(s1)
            for i in range(len(s1)):
                word.append(s1[i])
                word_set.add(s1[i])
            l = len(s1)
            if l == 1:
                wordtag.append('S')
                count_dic['S']=count_dic['S']+1
            elif l == 2:
                wordtag.append('B')
                wordtag.append('E')
                count_dic['B'] = count_dic['B'] + 1
                count_dic['E'] = count_dic['E'] + 1
            else:
                wordtag.append('B')
                for j in range(l-2):
                    wordtag.append('M')
                    count_dic['M'] = count_dic['M'] + 1
                wordtag.append('E')
                count_dic['B'] = count_dic['B'] + 1
                count_dic['E'] = count_dic['E'] + 1
        return word, wordtag

    # #根据训练集初始化Pi，A, B，并标记训练集
    # def dict(self,HMMtag_path=HMM_train_tag_path):
    #     '''
    #
    #     :param train_path: 分词好的训练集
    #     :param HMMtag_path: 标记好的结果，格式为一行字，一行标记
    #     :return:
    #     '''
    #     filetag=open(HMMtag_path,'w',encoding='utf-8')
    #     with open(train_path,'r',encoding='utf-8') as file:
    #         for line in file:
    #             if line=='\n':
    #                 continue
    #             word,wordtag=self.tag(line)
    #             # filetag.write(str(word)+'\n')
    #             filetag.write(str(wordtag)+'\n')
    #             #计算频数
    #             self.calculateA_B_PI(word,wordtag)
    #     #计算概率，对数形式
    #     file.close()
    #     self.Prob_Array()

   # 通过标记整个文本来训练参数，并将参数写入文本文件中
    def dict(self,train_txt=train_path,HMMtag_path=HMM_train_tag_path):
        filetag = open(HMMtag_path, 'w', encoding='utf-8')
        with open(train_txt, 'r', encoding='utf-8') as txt_f:
            lines = txt_f.readlines()
        for line in lines:
            if line == '\n':
                continue
            line=line[22:] #去除日期
            line_word, line_tag = self.tag(line)  # 得到一行标注
            filetag.write(str(line_tag) + '\n')
            array_Pi[line_tag[0]]=array_Pi.get(line_tag[0], 0) + 1
            for i in range(len(line_tag)):
                count_dic[line_tag[i]] += 1  # 计算状态总出现次数
                array_B[line_tag[i]][line_word[i]] = array_B[line_tag[i]].get(line_word[i], 0) + 1
                if i > 0:
                    array_A[line_tag[i - 1]][line_tag[i]] += 1  # 转移概率变化
        self.write_prob_array_to_file()
    #根据测试集的正确分词结果，生成标记文件
    def gen_reult_tag(self,test=test_compare,HMMtag_path=test_tag_path):
        '''
        将测试集进行标记，并生成文件
        :param test: 测试集
        :param HMMtag_path: 标记好的测试集文件
        :return:
        '''
        filetag = open(HMMtag_path, 'w', encoding='utf-8')
        with open(test, 'r', encoding='utf-8') as file:
            for line in file:
                if line == '\n':
                    continue
                word, wordtag = self.tag(line)
                # filetag.write(str(word)+'\n')
                filetag.write(str(wordtag) + '\n')

    # #计算Pi，A, B，
    # def calculateA_B_PI(self,word,wordtag):
    #     '''
    #     根据字和标签更新状态转移矩阵，初始化矩阵，发射矩阵
    #     :param word: 字
    #     :param wordtag:字对应的标签
    #     :return:
    #     '''
    #     global array_A,array_B,array_Pi
    #     if len(wordtag)==0:
    #         return
    #     #初始化概率矩阵，
    #     array_Pi[wordtag[0]] +=1
    #     #状态转移矩阵
    #     for i in range(len(wordtag)-1):
    #         array_A[wordtag[i]][wordtag[i+1]] +=1
    #     #发射矩阵
    #     for tag in wordtag:
    #         for w in word:
    #             if w in array_B[tag]:
    #                 array_B[tag][w] +=1
    #             else:
    #                 array_B[tag][w] =0


    # Viterbi算法求测试集最优状态序列
    def Viterbi(self,sentence, array_pi=array_Pi, array_a=array_A, array_b=array_B):
        # tab = [{}]  # 动态规划表
        # path = {}
        #
        # if sentence[0] not in array_b['B']:
        #     for state in status:
        #         if state == 'S':
        #             array_b[state][sentence[0]] = 0
        #         else:
        #             array_b[state][sentence[0]] = -3.14e+100
        #
        # array_b['S']['莠']=0
        # for state in status:
        #     # if sentence[0]=='莠':
        #     #     print(state)
        #     #     print(sentence[0])
        #     #     print(array_b['S'][sentence[0]])
        #     tab[0][state] = array_pi[state] + array_b[state][sentence[0]]
        #     # print(tab[0][state])
        #     # tab[t][state]表示时刻t到达state状态的所有路径中，概率最大路径的概率值
        #     path[state] = [state]
        # for i in range(1, len(sentence)):
        #     tab.append({})
        #     new_path = {}
        #     # if sentence[i] not in array_b['B']:
        #     #     print(sentence[i-1],sentence[i])
        #     for state in status:
        #         if state == 'B':
        #             array_b[state]['begin'] = 0
        #         else:
        #             array_b[state]['begin'] = -3.14e+100
        #     for state in status:
        #         if state == 'E':
        #             array_b[state]['end'] = 0
        #         else:
        #             array_b[state]['end'] = -3.14e+100
        #     for state0 in status:
        #         items = []
        #         # if sentence[i] not in word_set:
        #         #     array_b[state0][sentence[i]] = -3.14e+100
        #         # if sentence[i] not in array_b[state0]:
        #         #     array_b[state0][sentence[i]] = -3.14e+100
        #         # print(sentence[i] + state0)
        #         # print(array_b[state0][sentence[i]])
        #         for state1 in status:
        #             # if tab[i-1][state1] == -3.14e+100:
        #             #     continue
        #             # else:
        #             if sentence[i] not in array_b[state0]:  # 所有在测试集出现但没有在训练集中出现的字符
        #                 if sentence[i - 1] not in array_b[state0]:
        #                     prob = tab[i - 1][state1] + array_a[state1][state0] + array_b[state0]['end']
        #                 else:
        #                     prob = tab[i - 1][state1] + array_a[state1][state0] + array_b[state0]['begin']
        #                 # print(sentence[i])
        #                 # prob = tab[i-1][state1] + array_a[state1][state0] + array_b[state0]['other']
        #             else:
        #                 prob = tab[i - 1][state1] + array_a[state1][state0] + array_b[state0][
        #                     sentence[i]]  # 计算每个字符对应STATES的概率
        #             #                     print(prob)
        #             items.append((prob, state1))
        #         # print(sentence[i] + state0)
        #         # print(array_b[state0][sentence[i]])
        #         # print(sentence[i])
        #         # print(items)
        #         best = max(items)  # bset:(prob,state)
        #         # print(best)
        #         tab[i][state0] = best[0]
        #         # print(tab[i][state0])
        #         new_path[state0] = path[best[1]] + [state0]
        #     path = new_path
        #
        # prob, state = max([(tab[len(sentence) - 1][state], state) for state in status])
        # return path[state]
        Pre_State = {'B': 'ES', 'M': 'MB', 'S': 'SE', 'E': 'BM'}
        v = [{}]
        path = {}
        Min=-3.14e+100
        for state in status:  # 初始化
            v[0][state] = array_pi[state] + array_b[state].get(sentence[0], Min)
            path[state] = [state]
        for idx in range(1, len(sentence)):
            v.append({})
            new_path = {}
            for state_1 in status:
                em_p = array_b[state_1].get(sentence[idx], Min)
                (prob, state) = max([(v[idx - 1][y0] + array_a[y0].get(state_1, Min) + em_p, y0) for y0 in
                                     Pre_State[state_1]])
                v[idx][state_1] = prob
                new_path[state_1] = path[state] + [state_1]
            path = new_path
        (prob, state) = max((v[len(sentence) - 1][y], y) for y in 'ES')
        return path[state]

    def run(self,testpath,answerpath,resultpath='../io_file/segment/hmm.txt'):
        '''
        主函数
        :param testpath:
        :param answerpath:
        :return:
        '''
        self.dict()
        # self.gen_reult_tag()
        anserfile=open(answerpath,'w',encoding='utf-8')
        resultfile=open(resultpath,'w',encoding='utf-8')
        with open(testpath,'r',encoding='utf-8')as testfile:
            for line in testfile:
                if line=='\n':
                    continue
                sentence=line.strip()
                date=sentence[:19]
                sentence=line[19:] #去除\n
                path=self.Viterbi(sentence)
                begin, next_i = 0, 0
                res_word = ''  # 输入的to_seg_word分词结果
                for idx, char in enumerate(sentence):
                    tag = path[idx]  # 取一个tag
                    if tag == 'B':  # 表示开始
                        begin = idx
                    elif tag == 'E':  # 表示结束
                        res_word += sentence[begin:idx + 1] + '/'
                        next_i = idx + 1
                    elif tag == 'S':
                        res_word += char + '/'
                        next_i = idx + 1
                if next_i < len(sentence):
                    res_word += sentence[next_i:] + '/'
                res_word=res_word[:len(res_word)-1]
                resultfile.write(date+'/'+res_word)
            resultfile.write('\n')
                #anserfile.write(str(path)+'\n')


    def hmm_select_word(self,line,word_freq):
        line=line.split('/')
        line=line[:len(line)-1]
        result,seg='','' #分词结果
        for idx ,word in enumerate(line):
            if len(word)==1 and word not in word_freq:
                seg +=word
            else:
                if seg:
                    seg=self.hmm_n_gram(seg)
                    result +=seg
                    seg=''
                result +=word+'/'
        if seg:
            result += seg+'/'
        return result

    def hmm_n_gram(self,line):
        self.get_prob_array_from_file()
        path= self.Viterbi(line)
        resultline, seg = '', ''  # 分词结果
        endidx = 0
        for idx in range(len(path)):
            if path[idx] == 'S':
                resultline += line[idx] + '/'
                endidx = idx + 1
            elif path[idx] == 'B' or 'M':
                seg += line[idx]
            elif path[idx] == 'E':
                seg += line[idx]
                resultline += seg + '/'
                seg = ''
                endidx = idx + 1
        if endidx < len(line): resultline += line[endidx:] + '/'
        return resultline


    def precision(self):
        '''
        计算分词结果的准确率
        :return: 准确率
        '''
        f1 = open(test_tag_path, 'r', encoding='utf-8')
        f2 = open(answerpath, 'r', encoding='utf-8')
        l1 = []
        l2 = []
        count = 0
        num = 0
        for line in f1:
            l1.append(line)
        for line in f2:
            l2.append(line)
        for i in range(len(l1)):
            num += len(l1[i])
            for j in range(len(l1[i])):
                if l1[i][j] == l2[i][j]:
                    count += 1
        print('准确率为：',count / num)
        # 0.882476152872008
        return count,num,count/num

