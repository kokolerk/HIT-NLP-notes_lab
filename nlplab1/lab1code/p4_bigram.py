# coding=gbk
from math import log
from  p4HMM import HMM
from p2_SCORE import Score
# import tqdm

Train_File='../io_file/199801_seg&pos.txt'
dic_path='../io_file/bi_out_dict.txt'
result_path='../seg_LM_out_bi.txt'

word_dict = {}  # 保存 词 上一个词 两个词对出现的频率
word_freg = {'BOS':0}  # 保存 词 词频,先把头储存进去
wordnum = 0

class createDict:
    @ staticmethod
    def gen_bi_dict(train=Train_File,dict=dic_path):
        global word_freg,wordnum,word_dict
        with open(train,'r',encoding='gbk')as f:
            lines=f.readlines()
        for line in lines:
            if line=='\n':continue
            line=line[:len(line)-1]
            #划分词
            seg_line=line.split()
            wordnum +=len(seg_line)
            seg_line.append('EOS/ ')
            seg_line.insert(0, 'BOS')
            for idx,word in enumerate(seg_line):
                if word=='BOS':
                    word_freg[word] +=1
                    continue #第一个词不计入二元词典
                else:
                    word=word[1 if word[0]=='[' else 0:word.index('/')]
                    seg_line[idx]=word
                    #加入词频表
                    if word in word_freg: word_freg[word]+=1
                    else: word_freg[word]=1
                    #加入二元词频表
                    if word not in word_dict.keys():
                        word_dict[word]={}
                    if seg_line[idx-1] not in word_dict[word]:
                        word_dict[word][seg_line[idx-1]]=0
                    word_dict[word][seg_line[idx-1]] +=1
        word_freg={k: word_freg[k] for k in sorted(word_freg.keys())}
        word_dict={k: word_dict[k] for k in sorted(word_dict.keys())}
        with open(dic_path, 'w', encoding='utf-8') as f:
            for word in word_dict:
                for pre in word_dict[word]:
                    f.write(word + ' ' + pre + ' ' + str(word_dict[word][pre]) + '\n')
        #print(wordnum)
    #这个函数目前没啥用，可以不用实现
    @staticmethod
    def get_bi_dict(dict=dic_path):
        pass

    @staticmethod
    def probaliy(pre_word,word):
        '''
        计算前一个词出现的情况下，本次出现的概率
        :param pre_word:
        :param word:
        :return:
        '''
        pre_num=word_freg.get(pre_word,0) #前词词频
        pairnum=word_dict.get(word,{}).get(pre_word,0) #词对出现的频率
        return  log(pairnum+1)-log(pre_num+wordnum)

    @staticmethod
    #这个函数和p4_unigram的函数功能，代码相同
    def cre_undirect_graph(line):
        '''
                根据频数构建这句话的有向无环图，也就是把这句话当中，所有的词都列一遍
                :param line:
                :return: dag={},key为字的下标，value为组成词语的下标[key,value[i]]表示一个词的位置
                '''
        dag = {}  # 用于储存最终的DAG
        n = len(line)  # 句子长度
        for k in range(n):  # 遍历句子中的每一个字
            i = k
            dag[k] = []  # 开始保存处于第k个位置上的字的路径情况
            word_fragment = line[k]
            while i < n and word_fragment in word_freg:  # 以k位置开始的词的所在片段在词典中
                if word_freg[word_fragment] > 0:  # 若离线词典中存在该词
                    dag[k].append(i)  # 将该片段加入到临时的列表中
                i += 1
                word_fragment = line[k:i + 1]
            dag[k].append(k) if not dag[k] else dag[k]  # 未找到片段，则将单字加入
        return dag

    #最大概率分词，计算概率最大路径，二元的，相对于一元的复杂很多
    #这个代码是抄的
    @staticmethod
    def max_fre_idx_line(line, dag):
        n=len(line)-3 #去除EOS的长度
        start=3
        pre_graph={'BOS':{}} #关键词为前一个词，值对应的的是词和对数概率
        word_graph={} #每个词节点存有上一个相连词的词图
        for x in dag[3]:#初始化前词为BOS的情况
            pre_graph['BOS'][(3,x+1)]=createDict.probaliy('BOS',line[3:x+1])
        while start <n: #对每一个字可能的词生成下一个词的词典
            for idx in dag[start]:#遍历dag[start]中的每一个结束节点
                pre_word=line[start:idx+1]
                temp={}
                for next_end in dag[idx+1]:
                    last_word=line[idx+1:next_end+1]
                    if line[idx+1:next_end+3]=='EOS': #判断是否到达句尾
                        temp['EOS']=createDict.probaliy(pre_word,'EOS')
                    else:
                        temp[(idx+1,next_end+1)]=createDict.probaliy(pre_word,last_word)
                pre_graph[(start,idx+1)]=temp #每一个以start开始的词，都建立一个关于下一个词的字典
            start+=1
        pre_words=list(pre_graph.keys()) #表示所有的前一个词
        ###这之下的代码逐渐看不懂……感觉还是动态规划的思想，但是不太明白了
        for pre_word in pre_words:
            for word in pre_graph[pre_word].keys():
                word_graph[word]=word_graph.get(word,list())
                word_graph[word].append(pre_word)
        pre_words.append('EOS')
        route={}
        for word in pre_words:
            if word=='BOS':
                route[word]=(0.0,'BOS')
            else:
                pre_list=word_graph.get(word,list())
                route[word]=(-65507,'BOS') if not pre_list else max(
                    (pre_graph[pre][word]+route[pre][0],pre) for pre in pre_list
                )
        return route





    @staticmethod
    def bigram(test='../test_out.txt',result=result_path):
        '''

        :param test:
        :param result:
        :return:
        '''
        with open(test,'r',encoding='gbk')as f:
            lines=f.readlines()
        r=''
        for line in lines:
            if line=='\n':continue
            line='BOS'+line[ :len(line)-1]+'EOS'
            dag=createDict.cre_undirect_graph(line)
            route=createDict.max_fre_idx_line(line,dag)
            position='EOS'
            segline=''
            while True:
                position=route[position][1]
                if position=='BOS':
                    break
                segline =line[position[0]:position[1]]+'/ '+segline
            segline +='\n'
            h=HMM()
            segline=h.hmm_select_word(segline,word_freg)
            r +=segline+'\n'
        open(result,'w',encoding='utf-8').write(r)


createDict.gen_bi_dict()
createDict.bigram()