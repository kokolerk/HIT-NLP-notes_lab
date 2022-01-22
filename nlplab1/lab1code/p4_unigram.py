# coding=gbk
from math import log
from  p4HMM import HMM
from p2_SCORE import Score
# import tqdm

Train_File='../io_file/199801_seg&pos.txt'
dic_path='../io_file/out-dict.txt'
result_path='../io_file/segment/seg_Unigram.txt'

word_freg={} #保存 词 词频
wordnum=0

class createDict:
    @staticmethod
    def gen_uni_dict(train=Train_File,dict=dic_path):
        '''
        生成一元文法的词典，输出格式为：
        每行：词 词出现的频数
        :param train: 训练集
        :param dict: 词典地址
        :return:
        '''
        global word_freg
        with open(train,'r',encoding='gbk') as f:
            lines=f.readlines()
        for line in lines:
            for word in line.split():
                if '/m' in word:
                    continue
                word=word[1 if word[0]=='[' else 0 : word.index('/') ]
                word_freg[word]=word_freg.get(word,0)+1
        #字典排序
        word_freg={k: word_freg[k] for  k in sorted(word_freg.keys())}
        with open(dict,'w',encoding='utf-8') as f1:
            for word in word_freg.keys():
                # print(word + '/' + str(word_freg[word]) + '\n')
                f1.write(word+'/'+str(word_freg[word])+'\n')
        word_freg={}
        createDict.get_uni_dict(dict)

    @staticmethod
    def get_uni_dict(dict=dic_path):
        '''
        读入已有的词典,在词典里面加入每个词的前缀
        :param dict: 词典地址
        :return:
        '''
        global word_freg,wordnum
        with open(dict,'r',encoding='utf-8') as f:
            wordline=f.readlines()
        for item in wordline:
            word,fre=item.split('/')[0:2]
            word_freg[word]=int(fre)
            wordnum +=1
            for count in range(1,len(word)):
                pre=word[:count]
                if pre not in word_freg:
                    word_freg[pre]=0

    @staticmethod
    def cre_undirect_graph(line):
        '''
        根据频数构建这句话的有向无环图，也就是把这句话当中，所有的词都列一遍
        :param line:
        :return: dag={},key为字的下标，value为组成词语的下标[key,value[i]]表示一个词的位置
        '''
        dag={}
        for i in range(len(line)):
            dag[i] = []
            idx=i
            for j in range(idx+1,len(line)+1):
                word_seg=line[idx:j]
                if word_seg in word_freg:
                    dag[i].append(j)
            #如果是空，这个字加入
            dag[i].append(i+1) if not dag[i] else dag[i]
        return dag
        # dag = {}  # 用于储存最终的DAG
        # n = len(line)  # 句子长度
        # for k in range(n):  # 遍历句子中的每一个字
        #     i = k
        #     dag[k] = []  # 开始保存处于第k个位置上的字的路径情况
        #     word_fragment = line[k]
        #     while i < n and word_fragment in word_freg:  # 以k位置开始的词的所在片段在词典中
        #         if word_freg[word_fragment] > 0:  # 若离线词典中存在该词
        #             dag[k].append(i)  # 将该片段加入到临时的列表中
        #         i += 1
        #         word_fragment = line[k:i + 1]
        #     dag[k].append(k) if not dag[k] else dag[k]  # 未找到片段，则将单字加入
        # return dag

    @staticmethod
    def max_fre_idx_line(line,dag):
        '''
        最大概率分词，选择概率最大的分词路径
        :param line: 一句话
        :param dag: 这句话形成的有向无环图
        :return: route: 路径
        '''

        n=len(line)
        route={n:(0,0)}
        log_total=log(wordnum)
        for idx in range(n-1,-1,-1):
            route[idx]=max( ( log(word_freg.get(line[idx:x],0) or 1)-log_total+route[x][0],x) for x in dag[idx])
        return route

        # n = len(line)
        # route = {n: (0, 0)}
        # log_total = log(wordnum)
        # for idx in range(n - 1, -1, -1):  # 动态规划求最大路径
        #     route[idx] = max((log(word_freg.get(line[idx:x + 1], 0) or 1) - log_total +
        #                       route[x + 1][0], x) for x in dag[idx])
        # return route

    @staticmethod
    def unigram(test='../test_out.txt',result='../seg_LM_out_uni.txt'):
        with open(test,'r',encoding='gbk')as f:
            lines=f.readlines()
        r=''
        for line in lines:
            if line=='\n':continue
            #日期单独处理
            line=line[:len(line)-1]
            dag=createDict.cre_undirect_graph(line)
            route=createDict.max_fre_idx_line(line,dag)
            beforestart=0
            segline=''
            while beforestart < len(line):
                start=route[beforestart][1]
                segline +=line[beforestart:start]+'/ '
                beforestart=start
            # segline +='\n'
            # r += segline
            h=HMM()
            segline=h.hmm_select_word(segline,word_freg)
            r +=segline+'\n'
        open(result,'w',encoding='utf-8').write(r)

createDict.gen_uni_dict()
createDict.unigram()