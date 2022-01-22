# coding=gbk
from math import log
from  p4HMM import HMM
from p2_SCORE import Score
# import tqdm

Train_File='../io_file/199801_seg&pos.txt'
dic_path='../io_file/bi_out_dict.txt'
result_path='../seg_LM_out_bi.txt'

word_dict = {}  # ���� �� ��һ���� �����ʶԳ��ֵ�Ƶ��
word_freg = {'BOS':0}  # ���� �� ��Ƶ,�Ȱ�ͷ�����ȥ
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
            #���ִ�
            seg_line=line.split()
            wordnum +=len(seg_line)
            seg_line.append('EOS/ ')
            seg_line.insert(0, 'BOS')
            for idx,word in enumerate(seg_line):
                if word=='BOS':
                    word_freg[word] +=1
                    continue #��һ���ʲ������Ԫ�ʵ�
                else:
                    word=word[1 if word[0]=='[' else 0:word.index('/')]
                    seg_line[idx]=word
                    #�����Ƶ��
                    if word in word_freg: word_freg[word]+=1
                    else: word_freg[word]=1
                    #�����Ԫ��Ƶ��
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
    #�������Ŀǰûɶ�ã����Բ���ʵ��
    @staticmethod
    def get_bi_dict(dict=dic_path):
        pass

    @staticmethod
    def probaliy(pre_word,word):
        '''
        ����ǰһ���ʳ��ֵ�����£����γ��ֵĸ���
        :param pre_word:
        :param word:
        :return:
        '''
        pre_num=word_freg.get(pre_word,0) #ǰ�ʴ�Ƶ
        pairnum=word_dict.get(word,{}).get(pre_word,0) #�ʶԳ��ֵ�Ƶ��
        return  log(pairnum+1)-log(pre_num+wordnum)

    @staticmethod
    #���������p4_unigram�ĺ������ܣ�������ͬ
    def cre_undirect_graph(line):
        '''
                ����Ƶ��������仰�������޻�ͼ��Ҳ���ǰ���仰���У����еĴʶ���һ��
                :param line:
                :return: dag={},keyΪ�ֵ��±꣬valueΪ��ɴ�����±�[key,value[i]]��ʾһ���ʵ�λ��
                '''
        dag = {}  # ���ڴ������յ�DAG
        n = len(line)  # ���ӳ���
        for k in range(n):  # ���������е�ÿһ����
            i = k
            dag[k] = []  # ��ʼ���洦�ڵ�k��λ���ϵ��ֵ�·�����
            word_fragment = line[k]
            while i < n and word_fragment in word_freg:  # ��kλ�ÿ�ʼ�Ĵʵ�����Ƭ���ڴʵ���
                if word_freg[word_fragment] > 0:  # �����ߴʵ��д��ڸô�
                    dag[k].append(i)  # ����Ƭ�μ��뵽��ʱ���б���
                i += 1
                word_fragment = line[k:i + 1]
            dag[k].append(k) if not dag[k] else dag[k]  # δ�ҵ�Ƭ�Σ��򽫵��ּ���
        return dag

    #�����ʷִʣ�����������·������Ԫ�ģ������һԪ�ĸ��Ӻܶ�
    #��������ǳ���
    @staticmethod
    def max_fre_idx_line(line, dag):
        n=len(line)-3 #ȥ��EOS�ĳ���
        start=3
        pre_graph={'BOS':{}} #�ؼ���Ϊǰһ���ʣ�ֵ��Ӧ�ĵ��ǴʺͶ�������
        word_graph={} #ÿ���ʽڵ������һ�������ʵĴ�ͼ
        for x in dag[3]:#��ʼ��ǰ��ΪBOS�����
            pre_graph['BOS'][(3,x+1)]=createDict.probaliy('BOS',line[3:x+1])
        while start <n: #��ÿһ���ֿ��ܵĴ�������һ���ʵĴʵ�
            for idx in dag[start]:#����dag[start]�е�ÿһ�������ڵ�
                pre_word=line[start:idx+1]
                temp={}
                for next_end in dag[idx+1]:
                    last_word=line[idx+1:next_end+1]
                    if line[idx+1:next_end+3]=='EOS': #�ж��Ƿ񵽴��β
                        temp['EOS']=createDict.probaliy(pre_word,'EOS')
                    else:
                        temp[(idx+1,next_end+1)]=createDict.probaliy(pre_word,last_word)
                pre_graph[(start,idx+1)]=temp #ÿһ����start��ʼ�Ĵʣ�������һ��������һ���ʵ��ֵ�
            start+=1
        pre_words=list(pre_graph.keys()) #��ʾ���е�ǰһ����
        ###��֮�µĴ����𽥿����������о����Ƕ�̬�滮��˼�룬���ǲ�̫������
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