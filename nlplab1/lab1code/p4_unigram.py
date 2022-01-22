# coding=gbk
from math import log
from  p4HMM import HMM
from p2_SCORE import Score
# import tqdm

Train_File='../io_file/199801_seg&pos.txt'
dic_path='../io_file/out-dict.txt'
result_path='../io_file/segment/seg_Unigram.txt'

word_freg={} #���� �� ��Ƶ
wordnum=0

class createDict:
    @staticmethod
    def gen_uni_dict(train=Train_File,dict=dic_path):
        '''
        ����һԪ�ķ��Ĵʵ䣬�����ʽΪ��
        ÿ�У��� �ʳ��ֵ�Ƶ��
        :param train: ѵ����
        :param dict: �ʵ��ַ
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
        #�ֵ�����
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
        �������еĴʵ�,�ڴʵ��������ÿ���ʵ�ǰ׺
        :param dict: �ʵ��ַ
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
        ����Ƶ��������仰�������޻�ͼ��Ҳ���ǰ���仰���У����еĴʶ���һ��
        :param line:
        :return: dag={},keyΪ�ֵ��±꣬valueΪ��ɴ�����±�[key,value[i]]��ʾһ���ʵ�λ��
        '''
        dag={}
        for i in range(len(line)):
            dag[i] = []
            idx=i
            for j in range(idx+1,len(line)+1):
                word_seg=line[idx:j]
                if word_seg in word_freg:
                    dag[i].append(j)
            #����ǿգ�����ּ���
            dag[i].append(i+1) if not dag[i] else dag[i]
        return dag
        # dag = {}  # ���ڴ������յ�DAG
        # n = len(line)  # ���ӳ���
        # for k in range(n):  # ���������е�ÿһ����
        #     i = k
        #     dag[k] = []  # ��ʼ���洦�ڵ�k��λ���ϵ��ֵ�·�����
        #     word_fragment = line[k]
        #     while i < n and word_fragment in word_freg:  # ��kλ�ÿ�ʼ�Ĵʵ�����Ƭ���ڴʵ���
        #         if word_freg[word_fragment] > 0:  # �����ߴʵ��д��ڸô�
        #             dag[k].append(i)  # ����Ƭ�μ��뵽��ʱ���б���
        #         i += 1
        #         word_fragment = line[k:i + 1]
        #     dag[k].append(k) if not dag[k] else dag[k]  # δ�ҵ�Ƭ�Σ��򽫵��ּ���
        # return dag

    @staticmethod
    def max_fre_idx_line(line,dag):
        '''
        �����ʷִʣ�ѡ��������ķִ�·��
        :param line: һ�仰
        :param dag: ��仰�γɵ������޻�ͼ
        :return: route: ·��
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
        # for idx in range(n - 1, -1, -1):  # ��̬�滮�����·��
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
            #���ڵ�������
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