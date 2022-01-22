import re
#计算list各元素的和
def sumList(l):
    lenl=len(l)
    count=0
    for i in range(lenl):
        count=count+l[i]
    return count

#计算准确率，召回率，F-评价
def Score(calculate_path,human_path,scorepath,k):
    '''

    :param calculate_path: 机器分词的文件路径，utf-8

    :param human_path: 人工分词的文件路径，utf-8

    :param scorepath: 写分数的文件路径
    :return:
            presion
            reall
            F
    '''
    #首先计算正确的分词数
    calculate_file=open(calculate_path,'r',encoding='utf-8')
    human_file=open(human_path,'r',encoding='utf-8')
    #人工分词数
    numcs=0
    #正确分词数
    numhuman=0
    #具体的分词结果
    resultcs=[]
    resulthuman=[]
    #机器分词结果
    for line in calculate_file:
        if line=='\n': continue
        l=line.split('/')
        if '\n' in l:
            l.pop(l.index('\n'))
        resultcs.append(l)
        numcs +=len(l)
    #人工分词结果
    for line in human_file:
        #去除空行
        if line=='\n':
            continue
        #去除空格和标记的词性
        l=re.sub(r'\s*|[a-zA-Z]*|\[|\]','',line)
        #去除换行符
        if l.count('\n')>0: l.remove('\n')
        #分词
        l = l.split('/')
        l.pop()#最后有一个额外的/作为结尾
        resulthuman.append(l)
        numhuman +=len(l)

    count=0
    #正确分词总数
    for i in range(len(resultcs)):
        temp1=[]
        s=0
        temp1.append(s)
        for j in range(len(resultcs[i])):
            s=s+len(resultcs[i][j])
            temp1.append(s)
        temp2 = []
        s=0
        temp2.append(s)
        for j in range(len(resulthuman[i])):
            s=s+len(resulthuman[i][j])
            temp2.append(s)
        for i in range(len(temp1)-1):
            a=temp1[i]
            b=temp1[i+1]
            if temp2.count(a)>0 and temp2.count(b)>0 and temp2.index(b)-temp2.index(a)==1: count=count+1

    precision=count/numcs
    recall=count/numhuman
    F=(k *k +1)*precision*recall/(recall+k*k*precision)
    score=open(scorepath,'w',encoding='utf-8')
    score.write('precision:'+str(precision)+'\n')
    score.write('recall:' + str(recall) + '\n')
    score.write('F:' + str(F)+ '\n')
    print('分数地址为',scorepath)
    return precision,recall,F

