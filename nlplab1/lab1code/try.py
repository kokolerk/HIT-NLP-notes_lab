a = open('../test_out.txt', 'r', encoding='gbk')
b = open('../seg_LM_out.txt', 'w', encoding='utf-8')
c=open('../seg_LM_out_wang.txt', 'r', encoding='utf-8')
alines=a.readlines()
print(alines)
clines=c.readlines()
print(clines)
len=[]
for line in alines:
    len1=len(line)-1
    len.append(len1)
print(len)
for line in blines:
    line=line.split('')
    print(line)
