# nlplab1
## README.md文件说明
- 此文件对代码、文本文件以及如何测试我的代码进行了详细的介绍
- 实验报告为pdf格式
## 实验说明
##### 完成说明
- 完成了所有要求
- 为了检查方便，实现了一个测试类run_code.py文件
- 词典构建
- 正反向最大匹配分词的实现
- 正反向最大匹配分词效果分析
- 基于机械匹配分词系统速度优化
- 基于MM的一元文法分词实现
- 基于MM的二元文法分词
- 基于HMM的未登录词识别
- 对于整个系统分词性能的最终优化
##### 环境说明
- 本次实验使用编程语言是Python3
- 实验机器为个人电脑
- 系统为Mac0S
- 运行集成环境为Community Pycharm 2021.2.3
##### 代码复查：训练集改变时需要在下面程序运行时输入‘T’重构系统
- 测试第一部分：取消lab1code/run_code.py文件主函数中的test_part1code()的注释，运行此程序
- 测试第二部分：取消lab1code/run_code.py文件主函数中的test_part2code的注释，运行此程序
- 运行优化代码：取消lab1code/run_code.py文件主函数中的test_part3code()的注释，运行此程序
- 测试一元文法+hmm：取消lab1code/run_code.py文件主函数中的test_part4_Unigram的注释，运行此程序
- 测试二元文法+hmm：取消lab1code/run_code.py文件主函数中的test_part4_bigram的注释，运行此程序
- 测试纯HMM：取消lab1code/run_code.py文件主函数中的test_part4_hmm的注释，运行此程序
## 文件结构及格式
##### 文件格式
- 老师给的标准文本199801_seg&pos.txt和199801_sent.txt文本编码格式为'gbk'
- 除了以上两个文本之外的所有生成文本的编码格式为'utf-8'
#### lab1code：实验代码文件夹
- part_x…….py：文件为对应实验部分的代码。
- Run_code.py：为了方便检查实现的一个模块，测试方式已在上文介绍
##### io_file：实验输出输入文件夹
- dic：生成的词典文件夹
    - dic.txt：3.5节之前使用的词典，仅包括词汇，且不包含ASCII半角字符
    - uni_dict.txt：一元文法使用词典，包含词和对应的词频
    - bi_dict.txt：二元文法使用词典，包含词及其前词和对应的组合词频
    
- hmm：生成的隐马模型参数和训练文件夹
	- A.txt：HMM状态转移概率
	
	  B.txt：HMM的发射概率
	
	- pi.txt：HMM的初始状态概率
	
	- HMM_train_tag.txt：本次实验的训练集，取自老师给的分词标准文件
	
	- HMM_test_tag.txt：本次实验的测试集，取自老师给的分词标准文件且与训练集互补
	
- segment：生成的分词文件夹
	- seg_Unigram：一元文法分词结果（包括未登录词识别）
	- seg_bigram：二元文法分词结果（包括未登录词识别）
	- hmm：纯隐马模型生成的分词结果
	- seg_FMM：3.2节fmm分词结果
	- seg_BMM：3.2节bmm分词结果
	- seg_FMM_optimize：3.4节生成的fmm分词结果
	- seg_BMM_optimize：3.4节生成的bmm分词结果
	
- score:生成的分数文件夹

    - score_Unigram.txt：一元文法分词结果（包括未登录词识别）
    - score_bigram.txt: 二元文法分词结果（包括未登录词识别）
    - score_hmm:纯隐马模型生成的分词结果
    - score_FMM：3.2节fmm分词结果
    - score_BMM：3.2节bmm分词结果
    - score_FMM_optimize：3.4节生成的fmm分词结果
    - score_BMM_optimize：3.4节生成的bmm分词结果

- train_test

    - dict.txt：3.1节生成的词典
    - test.txt:  测试集
    - test_compare.txt: 测试集标准结果
    - train_seg&pos.txt：训练集

- 199801_seg&pos.txt：给定标准文件，是本次实验训练集和答案集产生来源

- 199801_sent.txt：给定标准文件，是本次实验的测试集的产生来源

    