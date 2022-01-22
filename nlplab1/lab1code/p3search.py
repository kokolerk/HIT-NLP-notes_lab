import re
import numpy as np
import time
import p1_DICT
class Node():
    '''
    制作一个hash表
    '''
    def __init__(self,is_word=False,char='',init_size=700):
        self.is_word=is_word #表示这是不是一个词最后一个字
        self.char=char
        self.wordnum=0
        self.list_len=init_size
        self.list=[None]*init_size

    def hash(self,char):
        '''
        返回一个字符的hash值
        :param char: 字符
        :return: 返回他的hash值
        '''
        return int(ord(char)%self.list_len) #取余和/，效果天壤之别

    def rehash(self,node):
        '''
        扩张hash表大小为原来的2倍
        :param char: 输入的字符
        :return:
        '''
        self.list_len= 2*self.list_len #hash表扩大2倍
        oldlist=self.list
        self.wordnum= self.wordnum + 1 # 字数增加1
        self.list=[None]*self.list_len
        for childnode in oldlist:
            if childnode is not None:
                # 计算每个字符的hashcode，并把他们赋值为index
                index = self.hash_char(char=childnode.char)
                # 如果这个位置已经有字符，往后顺位找空字符的index
                while self.list[index] is not None:
                    index = (index + 1) % self.list_len # 取模，可以循环
                self.list[index] = childnode
        self.addchild(node)

    def addchild(self,node):
        '''
        向hash表里面增加node
        :param node:需要增加的节点
        :return:
        '''
        if float(self.wordnum)/(self.list_len) > (float)(2/3):
            self.rehash(node)
        else:
            index=self.hash(node.char)
            while self.list[index]is not None:
                index = (index + 1) % self.list_len  # 取模，可以循环
            self.list[index]=node

    def search_node_by_char(self,char):
        '''
        通过char字符，来查找hash表里面对应的node的位置
        :param char: 查找的一个字符
        :return: 如果找到了，返回对应的node数据结构；如果找不到，返回None
        '''
        index=self.hash(char)
        while self.list[index] is not None:
            node=self.list[index]
            if node.char==char:
                return node
            else:
                index = (index + 1) % self.list_len
        return None


class DicAction:
    Words_List=[]

    @staticmethod
    def get_fmm_dic(dic_path='../io_file/train_test/dict.txt'):
        for line in open(dic_path, 'r', encoding='UTF-8'):
            DicAction.Words_List.append(line[:len(line)-1])
        # 初始化数据结构node，hash表长度一开始为7000
        root = Node(init_size=9000)
        # 利用insert函数插入
        for word in DicAction.Words_List:
            DicAction.insert_fmm(word, root)
        return root

    @staticmethod
    def get_bmm_dic(dic_path='../io_file/train_test/dict.txt'):
        for line in open(dic_path, 'r', encoding='UTF-8'):
            DicAction.Words_List.append(line[:len(line)-1])
        # 初始化数据结构node，hash表长度一开始为7000
        root = Node(init_size=9000)
        # 利用insert函数插入
        for word in DicAction.Words_List:
            DicAction.insert_bmm(word, root)
        return root

    @staticmethod
    def insert_bmm(word,root):
        '''
        反向插入节点
        :param word:
        :param root:
        :return:
        '''
        # count=1
        # length=len(word)-1
        # #找第最后一个字在root里面的位置
        # node=root.search_node_by_char(word[length-1])
        # beforenode=root
        # #如果可以找到该字
        # while node is not None:
        #     #如果字已经找完，标志true
        #     if count==length:
        #         node.is_word=True
        #         return
        #     #后缀存储，继续向前查找字
        #     beforenode = node
        #     node = node.search_node_by_char(word[length-1-count])
        #     count += 1
        # #如果出现了未在trie里面存储的字，加入这个字
        # count -=1
        # while count<length:
        #     node=Node()
        #     node.char=word[length-1-count]
        #     count += 1
        #     beforenode.addchild(node)
        #     beforenode = node
        # node.is_word = True
        count = len(word) - 1
        node = root.search_node_by_char(word[count])
        before_node = root
        while node is not None:
            if count == 0:
                node.is_word = True
                return
            count -= 1
            before_node = node
            node = node.search_node_by_char(word[count])
        while count >= 0:
            node = Node()
            node.char = word[count]
            count -= 1
            before_node.addchild(node)
            before_node = node
        node.is_word = True

    @staticmethod
    def insert_fmm(word,root):
        '''
        :param word:
        :param root:
        :return:
        '''
        count=1
        length=len(word)
        #找第一个字在root里面的位置
        node=root.search_node_by_char(word[0])
        beforenode=root
        #如果可以找到该字
        while node is not None:
            #如果字已经找完，标志true
            if count==length:
                node.is_word=True
                return
            #前缀存储，继续向后查找字
            beforenode = node
            node = node.search_node_by_char(word[count])
            count += 1
        #如果出现了未在trie里面存储的字，加入这个字
        count -=1
        while count<length:
            node=Node(char=word[count])
            count += 1
            beforenode.addchild(node)
            beforenode = node
        node.is_word = True



class strMarch:
    @staticmethod
    def bmm(root, txt_path='../io_file/train_test/test.txt', fmm_path='../io_file/seg_BMM_optimize.txt'):
        '''
        实现思路就是把待分割的句子翻转过来，用fmm分割
        :param root: 储存好的trie树的根节点
        :param txt_path: 需要分词的测试文件的地址
        :param fmm_path: 分词后的结果文件的地址
        :return:
        '''
        seg_result = ''
        file = open(txt_path, 'r', encoding='utf-8')
        for line in file:
            if line == '\n':
                continue
            seg_line, line = '', line[:len(line) - 1]  # 去掉读取的换行符
            line=line[::-1] #倒转
            while len(line) > 0:
                count = 0
                terminal_word=line[0] #单个词
                node = root.search_node_by_char(line[0])
                while node is not None:
                    count += 1
                    if node.is_word:
                        terminal_word = line[:count]
                    if count == len(line):
                        break
                    node = node.search_node_by_char(line[count])
                line = line[len(terminal_word):]
                seg_line +='/'+terminal_word
            seg_line=seg_line[::-1]
            seg_line = p1_DICT.pre_line(seg_line)  # 将数字、字母、字符之类的连起来
            seg_result += seg_line + '\n'
        open(fmm_path, 'w', encoding='UTF-8').write(seg_result)

    @staticmethod
    def fmm(root, txt_path='../io_file/train_test/test.txt', fmm_path='../io_file/seg_FMM_optimize.txt'):
        '''

        :param root: 储存好的trie树的根节点
        :param txt_path: 需要分词的测试文件的地址
        :param fmm_path: 分词后的结果文件的地址
        :return:
        '''
        seg_result = ''
        file = open(txt_path, 'r', encoding='utf-8')
        for line in file:
            if line=='\n':
                continue
            seg_line, line = '', line[:len(line) - 1]  # 去掉读取的换行符
            while len(line) > 0:
                count = 0
                terminal_word=line[0] #单个词
                node = root.search_node_by_char(line[0])
                while node is not None:
                    count += 1
                    if node.is_word:
                        terminal_word = line[:count]
                    if count == len(line):
                        break
                    node = node.search_node_by_char(line[count])
                line = line[len(terminal_word):]
                seg_line += terminal_word + '/'
            seg_line = p1_DICT.pre_line(seg_line) #将数字、字母、字符之类的连起来
            seg_result += seg_line+ '\n'
        open(fmm_path, 'w', encoding='UTF-8').write(seg_result)












