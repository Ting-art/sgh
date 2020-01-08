# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 13:45:07 2019

@author: 清矿轩
"""

import numpy as np
import itertools
import pandas as pd
import time
#np.random.seed(25)
def predict(data,mo_data,var_num,weight):
    '''
    输入：字句，模拟数据，字句的权重数组
    输出：当前得分
    '''
    #首先准备数据，得到变量的位置
    index=data.copy()
    din=(index<0)
    index[din]=-index[din]
    #得到不可满足的下界
    gin=(data>0)
    Data=data.copy()
    Data[gin]=0
    gt=np.sum(Data,axis=1)
    #第一步，首先得到模拟数据对应字句取值
    if np.ndim(mo_data)==1:
        mo_data=mo_data.reshape(1,var_num)
        #print(index)
        index=index.astype(int)
        #print(mo_data[0,index-1])
        e_=data*mo_data[0,index-1]
        f_=np.sum(e_,axis=1)
        
        gg=(f_!=gt)
        g_=gg+0
        h=(f_==gt)+0
        h1=np.where(h==1)
        h_=h1[0]
        l=np.dot(g_,weight)
    else :
        c_=np.arange(mo_data.shape[0]).reshape(mo_data.shape[0],1,1)
        index=index.astype(int)
        e_=data*mo_data[c_,index-1]
        f_=np.sum(e_,axis=2)
        
        gg=(f_!=gt)
        g_=gg+0
        h=(f_==gt)+0
        h1=np.where(h==1)
        h_=h1[0]
        #print(g_.shape)
        l=np.dot(g_,weight[:,None])
        
        pass
    
    return l,g_,h_,gg


def test_generation(var_n):
    #生成布尔函数的全部情况
    n=var_n
    m=2**n
    dat=np.zeros((m,n))
    k=0
    for i in itertools.product([0,1],repeat= n):
        i=np.array(i)
        dat[k]=i
        k+=1
        pass
    
    return dat

def Sflip(var_n,data,mo_data=None):
    #简单翻转算法
    var_num=var_n
    if mo_data is None:
        mo_data=np.random.choice([0,1],var_n)
        pass
    weight=np.ones(data.shape[0])
    #mo_data=np.array([0,0,1,1,0,1,0,1,0,0])
    L_max,g_,h_,gg=predict(data,mo_data,var_n,weight)
    k=0
    
    while True:
        j=0
        for i in range(var_num):
            m=mo_data[i]
            if m==0:
                mo_data[i]=1
                L,g_,h_,gg=predict(data,mo_data,var_num,weight)
                if L<=L_max:
                    mo_data[i]=0
                else:
                    L_max=L
                    j+=1
                    pass
            else :
                mo_data[i]=0
                L,g_,h_,gg=predict(data,mo_data,var_num,weight)
                if L<=L_max:
                    mo_data[i]=1
                else:
                    L_max=L
                    j+=1
                    pass
                pass
            pass
        if j>0:
            k+=1
            pass
        if j==0:
            break
        pass
    
    return mo_data,L_max,k

def local_opti(data,var_n,op_n,value,index,l,mo_data):
    
    #局部优化max-sat问题
    #data:字句集合
    #var_n:变量个数
    #op_n:选中的局部变量个数
    #value:当前迭代序列
    #a1：变量顺序
    
    d1=index[(l-1)*op_n:l*op_n]
    #print(d1)
    #d2=np.hstack((index[0:(l-1)*op_n],index[l*op_n:]))
    d3=list(d1)
    d4=sorted(d3)
    #g_v=value[d1-1]
    #value=np.random.choice([0,1],var_n)
    #grade=0
    #第一步，根据给定的变量简化数据集
    val=np.ones(var_n)
    
    val1=val.copy()
    val1[np.array(d4)-1]=0
    
    gada1=data.copy()
    
    index=data.copy()
    din=(index<0)
    index[din]=-index[din]
    
    val1=val1.reshape(1,var_n)
    gada1=gada1*val1[0,index-1]
    
    weight=np.ones(gada1.shape[0])
    l,g_,h_,gg=predict(gada1,value,var_n,weight)
    gada1=data[h_]
   
    #第二步，将固定变量全部赋值为0
    dval=np.zeros(var_n)
    dval[np.array(d4)-1]=1
    
    gada=gada1.copy()
    
    index=gada1.copy()
    din=(index<0)
    index[din]=-index[din]
    
    dval=dval.reshape(1,var_n)
    gada=gada*dval[0,index-1]
    
    c1=np.max(gada,axis=1)
    c2=np.min(gada,axis=1)
                
    #m=h4[c1==c2].shape[0]#获得字句取值为0的行数
    gada=gada[c1!=c2]#得到新的数据集
    data=gada.astype(int)
    
    #print(data.shape)
    
    j=1
    for i in d4:
        data[data==i]=j
        data[data==-i]=-j
        j+=1
        
        pass
    
    
    weig=np.ones(data.shape[0])
    l,g_,h_,gg=predict(data,mo_data,op_n,weig)
    
    g_v=value[np.array(d4)-1]
    Ll,g_,h_,gg=predict(data,g_v,op_n,weig)
    
    #print(data.shape[0])
    #print(l)
    #print(max(l))
    
    #print(Ll)
    #print(grade)
    a1=np.where(l==max(l))
    a2=a1[0][0]
    true_index=mo_data[a2]
    
    
    value[np.array(d4)-1]=true_index
    
    
    
    return value

df=pd.read_csv('./sat_data_300_1200.csv')#读取excel内容
gh=df.values#得到excel数组内容
data=np.delete(gh,gh.shape[1]-1,axis=1)#删除掉最后一行多余的0

Data=data.copy()
qada=Data
var_n=300
op_n=10
value=np.random.choice([0,1],var_n)
r_value=value.copy()
index=np.random.permutation(var_n)+1

#value=value[a1-1]
l_value=value.copy()

#g_v=l_value[0:op_n]
l=1
mo_data=test_generation(op_n)
val1=local_opti(data,var_n,op_n,value,index,l,mo_data)
weight=np.ones(data.shape[0])
l,g_,h_,gg=predict(data,val1,var_n,weight)
#print(value[0:10])
#print(value[0:10])
print(val1)
print('局部优化后得到的可满足字句个数为%d: '%(l))
print(r_value)

#print(l_value)



l,g_,h_,gg=predict(data,r_value,var_n,weight)
print('随机初始化得到的可满足字句个数为%d: '%(l))




value0=np.random.choice([0,1],var_n)

mo_data,L_max,k=Sflip(var_n,Data,mo_data=None)
#value0=mo_data

num=var_n//op_n
epoch=100#迭代的轮数


print('************************')
print('迭代开始------------')

mat=np.zeros((2,epoch))
start=time.time()
const=np.ones(Data.shape[0])
weight=const
ls,g1_,h_,gg=predict(Data,value0,var_n,const)
max_reward=ls
max_value=value0
start=time.time()
mo_data=test_generation(op_n)
for j in range(epoch):
    k=0
    print('^^^^^^^^^^^^^^^^^^^^^^^^')
    ls,g1_,h_,gg=predict(Data,value0,var_n,const)
    index=np.random.permutation(var_n)+1
    qada=Data
    #weight_1=weight.copy()
    for i in range(num):
        print(i)
        value1,g_,h_,gg=predict(Data,value0,var_n,weight)
        print('随机初始化得到的可满足字句个数为%d: '%(value1))
        
        l=k+1
        k+=1
        
        #g_v=value0[(l-1)*op_n:l*op_n]
        
        value=local_opti(Data,var_n,op_n,value0,index,l,mo_data)
        
        value2,g_,h_,gg=predict(Data,value,var_n,weight)
        print('局部优化后得到的可满足字句个数为%d: '%(value2))
        pass
    lt,g2_,h_,gg=predict(Data,value,var_n,const)
    if lt>max_reward:
        max_value=value
        max_reward=lt
        pass
    #mo_data,L_max,k=Sflip(var_n,Data,mo_data=value)
    '''
    new_v=0.9*value+0.1*value0
    a=np.random.rand(var_n)
    b=np.where(a<=new_v)
    
    cha_v=np.zeros(var_n)
    cha_v[b[0]]=1
    value0=cha_v
    '''
    value0=value.copy()
    
    
    if lt==ls:
        #dat=Data[h_]
        #print(dat)
        #print(abs(dat[0,0])-1)
        '''
        a=np.random.rand(var_n)
        b=np.where(a>0.90)
        value0[b[0]]=1-value0[b[0]]
        '''
        3
        
        a = np.random.randint(0, var_n-1, 1)
        value0[a[0]]=1-value0[a[0]]
        b = np.random.randint(0, var_n-1, 1)
        value0[b[0]]=1-value0[b[0]]
        
        c = np.random.randint(0, var_n-1, 1)
        value0[c[0]]=1-value0[c[0]]
        
        d = np.random.randint(0, var_n-1, 1)
        value0[d[0]]=1-value0[d[0]]
        '''
        e = np.random.randint(0, var_n-1, 1)
        value0[e[0]]=1-value0[e[0]]
        
        f = np.random.randint(0, var_n-1, 1)
        value0[f[0]]=1-value0[f[0]]
        g = np.random.randint(0, var_n-1, 1)
        value0[g[0]]=1-value0[g[0]]
        h = np.random.randint(0, var_n-1, 1)
        value0[h[0]]=1-value0[h[0]]
        '''
        pass
        
    pass
end=time.time()
#print(L_max)
print(max_value)
print(max_reward)
print('程序消耗的时间为%f: '%(end-start))
value2,g_,h_,gg=predict(Data,max_value,var_n,const)
#print(value)
#print(np.linalg.norm(value-max_value))
#print(Data[h_])