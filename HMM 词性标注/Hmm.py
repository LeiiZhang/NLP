# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import codecs

ww = []
pos = []
fre = {}
pi = {}
A = {}
B = {}
dp = []
pre = []
zz = {}

fin = codecs.open("treatmenttrain.txt", "r", "utf-8")
while (True):
    text = fin.readline()
    if (text == ""):
        break
    tmp = text.split(" ")
    n = len(tmp)
    for i in range(0, n - 1):
        word = tmp[i].split('/')
        if (word[0] not in ww):
            ww.append(word[0])
        if (word[1] not in pos):
            pos.append(word[1])

n=len(pos)
for i in pos:
    pi[i]=0
    fre[i]=0
    A[i]={}
    B[i]={}
    for j in pos:
        A[i][j]=0
    for j in ww:
        B[i][j]=0

line=0
fin=codecs.open("treatmenttrain.txt","r","utf-8")
while(True):
    text=fin.readline()
    if(text=="\n"):
        continue
    if(text==""):
        break
    tmp=text.split(" ")
    n=len(tmp)
    line+=1
    for i in range(0,n-1):
        word=tmp[i].split('/')
        pre=tmp[i-1].split('/')
        fre[word[1]]+=1
        if(i==1):
            pi[word[1]]+=1
        elif(i > 0):
            A[pre[1]][word[1]]+=1
        B[word[1]][word[0]]+=1

cx={}
cy={}
for i in pos:
    cx[i]=0
    cy[i]=0
    pi[i]=pi[i]*1.0/line
    for j in pos:
        if(A[i][j]==0):
            cx[i]+=1
            A[i][j]=0.5
    for j in ww:
        if(B[i][j]==0):
            cy[i]+=1
            B[i][j]=0.5

for i in pos:
    pi[i]=pi[i]*1.0/line
    for j in pos:
        A[i][j]=A[i][j]*1.0/(fre[i]+cx[i])
    for j in ww:
        B[i][j]=B[i][j]*1.0/(fre[i]+cy[i])

print "训练结束"

#用于验证
fin = codecs.open("treatmentyanzheng.txt","r","utf-8")
Num = 0
indiff = 0
ii = 0
while(True):
    txt = []
    poss = []
    text = fin.readline()
    if (text == ""):
        break
    tmp = text.split(" ")
    n = len(tmp)
    for i in range(0, n - 1):
        word = tmp[i].split('/')
        txt.append(word[0])
        poss.append(word[1])
    num=len(txt)
    for i in range(0,num):
        txt[i]=unicode(txt[i])
    dp=[{} for i in range(0,num)]
    pre=[{} for i in range(0,num)]

    for k in pos:
        for j in range(0,num):
            dp[j][k]=0
            pre[j][k]=""
    n=len(pos)
    ii += 1
    print ii
    for c in pos:
        if(B[c].has_key(txt[0])):
            dp[0][c]=pi[c]*B[c][txt[0]]*1000
        else:
            dp[0][c]=pi[c]*0.5*1000/(cy[c]+fre[c])
    for i in range(1,num):
        for j in pos:
            for k in pos:
                tt=0
                if(B[j].has_key(txt[i])):
                    tt=B[j][txt[i]]*1000
                else:
                    tt=0.5*1000/(cy[j]+fre[j])
                if(dp[i][j]<dp[i-1][k]*A[k][j]*tt):
                    dp[i][j]=dp[i-1][k]*A[k][j]*tt
                    pre[i][j]=k
    res={}
    MAX=""
    for j in pos:
        if(MAX=="" or dp[num-1][j]>dp[num-1][MAX]):
            MAX=j
    i=num-1
    while(i>=0):
        res[i]=MAX
        MAX=pre[i][MAX]
        i-=1
    Num += num
    for i in range(0,num):
        if res[i] == poss[i]:
            indiff += 1

print "验证完成\n"
print "准确率：", (1.0 *indiff / Num)