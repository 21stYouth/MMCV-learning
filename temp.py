import os


flist = os.listdir('data/main/14')
for i in range(len(flist)):
    flist[i] += " " + flist[i] + " 0 \n"
print(os.listdir('data/main/14'))
f = open('data/main/14/test.txt', mode='w')
f.writelines(flist)