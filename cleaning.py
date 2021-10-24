import sys

path = "E:\\n2v\\features\\drug_feature.txt"  # 数据来源
f = open(path , encoding='utf-8')
line = f.readline()
list = []
while line:
    a = line.split(" ")
    b = a[1:]
    list.append(b)
    line = f.readline()
f.close()

print(list)

with open('drug.txt', 'a') as month_file:  # 提取后的数据文件
    for line in list:
        s = ' '.join(line)
        month_file.write(s)
