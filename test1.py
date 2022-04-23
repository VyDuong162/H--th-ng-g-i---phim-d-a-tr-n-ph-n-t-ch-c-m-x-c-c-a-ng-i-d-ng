
s ="234*345*34+88+56*56+45+6*4"
b =[]
c =[]
d = ''

for i in s:
    if i.isnumeric():  
        d = d + i 
    else:
        c.append(int(d))
        c.append(i)
        d = ''
c.append(int(s[len(s)-1]))
print(b)   
print(c)
i=1
tong =c[0]
while i < len(c):
    if c[i]=='*':
        tong = c[i]*c[i+1]
        i= i+1+1
        continue
    if c[i]=='+':
        tong = tong+c[i+1]
        i= i+1+1
        continue
print(tong)
print(234*345*34+88+56*56+45+6*4)
