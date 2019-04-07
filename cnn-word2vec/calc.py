import sys,numpy as np

did= sys.argv[1]
k=int(did)
did=str(did)
print(k)
count =0
head='Songs \t\t'
genre= ['Metal\t\t','Country\t\t','Religious\t\t','Rap\t\t','R&B\t\t','Reggage\t\t','Folk\t\t','Blues\t\t','Average\t\t','Weighted Average\t\t'] 
total = np.zeros(10)
final = open("final.xlsx","w")
for i in range (1,k+1):
	head +='run_'+str(i)+'\t'
	if i<10:
		f=open("result_00"+str(i)+"_.xls","r")
	else:
		f=open("result_0"+str(i)+"_.xls","r")	
	for line in f:
		line = line.split("\t")
		if(len(line)>1):
			genre[count%10] += str(line[1][:-1])+'\t'
			total[count%10] += float(line[1])
			count += 1
			
			print(line[1])
			
final.write(head+"\tOverall Average\n\n")
for i in range(10):
	final.write(genre[i]+"\t"+str(total[i]/k)+"\n")
print(count,str(count)+"Hi")

