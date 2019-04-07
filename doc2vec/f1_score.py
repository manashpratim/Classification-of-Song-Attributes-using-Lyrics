import pickle
import sys


did = sys.argv[1]
did = str(did)

genre_list = ['Metal','Country','Religious','Rap','R&B','Reggae','Folk','Blues']
conf_mat= pickle.load(open("conf_mat_"+did+".pickle" , "rb"))
print ("Confusion Matrix")
for i in genre_list:
	for j in genre_list:
		print(conf_mat[i][j],end="\t")
	print(i)
sum=0
sum1=0
avg=0
w_avg=0
score={}
file=open("result_"+did+"_"+".xls","w")

print("Row_Sum  Col_Sum Precision \t\t Recall \t\t F_score \t\t Tag" )
for i in genre_list:
	sum=0
	sum1=0
	
	for j in conf_mat[i]:
		#print(conf_mat[i][j],end="\t")
		sum=sum+conf_mat[i][j]
		sum1=sum1+conf_mat[j][i]
	try:
		prec=conf_mat[i][i]/sum
	except ZeroDivisionError:
		prec = 0
	try:	
		recall=conf_mat[i][i]/sum1
	except ZeroDivisionError:
		recall = 0
	try:
		f=(2*prec*recall)/(prec+recall)	
	except ZeroDivisionError:
		f = 0
	avg=avg+f
	score[i]=f
	w_avg=w_avg+f*sum
	print("%s \t %s \t %s \t %s \t %s \t %s"  % (sum,sum1,prec,recall,f,i))



for i in genre_list:
	print(score[i])
	file.write(i+"\t"+str(score[i])+"\n")
file.write("\n \n Average:\t"+str(avg/8))
file.write("\n Weighted Average:\t"+str(w_avg/1656)+"\n\n\n")	
print("Avg=%s" %(avg/8))
print("Weighted Average=%s" %(w_avg/1656))
