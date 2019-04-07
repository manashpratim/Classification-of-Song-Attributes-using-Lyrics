import pickle
import sys

k = sys.argv[1]
k = str(k)
did = sys.argv[2]
did = str(did)


conf_mat= pickle.load(open("conf_mat_genre_knn_"+k+"_"+did+".pickle" , "rb"))
print ("Confusion Matrix")
for i in conf_mat:
	for j in conf_mat[i]:
		print(conf_mat[i][j],end="\t")
	print(i)
sum=0
sum1=0
avg=0
w_avg=0
score={}
file=open("result_"+k+"_"+did+"_"+".xls","w")

print("Row_Sum  Col_Sum Precision \t\t Recall \t\t F_score \t\t Tag" )
for i in conf_mat:
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

genre_list = ['GENRE_Metal','GENRE_Country','GENRE_Religious','GENRE_Rap','GENRE_R&B','GENRE_Reggae','GENRE_Folk','GENRE_Blues']

for i in genre_list:
	print(score[i[6:]])
	file.write(i[6:]+"\t"+str(score[i[6:]])+"\n")
file.write("\n \n Average:\t"+str(avg/8))
file.write("\n Weighted Average:\t"+str(w_avg/1656)+"\n\n\n")	
print("Avg=%s" %(avg/8))
print("Weighted Average=%s" %(w_avg/1656))
