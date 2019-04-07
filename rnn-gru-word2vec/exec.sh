max=9
for(( i=1; i <= $max; ++i ))
do
    #python train.py 00$i
    #python train_doc2vec.py dbow 000$i
    python eval.py 00$i
    python f1_score.py 00$i
done

#python train.py 010
#python train_doc2vec.py dbow 000$i
python eval.py 010
python f1_score.py 010

python calc.py 10
