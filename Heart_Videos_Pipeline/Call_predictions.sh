l=$(date -d "$D" '+%d'-'%m'-'%H')
for channel in  3 4
do
    for kfold in  0 1 2 3 4 5 6 7 8 9
    do
     folderCh="$channel"ch
     cp Get_predictions.py ./$folderCh/Fold$kfold
     cp Get_predictions.sh ./$folderCh/Fold$kfold

     cd $folderCh/Fold$kfold
     if [ ! -e  "testpredictions_augmented.csv"  ] #
            then  
                fold="test"
                out_file="Pred_test-$l.out"
                err_file="Err-pred_test.err"
                job_name=$channel'CTe'$kfold'.job'
                echo No Test predictions - $job_name
                sbatch --error=$err_file --output=$out_file --job-name=$job_name -t 0-1:00 --mem=10GB -c2 -p gpu --gres=gpu:1 Get_predictions.sh $channel $kfold $fold
         fi
     
     if [ ! -e  "valpredictions_augmented.csv"  ] #
            then  
                fold="val"
                out_file="Pred_val-$l.out"
                err_file="Err-pred_val.err"
                job_name=$channel'CVa'$kfold'.job'
                echo No Val predictions - $job_name
                sbatch --error=$err_file --output=$out_file --job-name=$job_name -t 0-1:00 --mem=10GB -c2 -p gpu --gres=gpu:1 Get_predictions.sh $channel $kfold $fold
         fi
         
     if [ ! -e  "trainpredictions_augmented.csv"  ] #
            then  
                fold="train"
                out_file="Pred_tr-$l.out"
                err_file="Err-pred_tr.err"
                job_name=$channel'CTr'$kfold'.job'
                echo No Train predictions - $job_name
                sbatch --error=$err_file --output=$out_file --job-name=$job_name -t 0-5:00 --mem=10GB -c2 -p gpu --gres=gpu:1 Get_predictions.sh $channel $kfold $fold
         fi
         
     cd ../..
     done
done
