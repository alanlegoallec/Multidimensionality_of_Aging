l=$(date-"$D" '+%d'-'%m'-'%H')

for channel in  3 4
do
    for kfold in  0 1 2 3 4 5 6 7 8 9
    do
         folderCh="$channel"ch
         folder="Fold$k"
         cp Train.py ./$folderCh/$folder
         cp Train.sh ./$folderCh/$folder
         job_name="$channel"chF"$k".job

         cd $folderCh/Fold$kfold
         out_file="Out-$l.out"
         err_file="Err.err"
         echo $folder

         RES=$(sbatch --error=$err_file --output=$out_file --job-name=$job_name -t 0-12:00 -c4 --mem=16G -p gpu  --gres=gpu:teslaV100:1 Train.sh $k $channel) 
         echo ${RES##* } 
         scontrol top ${RES##* } 
         
         #launch a second trainning that should start right after
         RES=$(sbatch --error=$err_file --output=$out_file --job-name=$job_name -t 0-12:00 -c4 --mem=16G -p gpu --gres=gpu:teslaV100:1 --dependency=afterany:${RES##* } Train.sh $k $channel)
         echo ${RES##* } 
         
         cd ../..
    done
done

