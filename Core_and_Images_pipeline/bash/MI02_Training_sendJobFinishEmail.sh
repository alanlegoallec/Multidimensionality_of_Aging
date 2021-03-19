#!/bin/bash
to="alanlegoallec@g.harvard.edu"
flag=$1
outfile="../eo/$flag.out"
[ ! -f ../eo/$flag.success ] && s="Subject: Failed: job id:$SLURM_JOBID name:$SLURM_JOB_NAME\n" ||  s="Subject: Success: job id:$SLURM_JOBID name:$SLURM_JOB_NAME\n"
stat=`tail -n 1 $outfile`
[[ "$stat" == *COMPLETED* ]] && echo *Notice the sacct report above: while the main job is still running for sacct command, user task is completed. >> $outfile
line_last_improvement=$(cat $outfile | grep "improved from" | tail -1)
toSend=`echo Last improvement:; echo $line_last_improvement`
toSend="$toSend\n...\n`echo Job output:; tail -n 15 $outfile`"
echo -e "$s\n$toSend" | sendmail $to

