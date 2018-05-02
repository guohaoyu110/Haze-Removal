#!/bin/bash
#
#  Set shell, otherwise the default shell would be used
#$ -S /bin/bash
#
#  Set memory limit
#$ -l h_vmem=10G
#
#  Set wallclock time limit
#$ -l h_rt=05:59:00
#
#  Make sure that the .e and .o file arrive in the
#  working directory
#$ -cwd
#
#  Specify path to job's stdout output file
#$ -o /scratch_net/nowin/csakarid/distcomp/logs/HazeCityscapes
#
#  Merge the standard out and standard error to one file
#$ -j y
#
#  Declare job name
#$ -N HazeCityscapes
#
#  Force / switch off an immediate execution of the job
#$ -now y
#
## schedule 100 jobs with ids 1-100
#$ -t 1-100
#
source /home/sgeadmin/BIWICELL/common/settings.sh
/bin/echo Running on host: `hostname`
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`
/bin/echo PATH: `echo $PATH`
/bin/echo TMP: `env | grep TMP` 
#/bin/echo SGE: `env | grep SGE`
/bin/echo MCR: `env | grep MCR`
#
if [ ! -d "job_storage" ]; then
  mkdir job_storage 
fi
#
TASK_ID=${SGE_TASK_ID:-"$1"}
# binary to execute
/usr/sepp/bin/matlab -nodesktop -nodisplay -nosplash -r "Haze_simulation_Cityscapes($TASK_ID)"
echo finished at: `date`
exit 0;
