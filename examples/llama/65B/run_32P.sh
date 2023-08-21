
cmd="
ps -ef | grep \"python\" | grep -v grep | awk '{print \$2}' | xargs kill -9;
cd /home/j00648035/AscendSpeed;
conda activate j00648035;
source /home/j00648035/packages/FA0802/ascend-toolkit/set_env.sh;
export LD_LIBRARY_PATH=\$ASCEND_OPP_PATH/vendors/customize/op_api/lib/:\$LD_LIBRARY_PATH;
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/lib/;
bash examples/llama_task/65B/pretrain_32P.sh
"

cmd0="${cmd} 0"
cmd1="${cmd} 1"
cmd2="${cmd} 2"
cmd3="${cmd} 3"

pdsh -R ssh -w 10.170.27.117 -l root $cmd0 &
pdsh -R ssh -w 10.170.27.146 -l root $cmd1 &
pdsh -R ssh -w 10.170.27.209 -l root $cmd2 &
pdsh -R ssh -w 10.170.27.41 -l root $cmd3 &

echo $cmd