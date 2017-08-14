# You can find the following job in the file /sge-root/examples/jobs/simple.sh.
#!/bin/sh
export PATH="/home/usuaris/rgarzonj/anaconda3/bin:$PATH"
source activate tensorflow_cpu
python /home/usuaris/rgarzonj/github/LSTMs/DQN_LSTM_BlocksWorld.py
source deactivate tensorflow_cpu