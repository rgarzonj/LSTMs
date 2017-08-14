# You can find the following job in the file /sge-root/examples/jobs/simple.sh.
#!/bin/sh
export PATH="/home/usuaris/rgarzonj/anaconda3/bin:$PATH"
source activate tensorflow_cpu
cd /home/usuaris/rgarzonj/github/LSTMs
python DQN_LSTM_BlocksWorld.py
source deactivate tensorflow_cpu