# You can find the following job in the file /sge-root/examples/jobs/simple.sh.
#!/bin/sh
source activate tensorflow_cpu
python DQN_LSTM_BlocksWorld.py
source deactivate tensorflow_cpu