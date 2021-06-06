# ml_practice
Machine learning practice with pytorch

### Quick Start Guide
From the cloned repository, run the following commands in the terminal:

$ conda env create -f environment.yml  
$ conda activate ml_env

To utilize GPU (Optional):

$ pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

If using pycharm, set the interpreter to the python version in the created conda env e.g:

.../anaconda3/envs/sheep_env/bin/python

When adding or removing a dependency from the environment.yml list, run:  
$ conda env update --file environment.yml

To run Tensorboard enter:  
$ tensorboard --logdir=/path/to/output/logs/folder/  
or  
$ tensorboard --logdir runs

### Used Sources/Dependencies
#### Python Engineer
dataset_dataloader_09: https://www.youtube.com/watch?v=PXOzkkB5eH0   
digit_recognition_13: https://www.youtube.com/watch?v=oPhxf2fXHkQ   
image_recognition_14: https://www.youtube.com/watch?v=pDdP0TFzsoQ   
save_load_17: https://www.youtube.com/watch?v=9L9jEOwRrCg   
rnn_lstm_18: https://www.youtube.com/watch?v=0_PgWWmauHk   


### System Dependencies:
TBD

### TODO:
TBD
