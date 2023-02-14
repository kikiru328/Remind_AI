docker run -it --gpus device=1 --name khkim_torch -p 51620:51620 -p 39390:39390 -v /mnt/dataset/KiTs21:/root/project/kits -v /home/khkim/project:/root/project  pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime /bin/bash
#docker run -it --gpus device=1 --name khkim_torch -p 51620:51620 -p 39390:39390 -v /home/khkim/project:/root/project pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime /bin/bash
