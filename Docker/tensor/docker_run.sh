docker run -it \
	--gpus device=1	\
	--name khkim \
	-v /home/khkim/project:/root/project \
	-v /mnt/dataset/BraTs20:/root/project/Brats/BraTs20 \
	-v /mnt/dataset/KiTs21:/root/project/kits\
	-p 36160:36160 \
	-p 44700:44700 \
	tensorflow/tensorflow:2.8.2-gpu-jupyter
