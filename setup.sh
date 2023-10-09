apt upgrade
apt update
apt install unrar rar
apt-get -y --force-yes install golang
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax jax_smi
pip install tensorflow==2.13 tensorflow-datasets webdataset keras-cv tf-models-official==2.13.1  tensorflow_addons
pip install albumentations einops tqdm matplotlib jax-smi pytorch_fid