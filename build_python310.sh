sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.10
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2
sudo update-alternatives --config python3
sudo rm /usr/bin/pip
sudo ln -s /usr/local/bin/pip3 /usr/bin/pip
wget https://bootstrap.pypa.io/get-pip.py
python3.10 get-pip.py
pip install --upgrade pip
pip install --upgrade setuptools