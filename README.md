# Computer-Vision-SFM

## install
You need to run 'install.sh' before execution the 'main.py'
```bash=1
sudo chmod 777 install.sh
./install.sh
```

## run
```bash=1
python3 main.py
```
The final estimated 3D point cloud will be stored in results.npy

## visual interface
```bash=1
python3 PtsVisualizer/visualize.py results.npy
```
- if 'ModuleNotFoundError: No module named 'glm''
> install GLM package
```bash=1
cd ~/Downloads
git clone https://github.com/N0rbert/PyGLM.git
cd PyGLM
debuild -b -uc -us

sudo apt-get install ../python3-pyglm_2.3.1_amd64.deb
```
