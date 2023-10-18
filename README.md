# Computer-Vision-SFM

## install
You need to run 'install.sh' before execution the 'main.py'
'''
sudo chmod 777 install.sh
./install.sh
'''

## run
'''
python3 main.py
'''
The final estimated 3D point cloud will be stored in results.npy

### visual interface
'''
python3 PtsVisualizer/visualize.py results.npy
'''
if 'ModuleNotFoundError: No module named 'glm''
install GLM package
'''
cd ~/Downloads
git clone https://github.com/N0rbert/PyGLM.git
cd PyGLM
debuild -b -uc -us

sudo apt-get install ../python3-pyglm_2.3.1_amd64.deb
'''
