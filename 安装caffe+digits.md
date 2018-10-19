# 安装caffe
先参照“从.run安装cuda+cuDNN.md”配置cuda+cudnn。

## 安装OpenBLAS以加速CPU推断速度
```sh
$ git clone https://github.com/xianyi/OpenBLAS.git
$ cd OpenBLAS
$ make OpenMP=1
$ sudo make install
```
默认安装位置是/opt/OpenBLAS

## 下载NVIDIA版caffe
```sh
$ git clone https://github.com/NVIDIA/caffe.git
```
## 安装caffe依赖的库
```sh
$ sudo pip install -r caffe/python/requirements.txt
$ cat caffe/python/requirements.txt | xargs -n1 sudo pip install
$ sudo apt-get install --no-install-recommends git graphviz python-dev python-flask python-flaskext.wtf python-gevent python-h5py python-numpy python-pil python-pip python-scipy python-tk -y
$ sudo apt-get install libatlas-base-dev -y
$ sudo apt-get install --no-install-recommends build-essential cmake git gfortran libatlas-base-dev libboost-filesystem-dev libboost-python-dev libboost-system-dev libboost-thread-dev libgflags-dev libgoogle-glog-dev libhdf5-serial-dev libleveldb-dev liblmdb-dev libopencv-dev libsnappy-dev python-all-dev python-dev python-h5py python-matplotlib python-numpy python-opencv python-pil python-pip python-pydot python-scipy python-skimage python-sklearn -y
$ sudo apt-get install libboost-all-dev -y
$ sudo apt-get install libgoogle-glog-dev -y
$ sudo apt-get install libprotobuf-dev protobuf-compiler -y
$ sudo apt-get install python-tk -y
$ sudo apt-get install libturbojpeg -y
$ sudo ln -s /usr/lib/x86_64-linux-gnu/libturbojpeg.so.0.1.0 /usr/lib/x86_64-linux-gnu/libturbojpeg.so
```

## 配置caffe
将Makefile.config.example的内容复制到Makefile.config并修改
```sh
$ cp Makefile.config.example Makefile.config
$ vim Makefile.config
```
若使用cudnn，则将
```sh
#USE_CUDNN := 1
```
修改成：
```sh
USE_CUDNN := 1
```
若使用的opencv版本是3.*的，则将
```sh
#OPENCV_VERSION := 3 
```
修改成：
```sh
OPENCV_VERSION := 3
```

## 编译caffe
```sh
$ cd caffe
$ sudo mkdir build
$ cd build
$ sudo cmake ..
$ sudo make -j8
$ sudo make install
```

# 安装digits

## 下载digits
```sh
$ git clone https://github.com/NVIDIA/DIGITS.git
$ mv DIGITS digits
$ sudo pip install -r digits/requirements.txt
$ sudo pip install -e digits
```
## 测试digits
```sh
$ cd digits
$ ./digits-devserver
```
