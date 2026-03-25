# Ares
Second team of rc2d's code repository

完整编译过程：

1. 安装依赖工具（应该可以跳过，毕竟已经安装过了）
先安装编译所需的基础依赖：
sudo apt update
sudo apt install -y build-essential cmake libboost-all-dev libqt5core5a libqt5gui5 libqt5widgets5 qtbase5-dev libfontconfig1-dev libfreetype6-dev libsdl1.2-dev libsdl-image1.2-dev libsdl-ttf2.0-dev libpng-dev libjpeg-dev libaudio-dev libaudiofile-dev
sudo apt install -y bison flex libboost-all-dev libssl-dev
sudo apt install -y build-essential cmake libboost-all-dev libqt5core5a libqt5gui5 libqt5widgets5 qtbase5-dev libfontconfig1-dev libfreetype6-dev libsdl1.2-dev libsdl-image1.2-dev libsdl-ttf2.0-dev libpng-dev libjpeg-dev libaudio-dev libaudiofile-dev

2. 编译安装 rcssserver：

cd rcssserver
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install

3. 编译安装 rcssmonitor

cd ../../rcssmonitor
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install

4. 验证安装
两个终端：
# 启动服务端
rcssserver
# 新开终端，启动监控端
rcssmonitor

如果能正常启动，说明安装成功。

5. 编译安装依赖库 librcsc

cd ../../librcsc
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig  # 刷新共享库缓存

6. 编译安装 helios-base

cd ../../helios-base
mkdir build && cd build
cmake ..
make -j$(nproc)

没有安装这一步

7.试运行：就是我之前的文档里面的的那几行代码（我有点困下次再打）
我试了一下，是可以运行的。在helios-base里面的build/bin文件夹里打开终端，输入./start.sh和./start.sh -t "abaaba" -C就可以试运行了。如果是自己改动的两支球队的话，可以把helios-base文件夹复制一个副本出来，作为对照组和修改好的进行比较。当然也可以探索别的方式。
