# pokemon
大木博士模拟器/基于树莓派4B与yolov8n-cls模型的宝可梦图鉴机实现

最近在B站看到了使用K210复现宝可梦图鉴的视频，视频指路：[宝可梦图鉴](https://www.bilibili.com/video/BV1nkudzeEwW/)
在观察了他的复现过程后，我有了尝试的冲动，教程指路：[K210复现宝可梦图鉴](https://www.cnblogs.com/xianmasamasa/p/18995912)
我手中只有一个树莓派4B，虽然没有了摄像头与语音模块，呈现结果会大打折扣，但是仅仅是实现逻辑功能就让我非常兴奋！
所以这个帖子将包含以下内容：

 - 树莓派4B系统构建
 - yolov8b-cls模型训练
 - pt模型权重文件转onnx模型权重文件

**硬件准备**：树莓派(4B)； 主机(推荐3060及以上)； 32/64G内存卡，不推荐太大的内存卡，会影响系统启动速度； 读卡器；
**软件准备**：putty（ssh连接工具）；VNC-viewer（远程桌面连接）；FinalShell（远程文件管理）；
# 树莓派4B初始化
树莓派的教程全网都不算太多，我参考的也是很老的教程，但是实际上树莓派的官网一直在更新，如今最新的树莓派系统镜像写入程序已经非常的简易！
1. 下载树莓派镜像写入软件
下载链接指路：[树莓派软件下载](https://www.raspberrypi.com/software/)
![树莓派下载](https://i-blog.csdnimg.cn/direct/cbf3d142aad44233a1b20a53f1807d61.png)
2. 写入镜像
将内存卡放入读卡器插入主机，类似于制作win的启动盘，但是可以存在可以自定义的内容
通用部分的内容只有WLAN一定需要填写，方便树莓派启动后自动连接wifi
服务部分的SSH也一定要打开，这样在树莓派不连接屏幕的情况下也能够开启VNC服务，之后就可以在主机中操作树莓派的图形化界面！
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/24e3b0a863084c5296cea230d3c7c617.png)
自定义的内容填写完成后点击保存，等待软件将镜像写入到内存卡中即可，写入之前软件会提示您它会将内存卡中的文件全部清除，这也是必要的。烧录的这段时间可能很长，如果发生读卡器与主机断连的情况，建议插拔换个插口以及重新写入！
系统写入成功后，将内存卡取出插入树莓派，树莓派4B的内存卡插槽位置在树莓派的反面~
3. 查看树莓派的ip
查看树莓派ip的方法有很多，最简单的方法就是就如wifi的后台查看树莓派的ip，如果你在第二部的自定义配置中设置了主机名， 那么在路由器的后台界面就能够看到该主机名。
路由器的后台网址通常可以在cmd中使用ipconfig命令看到
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/daa0c299c94b42fd8a49f9a2bb833f08.png)
无线网络适配器的默认网关一般就是路由器的后台管理界面，密码默认admin或者是wifi的密码
4. 树莓派系统配置
由于树莓派的系统中默认安装的是nano而不是vim所以需要熟悉一下nano的操作：
编辑文件的命令与vim相同：nano *.txt
文件修改完成后需要 ctrl+O --> enter --> ctrl+X 进行文件保存
对树莓派的apt-get进行换源：

```bash
sudo nano /etc/apt/sources.list
#把原本的官方源用‘#’进行注释，而后添加下述镜像源
deb http://mirrors.tuna.tsinghua.edu.cn/raspbian/raspbian/ stretch main contrib non-free rpi
deb-src http://mirrors.tuna.tsinghua.edu.cn/raspbian/raspbian/ stretch main contrib non-free rpi
```

对pip进行换源
新版本的树莓派系统已经默认安装python3，所以不需要额外的分别处理pip与pip3的换源，仅需要：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
如果不灵那还是建议采用up主：同济子豪兄的方法去改写pip.conf

```bash
sudo mkdir ~/.pip
cd .pip
sudo nano pip.conf

#输入以下内容
[global]
timeout = 10
index-url =  http://mirrors.aliyun.com/pypi/simple/
extra-index-url= http://pypi.douban.com/simple/
[install]
trusted-host=
    mirrors.aliyun.com
    pypi.douban.com
```

打开VNC

```bash
sudo raspi-config
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/92bb693eef5943c9a294fe527f613b82.png)![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0f753d88701541549249f920e97a14cf.png)![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4b587e7c87fc4a259f5c92e4dce89c2c.png)
OK，这样就可以使用VNC愉快的连接了，前提是主机和树莓派在同一个局域网中！

# 模型训练
首先自然是配置环境，在主机中创建yolo能够运行的环境：


conda中创建环境所需的yaml文件已上传
环境创建完成后就可以训练（更像是微调预训练的yolov8n-cls轻量化模型）

```python
python train_cls.py
```

训练过程示意：![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f6e8452e1a284b269498944baae7e8fc.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8bab676b66cb4161abbe1faafd033be0.png)

混淆矩阵：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a85031f5d9334a9e8b3d073b4a7af4f4.png)
模型训练完成后得到易于在树莓派端部署的onnx模型权重。

# 树莓派部署
树莓派端同样需要创建虚拟环境用以更好的管理项目：

```powershell
python -m venv myenv
```

requirements.txt中包含树莓派端环境依赖。
将之前模型训练得到onnx模型权重复制到树莓派中，在树莓派中进行模型加载：


关于文本转语音：自行配置的pyttsx3包以及espeak包，后者非常难听的机械音，前者一直报错，迫不得已部署了clash，使用的google tts，啊，相当好用。
BTW，部署clashs时，需要订阅链接生成config.yaml，如果直接使用

```powershell
wget -O config.yaml [订阅链接]
```
使用上述命令貌似生成的yaml文件内是一大串字符，建议将主机的yml文件改个名字与后缀丢到树莓派中就行

执行入口：
```python
python deploy_interactive.py
```

# 成功演示
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/57977c54f1864ebdaa0e0f8e98d9cb3d.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d78b1898d8f5451493fa2722aee03adc.png)

相关教程指路：
【1】[同济子豪兄](https://www.bilibili.com/video/BV1pb411g7Bn/)
【2】[K210复现宝可梦图鉴](https://www.cnblogs.com/xianmasamasa/p/18995912)
【3】[pyttsx3安装](https://blog.51cto.com/u_16213363/11863082)
【4】[树莓派安装clash](https://github.com/Xizhe-Hao/Clash-for-RaspberryPi-4B?tab=readme-ov-file)
【5】[树莓派4B介绍](https://blog.csdn.net/bhniunan/article/details/104783321)
【6】[yolov8预训练权重下载及配置](https://github.com/RhineAI/YOLOv8/blob/main/README.zh-CN.md)
【7】[同济子豪兄github相关教程](https://github.com/TommyZihao/ZihaoTutorialOfRaspberryPi/blob/master/%E7%AC%AC3%E8%AE%B2%EF%BC%9A%E4%B8%80%E5%8A%B3%E6%B0%B8%E9%80%B8%E9%85%8D%E7%BD%AE%E6%A0%91%E8%8E%93%E6%B4%BE.md)
