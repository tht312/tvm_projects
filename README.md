# 大幅面遥感图像深度学习编译优化

#### 介绍
- 该项目意在解决遥感领域使用深度学习处理大幅面图像时的困难。  
- 该项目通过编译优化的方法，在模型的中间表达（IR）上进行算子分解优化，即对算子进行数学上严格等价的切分，来减小单个算子计算时的输入大小。使得模型可以对任意大小的图像进行计算，而不必再担心显存不足的问题。  
- 该项目基于TVM Relay IR对FCN32s模型实现了算子分解编译优化算法，并借助于TVM后端实现了推理和训练。  


#### 安装教程
TVM源码基于TVM0.8 Release版，可以采用两种方式进行编译  
1. 直接下载仓库内的tvm代码并编译。所有需要对源码修改的位置已经修改完毕，具体编译过程可参考TVM官网。  
2. 基于自己的TVM版本进行修改。具体修改位置如下：  

- python部分，基础路径为tvm/python/tvm:  
&emsp;&emsp; - relay/op/nn/nn.py  
&emsp;&emsp; - relay/op/nn/_nn.py  
&emsp;&emsp; - relay/op/strategy/cuda.py  
&emsp;&emsp; - relay/op/strategy/generic.py  
&emsp;&emsp; - topi/cuda/conv2d.py  
&emsp;&emsp; - contrib/cudnn.py  
&emsp;&emsp; - relay/op/_tensor_grad.py  

- c++部分，基础路径为tvm/src:  
&emsp;&emsp; - relay/op/nn/convolution.cc  
&emsp;&emsp; - relay/op/nn/convolution.h  
&emsp;&emsp; - runtime/contrib/cudnn/conv_backward.cc  
&emsp;&emsp; - runtime/contrib/cudnn/cudnn_utils.h  

#### 使用说明

1.  编译TVM时需要llvm环境，Ubuntu环境下可以在llvm官网下载编译好的二进制版本，目前测试11和12均可以，13时tvm编译会报错。Windows环境由于二进制版本没有llvm-config，需要自己编译一遍llvm，这个过程比较麻烦。
2.  编译TVM需要cuda和cudnn环境。
3.  根据官网TVM源码编译教程，需要在build目录下拷贝一份cmake.config，本仓库已经提供，可根据具体情况修改llvm的路径。
4.  对TVM源码的修改主要包括添加对反向传播算子的支持和con2d算子反向传播时通过自定义算子调用cudnn库。
5.  使用时直接运行python_projects下的文件即可，各个文件的具体功能如下：   

    + 程序入口为tvm_train_fcn_gid和tvm_infer_fcn_gid，分别用来训练和推理，使用时需要修改数据集的路径以及输入图像大小等。  
    + module_,gradient_,optimizer,用来支持训练。其中module_负责训练整体的过程，gradient_实现反向传播，optimizer目前实现了Adam优化器。  
    + fcn_for_train是torch定义的fcn模型，pass_reset_input将torch模型导入TVM，并修改模型的输入大小。（因为TVM导入模型是通过CPU模拟一遍模型的计算过程来实现，因此输入尺寸较大时这个时间会很长，这里直接通过Pass修改模型输入大小）  
    + mypass1_2为一维的算子分解，使用时需要修改第68行N的值来改变切割的块数。mypass4_3是二维算子分解，也需要修改N的大小，表示每个算子切成N×N。  
    + pass_for_device负责将部分算子的计算放在GPU上，这一过程本可以在算子分解时完成，但是由于TVM开发早期在自动微分时对device总是存在BUG，因此单独拿出来在反向传播后的计算图上进行。  
    + 根据经验，对大幅面图像进行训练时最好使用预训练的模型对参数进行初始化，否则在softmax时由于数据过小会全部归一化，训练就会实效。可以用load_params读取预训练的模型参数。  
