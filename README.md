# blockGPT

## 代码运行方式
- main->train->detection
    - main中集成了构建交易embedding、构建词汇表、为token打上one hot标签，将生成的embedding和one hot保存到文件中
    - train中集成了加载数据、调用encoder、模型训练，加载数据时去文件中到embedding，同时因为token的数量少于256，所以补0对齐到256，每个交易的embedding大小为【256，64】，经过encoder后变为【256，词汇表大小】方便和one hot做交叉熵
    - detection用于为每个交易计算loss，评估模型能检测出几个异常合约，如果检测出一个交易，则认为检测出了这个异常合约。
    
## 获取数据
1. 修改Geth，记录trace和相关opcode执行的结果

## 处理数据
1. 将trace处理成树的方式
2. 为训练数据构建词汇表和one hot编码
3. 将交易映射到[256,64]大小的embedding矩阵

## 训练数据
1. 使用默认的transformer encoder结构
2. 在encoder中使用mask矩阵
3. 使用交叉熵计算encoder的输出和onehot标签完成反向传播

## 检测模型
1. 直接将每一个交易丢入模型，按loss为交易排行


