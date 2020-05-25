## 代码阅读：
1. 输入类似于 Bert，包含 \[User\] embedding 和 \[Sys\] embedding；
2. 多轮对话的处理方式——将 context 拼接成一句话，感觉可以优化，使用层次结构；
3. svt先 encoder 编码 context，之后 decoder 是先 n_var_layers 层包含隐变量的解码层，再包含 n_layers - n_var_layers 层 common decoder。其中，隐变量解码层过程如下：两条路径，prior 和 posterior（不包含上三角矩阵 mask），其他输入相同，如都包含 self-attention, decoder-encoder attention等。
4. gvt 是直接使用 Encoder 结构编码 context 和 无掩码的 target，之后采样隐变量。

## motivation
1. 层次结构；（代码已实现，完善评测指标）
2. 进一步地探索隐变量的构建；
3. 离散隐变量.