# 实验记录协议

执行前阅读研究计划第 4 至 6 节。当前没有性能结果。

输入元数据：来源、原始 SHA256、清洗规则、清洗后 n/m、定向后短表长度与工作量代理分布。
环境元数据：GPU 型号、显存、驱动、nvcc、CPU、主机内存、系统、编译参数、提交 SHA。

计时分别记录 preprocessing_ms、classification_ms、h2d_ms、initialization_ms、count_kernel_ms、reduction_ms、d2h_ms、e2e_ms、full_with_io_ms。
host 墙钟与 GPU events 的时间不机械相加；端到端由独立主机计时实测，结束前同步。

配置：先预热 5 次、测量 30 次；版本顺序交错，保存全部原始记录。阈值在独立开发图上选择，主测试冻结；敏感性扫描单列。
一次执行与 R=10/100 重复场景分开，明确哪些准备成本复用。每个版本输出完全相同的三角形总数。

空 CSV 为表头模板，不表示完成测量。失败、OOM、超时写入 status，不静默删除。
