# 评测图与获取方式

- [benchmark.txt](benchmark.txt)：人读清单、直接下载链接、构造规则、规模与运行命令。
- [benchmark.json](benchmark.json)：机器可读来源、参数、参考计数。
- [prepare_benchmark.py](../scripts/prepare_benchmark.py)：下载/解压、确定性生成、列出批量输入；不需要第三方 Python 库。

当前 `core` 为 **7 张真实图 + 7 张合成图，共 14 张**。`extended` 另加 Orkut 和 K1024 团链。选图参考 WeTriC 表 3 和 TRUST 表 2，合成图为课程样例及自行设计的机制分析图；Graph500 尚未纳入可执行套件。

```bash
python3 scripts/prepare_benchmark.py list
python3 scripts/prepare_benchmark.py download --ids W1 W2 W3 W4
python3 scripts/prepare_benchmark.py generate --suite core
```

大型边表不提交 Git。默认下载路径仍兼容已有 `data/com-amazon.ungraph.txt`、`data/roadNet-CA.txt` 和 `data/web-Google.txt`；已有文件不覆盖。新文件写入 SHA256/参数侧录，生成图位于 `data/generated/`。

**不要用 `data/*.txt` 直接收集输入**，因为其中包括说明文件 `benchmark.txt`。使用脚本 `paths` 动作导出已准备好的图路径。

这是探索集，参考计数不代表已在当前机器实测。先检查正确性，再比较同一设备上的完整任务时间；后续论文实验另行冻结配置和评测划分。
