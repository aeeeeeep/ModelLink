# 启动脚本使用指南

```shell
bash start.sh ${device_ids} ${max_memory_gb} ${running_mode}
```

运行

```shell
```shell
bash start.sh 2 35 
```

调试

```shell
bash start.sh 2 35 debug
```

## 推荐参数

| 场景      | max_memory_gb |
|---------|---------------|
| 310p 单芯 | 35            |
| 310p 双芯 | 35            |
| 910b 单卡 | 57            |
| 910b 双卡 | 57            |