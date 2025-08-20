## 在Bohrium开发机上使用Google ADK测试的方法

### 复制结构文件

为了让智能体能读取到测试用例中的结构文件，需要将`strus` 目录中的结构文件复制到`/personal/google-adk-eval-stru`文件夹
内：
```bash
cp stru/* /personal/google-adk-eval-stru
```

### 复制evalset到智能体目录下
将`evalsets`目录下的所有json文件复制到自己的智能体目录下（与`agent.py`在同一个目录）。

### 启动 Agent
#### web网页
```bash
adk web --port 50002 --host 0.0.0.0
```
