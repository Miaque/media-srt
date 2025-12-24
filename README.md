# 运行

## CPU 版本

先执行
```shell
uv pip install torch==2.1.1+cpu torchvision==0.16.1+cpu torchaudio==2.1.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

再执行
```shell
uv pip install -r requirements.txt
```

## GPU 版本

删除 cpu 版本依赖
```shell
uv pip uninstall torch torchvision torchaudio
```

安装 gpu 版本
```shell
uv pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 torchaudio==2.1.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

## 环境快速检查

```shell
python - <<EOF
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
EOF
```
