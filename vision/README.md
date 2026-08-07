# 树莓派拼图视觉系统

本项目运行在 Raspberry Pi 4B，使用 USB 摄像头识别绿色 A4 卡纸上的
拼图碎片，计算吸取位置、目标位置和旋转角，并通过可靠串口协议把完整
方案发送给 ESP32。

完整且优先级最高的项目上下文见 [`AGENTS.md`](AGENTS.md)。

## 四种模式

| 模式 | 任务 | 求解策略 |
|---|---|---|
| 1 | 固定四块拼图 | 固定形状模板匹配 |
| 2 | 未知纯色碎片 | 多轮廓候选与矩形几何拼接 |
| 3 | 不规则扑克牌碎片 | 轮廓几何搜索与矩形目标约束 |
| 4 | 直角矩形扑克牌碎片 | 数字角标、双轴花纹和180°验证 |

纸面坐标固定为：

- 原点：A4左上角；
- X轴：沿210 mm短边向右；
- Y轴：沿297 mm长边向下；
- 旋转角：顺时针为正，逆时针为负，范围 `[-180°, 180°)`。

## 主要文件

| 文件 | 用途 |
|---|---|
| `rpi_realtime_detection.py` | 正式入口：相机、实时显示、ESP32控制 |
| `fragment_vision.py` | A4矫正、碎片分割和轮廓候选 |
| `vision_state_machine.py` | 四种模式的统一求解接口 |
| `puzzle_solver.py` | 几何拼接搜索 |
| `card_pattern_solver.py` | 扑克牌图案评分 |
| `esp32_puzzle_link.py` | COBS、CRC-8、ACK和结果打包 |
| `native/vision_fast.cpp` | Mode2/3/4几何热点的C++/OpenMP加速 |
| `setup_fast.py` | 构建可选的 `vision_fast` 原生扩展 |

## 安装

```bash
cd ~/vision
python3 -m pip install -r requirements.txt
sudo apt install -y v4l-utils build-essential python3-dev
python3 setup_fast.py build_ext --inplace
```

## 单机测试

指定模式直接运行，不连接ESP32：

```bash
python3 rpi_realtime_detection.py \
  --camera 0 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --rotate 180 \
  --detect-every 3 \
  --mode 2
```

快捷键：

- `P`：立即求解；
- `R`：清除结果并重新检测；
- `S`：保存原图、检测图和JSON；
- `Q` 或 `Esc`：退出。

## ESP32联调

先确认树莓派板载UART：

```bash
ls -l /dev/serial0
```

启动等待题号的自动模式：

```bash
python3 rpi_realtime_detection.py \
  --camera 0 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --rotate 180 \
  --detect-every 3 \
  --solver-workers 3 \
  --serial-device /dev/serial0 \
  --serial-baud 115200
```

如果使用树莓派Wayland显示器：

```bash
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export WAYLAND_DISPLAY=wayland-0
export QT_QPA_PLATFORM=wayland
```

ESP32可靠发送 `0x03 + uint8 mode`。`mode=0` 第一次进入蓝色卡纸
标定，第二次保存并立即启用标定；`mode=1/2/3/4` 运行对应求解模式。
树莓派完成求解后，
再可靠发送：

1. `0x01 + uint8 count`；
2. 每块发送 `0x02 + >Bhhhhh`，字段为编号、起点X/Y、终点X/Y和
   0.01°旋转角。

每包收到ACK后才发送下一包。全部确认后，树莓派重新等待题号。
详细流程见 `STATE_MACHINE.md`。

## 本地图片回归

```powershell
& 'D:\Anaconda\python.exe' .\vision_state_machine.py `
  --demo 'C:\Users\77019\Desktop\rpi_solve_input.jpg' `
  --demo-mode 2
```

完整测试：

```powershell
& 'D:\Anaconda\python.exe' -m pytest -q
```

当前原生模式基线为 `41 passed`。原生编译、验证和回退见
[`NATIVE_ACCELERATION.md`](NATIVE_ACCELERATION.md)。
