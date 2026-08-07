# 树莓派拼图视觉系统

本项目运行在 Raspberry Pi 4B，使用 USB 摄像头识别绿色 A4 卡纸上的
拼图碎片，计算吸取位置、目标位置和旋转角，并通过可靠串口协议把完整
方案发送给 ESP32。

## 三种模式

| 模式 | 任务 | 求解策略 |
|---|---|---|
| 1 | 固定四块拼图 | 固定形状模板匹配 |
| 2 | 未知纯色碎片 | 多轮廓候选与矩形几何拼接 |
| 3 | 扑克牌碎片 | 几何拼接、图案连续性和180°对称评分 |

纸面坐标固定为：

- 原点：A4左上角；
- X轴：沿210 mm短边向右；
- Y轴：沿297 mm长边向下；
- 旋转角：顺时针为正，逆时针为负，范围 `[-180°, 180°)`。

## 主要文件

| 文件 | 用途 |
|---|---|
| `main.py` | 树莓派统一入口 |
| `rpi_realtime_detection.py` | USB相机、实时显示和运行状态 |
| `fragment_vision.py` | A4矫正、碎片分割和轮廓候选 |
| `vision_state_machine.py` | 三种模式的统一求解接口 |
| `puzzle_solver.py` | 几何拼接搜索 |
| `card_pattern_solver.py` | 扑克牌图案评分 |
| `esp32_puzzle_link.py` | COBS、CRC-8、ACK和结果打包 |

## 安装

```bash
cd ~/vision
python3 -m pip install -r requirements.txt
```

## 单机测试

指定模式直接运行，不连接ESP32：

```bash
python3 main.py \
  --camera 0 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --rotate 90 \
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
python3 main.py \
  --camera 0 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --rotate 90 \
  --detect-every 3 \
  --serial-device /dev/serial0 \
  --serial-baud 115200
```

如果使用树莓派Wayland显示器：

```bash
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export WAYLAND_DISPLAY=wayland-0
export QT_QPA_PLATFORM=wayland
```

ESP32可靠发送 `0x03 + uint8 mode`。树莓派收到后运行对应模式，
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

协议测试：

```powershell
& 'D:\Anaconda\python.exe' -m unittest -v test_esp32_puzzle_link.py
```
