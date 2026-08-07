# 树莓派拼图视觉系统稳定版

> 历史归档：本文描述的是 2026-07-31 stable.4，不代表当前工作树。
> 当前四模式、C++/OpenMP、Mode4双轴规则和部署方式以根目录 `AGENTS.md`
> 与 `NATIVE_ACCELERATION.md` 为准。

版本：`v2026.07.31-stable.4`

## 当前状态

- 支持标定模式 0 和求解模式 1、2、3。
- 使用绿色卡纸作为碎片识别背景。
- 使用一次性湖蓝色半张 A4 卡纸建立固定全局坐标。
- 固定标定后，运行时不再依赖绿色 A4 边界重新标定。
- A4 坐标范围：`X=0..210 mm`，`Y=0..297 mm`。
- 碎片放在物理右半张纸识别，拼接目标位于物理左半张纸。
- 规范坐标对应关系：源区 `Y>148.5 mm`，目标原点 `[55, 30] mm`。
- 最终目标拼接间隙：`8 mm`。Mode 1 使用模板内预先保存的固定间隙终点，Mode 2～4 使用通用几何间隙算法。
- Mode 2/3 会测量拼接后真实外轮廓的四个角：相对 `90°` 最大误差 `5°`，真实角点距离最小外接矩形角点最多 `3 mm`；不再只凭外接矩形接受近似直角候选。
- 自动模式求解失败时不会结束当前 ESP32 任务或发送坐标：错误照常打印并追加到 `rpi_failure_history.jsonl`，等待 `0.75 s` 后重新采集两张稳定帧并无限重试，直到得到通过最终安全检查的唯一运动计划。
- ESP32 可随时用可靠控制帧 `TYPE=0x03, PAYLOAD=0xFF` 取消误触任务。树莓派会中断检测、求解、自动重试或结果发送，清空当前任务且不再发送坐标，然后回到 `WAIT_CMD`；ESP32 收到本地退出操作时也必须同步清空自身尚未执行的动作队列。
- Mode 3 搜索阶段允许最多 `15 mm²` 的临时轮廓重叠；输出前仍会消除重叠并严格检查为零。
- 旧版拼缝连续性检查已停用；当前 Mode4 不执行拼缝形态学或墨迹邻域计算。
- Mode 3 最终检查整张牌的灰度、彩色墨迹和 180° 对称性。
- 快速候选花纹不明确时自动扩大到 `beam=160`；最终候选分差不足会拒绝输出。
- 目标坐标使用旋转后的吸取点偏移，避免偏心吸取造成整体平移误差。

## 文件说明

- `rpi_calibrate_fixed.py`：湖蓝色半张 A4 固定标定。
- `rpi_camera_controls.py`：固定 UVC 参数的自动设置与回查。
- `rpi_realtime_detection.py`：树莓派实时识别、ESP32 串口控制。
- `fragment_vision.py`：固定标定、A4 矫正、背景与碎片轮廓检测。
- `vision_state_machine.py`：三个模式的求解流程和动作计划。
- `puzzle_solver.py`：几何拼接搜索。
- `card_pattern_solver.py`：Mode 3 扑克牌图案评分。
- `esp32_puzzle_link.py`：ESP32 可靠串口协议。
- `rpi_camera_calibration.json`：当前摄像头位置的固定标定数据。
- `fixed_puzzle_template.json`、`card_puzzle_template.json`：求解模板。

## 安装依赖

```bash
cd /home/debug1/vision
sudo apt update
sudo apt install -y v4l-utils
python3 -m pip install -r requirements.txt
```

`v4l-utils` 提供 `v4l2-ctl`，程序用它自动设置并回查摄像头参数。
如果缺少该工具，程序会明确报错，而不是继续使用不确定的相机默认值。

## 固定摄像头参数

以下参数是老USB摄像头当前验证值，已经保存在 `rpi_camera_controls.py` 和
`camera_profiles/old_usb_camera.json`。实时检测、固定标定和对焦预览每次打开
摄像头后都会自动应用；RaspberryPi-A摄像头使用独立JSON配置：

| 参数 | 数值 |
|---|---:|
| 自动白平衡 | 关闭 |
| 白平衡色温 | 4600 |
| 自动曝光 | 手动模式 |
| 曝光时间 | 320 |
| 动态帧率 | 关闭 |
| 亮度 | 0 |
| 对比度 | 42 |
| 饱和度 | 64 |
| 色调 | 0 |
| Gamma | 300 |
| 增益 | 55 |
| 电源频率 | 50 Hz |
| 锐度 | 70 |
| 背光补偿 | 0 |
| 数字变焦 | 0 |

正常启动时会显示：

```text
Fixed camera controls applied: /dev/video0 (15 values)
```

程序会逐项读取驱动中的实际值进行验证。如果使用不支持这些控制项的
其他型号摄像头，会明确报错。仅在诊断或兼容其他相机时，可以添加
`--skip-camera-controls` 跳过自动设置。

## 固定标定

正式联调时不需要先单独运行标定脚本，也不需要键盘按 `S`。

ESP32 控制流程：

```text
第一次 MODE 0 -> 开始实时蓝色卡纸标定
第二次 MODE 0 -> 保存并立即启用固定标定
MODE 1/2/3    -> 使用当前固定标定执行拼图
```

第二次 MODE 0 只有在角点连续两帧稳定后才会保存。如果尚未稳定，程序
会继续留在标定状态，稳定后让 ESP32 再发送一次 MODE 0。

如果启动时没有 `rpi_camera_calibration.json`，串口模式会正常等待
MODE 0；在完成标定前收到 MODE 1/2/3 会拒绝求解，不会输出错误坐标。

只有摄像头位置、角度、分辨率或旋转参数改变时才需要重新发送 MODE 0
完成标定。

下面的独立脚本仅作为离线调试备用：

将沿长边裁成一半的 `105 x 297 mm` 湖蓝色卡纸放在画面上半区域，然后运行：

```bash
python3 rpi_calibrate_fixed.py \
  --camera 0 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --rotate 180
```

画面稳定识别后按 `S`，生成 `rpi_camera_calibration.json`。保存成功后移走蓝色卡纸。

当前发布包内的标定文件只适用于生成它时的同一摄像头、同一固定位置和以下参数：

```text
1280 x 720
rotate = 180
```

## 正式运行

```bash
python3 rpi_realtime_detection.py \
  --camera 0 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --rotate 180 \
  --detect-every 3 \
  --serial-device /dev/serial0 \
  --serial-baud 115200
```

当前搬运方向固定为：

```text
物理右半张纸（放置散碎片） -> 物理左半张纸（输出拼接结果）
```

终端会显示：

```text
Workspace direction: source=physical right (canonical lower half),
target=physical left (canonical upper half)
```

不连接 ESP32、手动按 `P` 求解：

```bash
python3 rpi_realtime_detection.py \
  --camera 0 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --rotate 180 \
  --detect-every 3 \
  --manual
```

快捷键：

- `P`：求解。
- `R`：清除结果，准备下一次。
- `S`：保存当前检测数据。
- `Q` 或 `Esc`：退出。

## 正常启动标志

终端应显示：

```text
Fixed calibration loaded: rpi_camera_calibration.json
cal=fixed_upper_half_blue reg=saved_global
```

没有标定文件时，串口模式会显示：

```text
WAITING FOR ESP32 MODE 0 TO CREATE FIXED CALIBRATION
```

此时让 ESP32 执行两段式 MODE 0 标定即可。

## 回归测试

```bash
python3 -m unittest \
  test_card_artwork.py \
  test_inline_calibration.py \
  test_fragment_vision.py \
  test_esp32_puzzle_link.py \
  test_rpi_camera_controls.py
python3 -m py_compile \
  fragment_vision.py \
  puzzle_solver.py \
  card_pattern_solver.py \
  vision_state_machine.py \
  rpi_realtime_detection.py \
  rpi_calibrate_fixed.py
```

## 注意事项

- 摄像头固定标定后不要移动摄像头、支架或工作台。
- 改变分辨率或 `--rotate` 后必须重新标定。
- 识别碎片时使用绿色背景，蓝色半张 A4 只用于固定标定。
- 开始识别前必须把散碎片放在物理右半张纸；左半张纸应保持空白，作为拼接目标区。
- 不要把 `15 mm²` 临时重叠面积误认为 `15 mm` 摆放间隙；最终摆放间隙是 `8 mm`。
- Mode 3 花纹模糊时会执行更宽搜索，因此困难牌面可能明显慢于普通牌面；这是为了避免四片同形时强行输出错误花色。
- `rpi_failure_*`、`rpi_solve_*` 文件属于诊断数据，不是程序运行依赖。
