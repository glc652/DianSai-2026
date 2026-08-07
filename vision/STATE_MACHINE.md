# 树莓派—ESP32视觉状态机

## 主流程

```text
WAIT_COMMAND
  ↓ 收到可靠消息 0x03 + mode
DETECTING
  ↓ 碎片数量连续两次稳定
SOLVING
  ├─ 失败 → ERROR → WAIT_COMMAND
  └─ 成功
SENDING_COUNT
  ↓ ESP32 ACK
SENDING_PIECES
  ↓ 每块分别等待 ESP32 ACK
WAIT_COMMAND
```

一条题号命令只冻结并求解一次。发送结果期间收到的新题号会被暂存，
当前方案全部发送完成后再执行。

## 题号消息

ESP32向树莓派可靠发送：

```text
TYPE    = 0x03
PAYLOAD = uint8 mode
```

合法模式为 `0`、`1`、`2`、`3`、`4`。

模式 `0` 用于运行过程中建立固定蓝色卡纸标定：

1. ESP32 第一次可靠发送 `TYPE=0x03, PAYLOAD=0`，树莓派进入标定状态；
2. 放置并对准 `105 x 297 mm` 湖蓝色半张 A4；
3. 树莓派连续检测角点，至少两帧稳定后标记为可保存；
4. ESP32 第二次可靠发送 MODE 0，树莓派保存
   `rpi_camera_calibration.json` 并立即热更新；
5. 移走蓝色卡纸，随后可直接发送模式 `1`、`2`、`3` 或 `4`。

如果第二次 MODE 0 到达时角点还没有连续两帧稳定，树莓派不会写入
不可靠标定，也不会退出标定状态；等待稳定后再次发送 MODE 0 即可。
MODE 0 只使用可靠链路本身的 ACK，不发送拼图数量或碎片坐标。

## 结果消息

树莓派首先可靠发送碎片数量：

```text
TYPE    = 0x01
PAYLOAD = uint8 count
```

随后按机械执行顺序逐块可靠发送：

```text
TYPE    = 0x02
PAYLOAD = >Bhhhhh
```

| 字段 | 类型 | 单位 |
|---|---|---|
| `piece_id` | `uint8` | 从1开始的发送顺序 |
| `source_x` | `int16` | mm |
| `source_y` | `int16` | mm |
| `target_x` | `int16` | mm |
| `target_y` | `int16` | mm |
| `rotation` | `int16` | 0.01°，顺时针为正 |

当前ESP32固件接受1至4块碎片，并要求坐标位于
`X=0..210 mm`、`Y=0..297 mm`。

## 链路层

串口参数为115200 baud、8N1：

```text
线格式 = COBS(原始包) + 00
```

可靠消息：

```text
TYPE|80  SEQ  PAYLOAD  CRC
```

确认包：

```text
00  SEQ  CRC
```

CRC-8多项式为 `0x07`，初始值和结果异或值均为 `0x00`。发送端
每50 ms检查一次ACK，未确认时重发同一帧。

## 失败行为

视觉失败时不会发送虚假坐标。实时程序记录失败后重新采集稳定帧并重试，
直到得到通过最终安全检查的方案，或收到可靠取消命令 `0xFF`。失败期间
不得发送占位坐标或部分动作。

Mode3只负责不规则扑克碎片的几何拼接。Mode4负责直角矩形扑克碎片的数字
角标、长短双轴和180°花纹验证；拼缝检测已完全停用，不再计算
`seam_mismatch_ratio`，并以 `seam_diagnostics_computed=false` 标记。Mode4候选
歧义或像素证据不足时安全拒绝，不发送动作。
