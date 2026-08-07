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

合法模式为 `1`、`2`、`3`。

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
| `rotation` | `int16` | 相对旋转量，0.01°，顺时针为正 |

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

视觉失败时不会发送虚假坐标，终端显示错误并重新等待题号。目前ESP32
协议没有定义视觉失败消息；如果需要在屏幕显示失败原因，应新增独立的
状态消息类型。
