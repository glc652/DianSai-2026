# Repository Guidelines

## 项目说明

本仓库是用于全国大学生电子设计竞赛的电机控制项目，包含 ESP32 固件、MaixCAM 视觉端程序和 Python 上位机。ESP32 固件基于 Arduino 框架和 PlatformIO，上位机使用 uv 管理。

## 项目结构

- `firmware/src/main.cpp`：固件入口，包含 `setup()` 和 `loop()`。
- `firmware/include/board/pins.h`：两套控制板的外设引脚映射。
- `firmware/include/`、`firmware/src/`：电机 SDK 公开接口和实现。
- `firmware/platformio.ini`：开发板、框架、依赖和串口配置。
- `host/`：Python 手柄和 MaixCAM 仿真调试程序。
- `maixcam/`：MaixVision 单文件视觉端串口程序。
- `docs/reference/`：赛题和硬件参考资料。
- `firmware/.clangd`、`.vscode/`、`.zed/`：编辑器及语言服务配置。

## 常用命令

- `pio run -d firmware -e esp32s3`：编译 ESP32-S3 N16R8 固件。
- `pio run -d firmware -e esp32`：编译旧 ESP32 固件。
- `pio run -d firmware -e esp32s3 -t upload`：编译并烧录 ESP32-S3 N16R8 固件。
- `pio run -d firmware -e esp32 -t upload`：编译并烧录旧 ESP32 固件。
- `./firmware/build.sh <esp32|esp32s3>`：更新指定控制板的编译数据库并烧录固件。
- `pio device monitor -d firmware`：打开串口监视器。
- `pio run -d firmware -e esp32s3 -t compiledb`：更新 clangd 使用的编译数据库，旧板开发时使用 `esp32` 环境。
- `pio run -d firmware -t clean`：清理固件构建产物。
- `uv sync --project host`：安装 Python 上位机依赖。
- `uv run --project host host/test_main.py`：运行 Python 上位机测试。

修改 ESP32 固件后分别运行 `pio run -d firmware -e esp32s3` 和 `pio run -d firmware -e esp32`。修改 Python 上位机时还需运行对应测试。不要提交 `.pio/` 和 `compile_commands.json` 等生成文件。

`host/main.py` 是 MaixCAM 视觉端的完整串口仿真。除图像采集和识别外，它与 `maixcam/maixvision_main.py` 对 ESP32 呈现的消息格式、发送顺序、可靠投递和运行生命周期必须一致；模拟数据可以不同，但必须满足固件接受的业务约束。修改 MaixCAM 与 ESP32 之间的通信行为时，同步更新 Host 仿真及其测试。

# Repository Rules

1. 任何模块对外暴露的接口，需要进行注释
2. 注释避免使用行尾注释，对于接口使用 Doxgen 风格块注释 `/* */`，对于过程性代码使用行注释 `//`
3. 注释末尾禁止使用句号

## Agent skills

### Issue tracker

任务与需求使用 GitHub Issues；外部 PR 不作为需求分类入口。详见 `docs/agents/issue-tracker.md`。

### Triage labels

使用 `needs-triage`、`needs-info`、`ready-for-agent`、`ready-for-human`、`wontfix` 五种分类标签。详见 `docs/agents/triage-labels.md`。

### Domain docs

采用单一领域上下文，使用根目录 `CONTEXT.md` 和 `docs/adr/`。详见 `docs/agents/domain.md`。
