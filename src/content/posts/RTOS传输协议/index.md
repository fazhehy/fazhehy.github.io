---
title: RTOS传输协议
published: 2026-08-27
description: ''
image: ''
tags: [embedded, stm32, esp32]
category: 'embedded'
draft: false 
lang: ''
---

## 项目简介

本项目是一套基于 RTOS 的一主一从、逻辑半双工、固定帧长、请求—响应式通用通信框架。主机通过优先级队列统一调度多个业务任务的通信请求，协议内部提供事务匹配、固定帧封装、CRC 校验、超时重试、重复请求处理和设备状态管理等功能。

项目通过统一的 Transport 接口隔离协议逻辑和底层通信驱动，计划支持 UART、SPI、I2C 等多种通信方式，并适配 Arduino、STM32 和 Linux 等运行平台。底层驱动可以根据平台使用轮询、中断或 DMA，而不改变 Master、Slave 和 Frame 的核心逻辑。

当前实现状态：

| 平台/Transport       | 状态   |
| ------------------ | ---- |
| Arduino/ESP32 UART | 已实现  |
| Linux POSIX 串口     | 已实现  |
| STM32 UART         | 计划实现 |
| Arduino/STM32 SPI  | 计划实现 |
| Arduino/STM32 I2C  | 计划实现 |

协议的主要特点：

- **可靠性机制**：通过 CRC32、Transaction ID、超时重试、重复请求响应缓存和设备状态管理处理通信异常。
- **多优先级调度**：High、Normal、Low 三级队列保证控制请求优先，同时允许普通查询和后台数据传输共用总线。
- **并发访问安全**：业务线程只负责提交请求，唯一的 Protocol Master 串行操作 Transport，避免多个线程同时读写总线。
- **大规模数据传输**：业务层可以将文件、图像或固件分块传输，并在两个数据块之间处理更高优先级的请求。
- **业务与通信解耦**：协议层统一传输 Payload，业务消息格式可以根据具体产品独立定义。
- **跨平台与多总线扩展**：通过统一 Transport 接口隔离 UART、SPI、I2C 以及不同操作系统和 MCU 的驱动差异。
- **资源使用可预测**：固定帧长、固定请求对象池和静态 RTOS 队列减少运行时动态内存分配，适合嵌入式系统。
- **同步与异步调用**：Master 同时提供阻塞事务和异步完成回调，方便不同业务任务选择调用方式。

## 通信流程与实现原理

### 整体架构

```text
                     业务层

       Control        Sensor         Data
          │              │              │
        High           Normal           Low
          └──────────────┼──────────────┘
                         ↓
                  Priority Queues
                         ↓
                  Protocol Master
                         ↓
               Transaction + Frame
                         ↓
                     Transport
                         ↓
                  Protocol Slave
                         ↓
                 Business Callback
                         ↓
                  Response Frame
```

总线始终遵循一个请求对应一个响应：

```text
Master Request
      ↓
Slave Response
      ↓
Transaction Complete
```

Master 是总线的唯一发起方。一个事务没有结束时，不会开始另一个事务，因此逻辑上始终只有一个方向在使用总线。

### Master 请求流程

```text
业务任务提交 MasterRequest
            ↓
从对象池获取 RequestSlot
            ↓
进入 High / Normal / Low 队列
            ↓
Protocol Master 选择最高优先级请求
            ↓
生成 Transaction ID
            ↓
构建并发送固定长度 Request
            ↓
接收固定长度 Response
            ↓
校验 Frame 和 Transaction ID
            ↓
复制 Response Payload
            ↓
唤醒阻塞任务或调用完成回调
```

Master 内部只有一个协议执行线程或 RTOS 任务，因此多个业务线程不会同时操作 Transport。

请求被放入三个独立队列。调度器每次优先检查 High，然后检查 Normal，最后检查 Low。优先级切换发生在事务边界：高优先级请求可以插入两个低优先级文件块之间，但不会中断一个正在收发的事务。

### Slave 响应流程

```text
接收固定长度 Request
          ↓
检查 SOF / Version / Length / CRC32
          ↓
检查 Request Error Code
          ↓
检查是否为重复 Transaction ID
          ↓
调用业务回调处理 Payload
          ↓
构建并发送 Response
          ↓
缓存 Transaction ID 和 Response
```

Slave 只注册一个统一业务回调。协议层完成帧校验后，将有效 Payload 交给回调；回调负责解释业务内容并填写响应 Payload 和 Error Code。

如果收到与上一次相同的 Transaction ID，Slave 不会重复执行业务回调，而是重新发送缓存的响应。这可以避免 Master 因响应丢失而重试时，控制命令被重复执行。

### 超时与重试

以下问题属于通信异常：

- Transport 发送或接收失败。
- 接收超时。
- SOF 或 Version 错误。
- Payload Length 非法。
- CRC32 错误。
- Response Transaction ID 不匹配。

Master 遇到通信异常后重新初始化 Transport，并使用原 Transaction ID 重试当前事务。尝试次数为一次初始发送加 `maximumRetries` 次重试。

Slave 明确返回的业务 Error Code 表示通信已经成功，Master 会将结果直接返回给业务任务，不自动重试业务错误。

设备状态根据连续事务结果更新：

```text
通信成功                     → Online
单个事务重试耗尽             → Degraded
连续失败达到离线阈值         → Offline
```

### 大规模数据传输

协议核心不维护文件、图像或固件状态。大规模数据由业务层先拆成多个不超过 Payload 上限的数据块，再作为普通事务提交：

```text
Data Block 0  ── Low Priority Transaction
Data Block 1  ── Low Priority Transaction
Data Block 2  ── Low Priority Transaction
...
```

业务层可以为数据块增加文件编号、Offset、块长度和整文件摘要。因为每个块都是独立事务，所以控制和传感器请求可以在块之间得到及时处理。

## 固定帧格式

协议帧总长度固定为 512 Byte：

| 字段                | Offset | 大小       | 说明                          |
| ----------------- | ------:| --------:| --------------------------- |
| SOF               | 0      | 2 Byte   | 帧起始标志，当前为 `0xA55A`          |
| Version           | 2      | 1 Byte   | 协议版本，当前为 `0x01`             |
| Transaction ID    | 3      | 2 Byte   | 请求响应匹配和重传去重                 |
| Error Code        | 5      | 1 Byte   | Request 为 0，Response 表示执行结果 |
| Payload Length    | 6      | 2 Byte   | 有效 Payload 长度               |
| Payload / Padding | 8      | 500 Byte | 业务数据，不足部分补 0                |
| CRC32             | 508    | 4 Byte   | Byte 0～507 的 CRC32          |

```text
Byte 0                                                    Byte 511
┌─────┬─────────┬────────────┬───────┬────────┬────────────┬───────┐
│ SOF │ Version │ Trans. ID  │ Error │ Length │ Payload/0  │ CRC32 │
│ 2 B │   1 B   │    2 B     │  1 B  │  2 B   │   500 B    │  4 B  │
└─────┴─────────┴────────────┴───────┴────────┴────────────┴───────┘
```

协议帧中的多字节整数统一使用小端序。CRC 使用 CRC-32/ISO-HDLC 反射算法，计算区域固定为前 508 Byte，因此 CRC 位置和计算长度不依赖 Payload Length。

协议核心只根据 Payload Length 搬运 Payload，不解释其中的业务字段。

## 代码结构

```text
.
├── include/
│   ├── messages.h                    # 示例业务消息格式
│   └── protocol/
│       ├── config.h                  # 协议、任务和Transport配置
│       ├── master/master.h           # Master公共接口
│       ├── slave/slave.h             # Slave公共接口
│       └── utils/
│           ├── byte_codec.h          # 大小端整数编解码
│           ├── crc32.h               # 帧CRC32
│           ├── frame.h               # 固定帧构建与校验
│           ├── pingpong_buffer.h     # 帧双缓冲
│           ├── sha256.h              # 增量SHA-256与文本格式化
│           ├── transport.h           # Transport统一接口
│           └── types.h               # Frame和Transport状态
├── src/
│   ├── main.cpp                      # ESP32 Slave示例
│   └── protocol/
│       ├── master/
│       │   ├── master.cpp            # Master通用事务逻辑
│       │   ├── master_freertos.cpp   # FreeRTOS队列与任务实现
│       │   └── master_linux.cpp      # Linux线程与队列实现
│       ├── slave/
│       │   ├── slave.cpp             # Slave通用请求处理
│       │   ├── slave_freertos.cpp    # FreeRTOS任务实现
│       │   └── slave_linux.cpp       # Linux线程实现
│       └── utils/
│           ├── crc32.cpp
│           ├── frame.cpp
│           ├── sha256.cpp
│           └── transport.cpp
├── linux/
│   ├── CMakeLists.txt
│   ├── main.cpp                      # Linux Master示例
│   └── test_data.txt                 # 文件上传测试数据
└── platformio.ini
```

### Frame

`Frame` 负责：

- 写入 SOF、Version、Transaction ID、Error Code 和 Payload Length。
- 清零 Padding。
- 计算并写入 CRC32。
- 校验收到的固定帧。
- 提供有效 Payload 和帧头字段。

Frame 内部使用 `PingPongBuffer<PROTOCOL_FRAME_SIZE>`。一个缓冲区保存当前有效帧，另一个缓冲区用于接收或构建下一帧；校验成功后再交换读写缓冲区，避免无效数据覆盖当前帧。

### Transport

协议层只依赖三个接口：

```cpp
TransportStatus transportInit();
TransportStatus transportSend(const uint8_t* data, size_t length);
TransportStatus transportReceive(uint8_t* data, size_t length);
```

当前 Arduino 分支使用 `Serial1`，Linux 分支使用 POSIX `termios`、`poll`、`read` 和 `write`。新增平台时只需要实现这三个函数，不需要修改 Frame、Master 或 Slave 的事务逻辑。

### Master API

初始化：

```cpp
Master master;

if (master.begin() != MasterStatus::Ok) {
    // Transport或协议任务启动失败
}
```

阻塞事务：

```cpp
uint8_t requestPayload[] = {0x01};
uint8_t responsePayload[32];

MasterRequest request;
request.priority = RequestPriority::Normal;
request.transmitData = requestPayload;
request.transmitLength = sizeof(requestPayload);
request.receiveBuffer = responsePayload;
request.receiveCapacity = sizeof(responsePayload);

MasterResult result = master.transact(request);
```

`transact()` 将请求入队并阻塞调用线程，事务完成后返回 `MasterResult`。

异步事务：

```cpp
void onComplete(const MasterResult& result, void* context)
{
    // context由业务层定义
}

MasterRequest request;
request.priority = RequestPriority::Low;
request.transmitData = requestPayload;
request.transmitLength = sizeof(requestPayload);
request.receiveBuffer = responsePayload;
request.receiveCapacity = sizeof(responsePayload);
request.completionCallback = onComplete;
request.callbackContext = nullptr;

MasterStatus status = master.submit(request);
```

`submit()` 只负责提交请求，完成回调在 Protocol Master 的线程或任务中执行，因此回调不应进行长时间阻塞操作。

Master 使用固定大小的 `RequestSlot` 对象池。Free Queue 保存尚未使用的 Slot，三个优先级队列保存等待执行的 Slot 指针，从而避免频繁动态分配内存。

### Slave API

业务层实现一个回调：

```cpp
void handleRequest(
    const SlaveRequest& request,
    SlaveResponse& response,
    void* context)
{
    // 解析request.payload
    // 填写response.payload、payloadLength和errorCode
}
```

然后启动 Slave：

```cpp
Slave slave;
SlaveStatus status = slave.begin(handleRequest);
```

进入回调前，协议层已经完成固定帧长度、SOF、Version、Payload Length 和 CRC32 检查。业务层仍需检查自己的 Payload 格式、字段范围和状态条件。

### Utils

- `byte_codec`：提供 `readUint16Le()`、`readUint32Le()`、`readUint32Be()` 等显式字节序函数。
- `crc32`：计算固定帧的 CRC32。
- `sha256`：提供可分块调用的 `Sha256::update()`、`finish()` 和 `formatSha256()`。
- `pingpong_buffer`：提供两个编译期固定大小的静态缓冲区。
- `types`：定义 Frame 和 Transport 的状态类型。

## Linux Master 与 ESP32 Slave 示例

本节的 `messages.h` 是测试程序采用的一种业务消息设计，用于演示如何在 Payload 中组织控制、传感器和文件分块数据。实际项目可以根据业务需要替换这些消息，不需要修改固定帧和事务层。

### 示例任务

Linux Master 同时运行三个业务来源：

| 业务       | 周期/方式    | 优先级    |
| -------- | -------- | ------ |
| 控制任务     | 每 100 ms | High   |
| 传感器查询    | 每 250 ms | Normal |
| TXT 文件上传 | 逐块发送     | Low    |

文件块之间等待 30 ms。Protocol Master 仍然根据队列状态调度，因此控制或传感器请求可以插入两个文件块之间。

### 示例 Payload

所有示例消息的第一个字节是 `type`，多字节整数使用小端序。这是示例业务层的约定。

Request：

```text
Control:
[type:1][sequence:4]

Sensor:
[type:1]

FileIndex:
[type:1][fileIndex:4][fileSize:4][sha256:32]
[nameLength:1][fileName:N]

FileBlock:
[type:1][fileIndex:4][offset:4][dataLength:2][data:N]

FileComplete:
[type:1][fileIndex:4]
```

Response：

```text
Control:
[type:1][sequence:4]

Sensor:
[type:1][temperatureX10:2][humidity:2]

FileIndex:
[type:1][fileIndex:4]

FileBlock:
[type:1][fileIndex:4][nextOffset:4]

FileComplete:
[type:1][fileIndex:4]
```

FileBlock 固定头为 11 Byte，因此单帧最多携带：

```text
500 - 11 = 489 Byte 文件数据
```

### 测试结果

测试使用 Linux 作为 Master、ESP32-C3 作为 Slave，通过 921600 波特率 UART 连接。测试过程中同时运行：

- 每 100 ms 提交一次 High 优先级控制请求。
- 每 250 ms 提交一次 Normal 优先级传感器查询。
- 以 Low 优先级上传 `test_data.txt`。

测试文件大小为 3432 Byte。文件数据按每块最多 489 Byte 拆分，共完成 8 个文件块事务：前 7 块累计到 3423 Byte，最后一块传输剩余 9 Byte。

文件传输期间，控制请求和传感器查询成功插入相邻文件块之间，文件块的 `nextOffset` 响应与 Linux 发送进度一致。文件上传完成后，控制和传感器任务继续运行，说明多个业务线程能够通过同一个 Protocol Master 安全访问总线。

Linux 发送的整文件 SHA-256 与 ESP32 根据实际接收数据计算的 SHA-256 完全一致：

```text
d7ddf404b71069d1caa05558119880e66b222bcc1a175352abc238e511a99dd6
```

下面是一次实际测试的输出节选。ESP32 接收了文件索引和全部 8 个文件块，随后输出文件内容并比较 SHA-256：

```text
[=LOG=]: File index 1 accepted: test_data.txt, 3432 bytes
[=LOG=]: Expected SHA-256: d7ddf404b71069d1caa05558119880e66b222bcc1a175352abc238e511a99dd6
[=LOG=]: File block 1, received 489 bytes
[=LOG=]: File block 2, received 978 bytes
[=LOG=]: File block 3, received 1467 bytes
[=LOG=]: File block 4, received 1956 bytes
[=LOG=]: File block 5, received 2445 bytes
[=LOG=]: File block 6, received 2934 bytes
[=LOG=]: File block 7, received 3423 bytes
[=LOG=]: File block 8, received 3432 bytes
[=LOG=]: File complete: test_data.txt, 3432 bytes

----- Received TXT file: test_data.txt -----
Protocol transfer test document.
This file is intentionally larger than a single 500-byte protocol payload.
The Linux master divides it into low-priority blocks while control and sensor tasks continue to run.
001 The quick brown fox jumps over the lazy dog. Fixed frames carry only valid payload bytes.
002 High-priority control traffic may run between two low-priority file blocks.
003 Sensor requests use normal priority and return a short text response.
...
038 Longer business operations require a larger timeout or a deferred operation design.
039 The current example keeps all callback processing short and deterministic.
040 End of the protocol scheduling and large-file transfer test document.
----- End of TXT file -----
[=LOG=]: Expected SHA-256: d7ddf404b71069d1caa05558119880e66b222bcc1a175352abc238e511a99dd6
[=LOG=]: Received SHA-256: d7ddf404b71069d1caa05558119880e66b222bcc1a175352abc238e511a99dd6
```

文件上传完成后，Linux 侧的控制任务和传感器任务仍持续提交事务。下面的连续日志展示了两类请求交错运行，并共用递增的 Transaction ID：

```text
[Control] transaction=607 sequence=418 acknowledged
[Sensor] transaction=608 temperature=26.0 humidity=45
[Control] transaction=609 sequence=419 acknowledged
[Control] transaction=610 sequence=420 acknowledged
[Control] transaction=611 sequence=421 acknowledged
[Sensor] transaction=612 temperature=24.1 humidity=46
[Control] transaction=613 sequence=422 acknowledged
[Control] transaction=614 sequence=423 acknowledged
[Sensor] transaction=615 temperature=25.2 humidity=47
[Control] transaction=616 sequence=424 acknowledged
[Control] transaction=617 sequence=425 acknowledged
[Sensor] transaction=618 temperature=26.3 humidity=48
```

本次测试验证了固定帧收发、多优先级事务调度、多线程安全访问、文件分块传输、Offset 确认以及整文件 SHA-256 完整性校验。

[源码地址](https://github.com/fazhehy/protocol)
