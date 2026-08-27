---
title: ESP32-C3 OTA 学习与实现
published: 2026-08-28
description: ''
image: ''
tags: [embedded, stm32, esp32]
category: 'embedded'
draft: false 
lang: ''
---

# ESP32-C3 OTA 学习与实现

本文用于记录 ESP32-C3 的程序启动过程、Arduino OTA API 以及本工程的 OTA 示例实现。

## 第一部分：ESP32-C3 程序启动流程

### 1. 启动整体时序

ESP32-C3 从上电到执行 Arduino `loop()`，需要依次经过片内 ROM、二级 Bootloader、ESP-IDF、FreeRTOS 和 Arduino Core。

完整时序如下：

```text
设备上电或复位
    ↓
CPU 进入芯片内部 ROM
    ↓
ROM 判断启动模式
    ├── 下载模式：等待 esptool 通过 USB/UART 烧录
    └── Flash 启动模式
            ↓
       ROM 加载二级 Bootloader
            ↓
       二级 Bootloader 初始化 Flash、Cache 和 MMU
            ↓
       读取分区表
            ↓
       读取 OTA 启动信息
            ↓
       选择 ota_0 或 ota_1
            ↓
       验证、加载和映射应用镜像
            ↓
       跳转到应用入口
            ↓
       ESP-IDF 初始化运行环境
            ↓
       启动 FreeRTOS
            ↓
       调用 Arduino app_main()
            ↓
       initArduino()
            ↓
       创建 loopTask
            ↓
       setup() 执行一次
            ↓
       loop() 循环执行
```

结合当前工程的主要 Flash 地址，可以简化为：

```text
芯片内部 ROM
    ↓
Flash 0x000000：二级 Bootloader
    ↓
Flash 0x008000：分区表
    ↓
Flash 0x00E000：OTA 启动信息
    ↓
选择应用分区
    ├── ota_0：Flash 0x010000
    └── ota_1：Flash 0x150000
            ↓
加载到内部 RAM，或映射到 CPU 虚拟地址
            ↓
ESP-IDF → FreeRTOS → Arduino
            ↓
setup() → loop()
```

> ESP32-C3 复位后不会直接运行 `firmware.bin`，也不会直接调用 `setup()`。它必须依次经过片内 ROM、二级 Bootloader 和应用启动环境。

### 2. 启动各阶段详解

#### 2.1 CPU 复位与片内 ROM

以下情况都会使 ESP32-C3 重新进入启动流程：

- 设备上电；
- 按下复位键；
- 调用 `ESP.restart()`；
- 看门狗复位；
- 异常复位；
- 从 Deep Sleep 唤醒。

CPU 复位后不会直接从外部 Flash 读取 Arduino 应用，而是首先进入芯片内部固化的 ROM 启动程序。ROM 由芯片制造商写入，普通烧录操作不能修改它。

ROM 启动程序负责完成最低限度的准备工作，包括：

- 建立早期运行环境和临时栈；
- 读取启动引脚和 eFuse 配置；
- 判断进入下载模式还是正常 Flash 启动模式；
- 初始化读取外部 SPI Flash 所需的基本硬件；
- 读取并加载二级 Bootloader。

> 启动引脚也称为 Strapping Pins。ROM 会在复位瞬间采样这些引脚的电平，用来决定芯片进入正常 Flash 启动模式还是 USB/UART 下载模式。例如按住开发板的 BOOT 键再复位，本质上就是改变启动引脚在复位时的电平。进入启动流程后再改变这些引脚的电平，不会改变本次已经确定的启动模式。
>
> eFuse 是芯片内部的一次性可编程配置区域。ROM 会读取其中与启动有关的配置，例如安全启动、Flash 加密以及调试或下载接口权限。启动引脚提供本次复位时的外部启动条件，eFuse 则提供芯片长期保存的内部启动和安全策略；ROM 综合两者决定后续行为。eFuse 写入后通常不能恢复，修改相关配置前必须确认具体芯片手册和安全影响。

启动模式可以简化为：

```text
片内 ROM
    ├── USB/UART 下载模式
    └── SPI Flash 正常启动模式
```

STM32 Cortex-M 通常在复位后从向量表取得初始 MSP 和 `Reset_Handler` 地址。ESP32-C3 是 RISC-V 架构，芯片级复位入口位于内部 ROM，后续镜像入口则记录在 ESP 镜像头中。

> ESP32-C3 也有类似 `Reset_Handler` 的启动过程，但它不是 STM32 式的 Flash 向量表结构。芯片级复位首先进入片内 ROM。

#### 2.2 ROM 加载二级 Bootloader

正常 Flash 启动时，ROM 从 Flash 物理偏移 `0x000000` 读取二级 Bootloader 镜像。

```text
ROM
    ↓
读取 Flash 0x000000
    ↓
解析 ESP 镜像头
    ↓
读取各程序段的加载地址和长度
    ↓
把 Bootloader 的 IRAM/DRAM 段复制到内部 RAM
    ↓
读取 Entry address
    ↓
跳转到 Bootloader 入口
```

ROM 并不是把 CPU 的 PC 直接设置为 `0x000000`。`0x000000` 是镜像在外部 Flash 中的存储位置，不是 Bootloader 最终执行时的 CPU 地址。

> CPU 的 PC 保存的是 CPU 地址空间中的指令地址，而 `0x000000` 在这里表示外部 SPI Flash 芯片内的物理偏移，不能直接当作可执行函数地址使用。启动初期，ROM 会读取 Bootloader 镜像的段描述，把需要执行的代码复制到内部 IRAM，例如 `0x403CC710`，再把 PC 设置为镜像头中的 Entry address。
>
> 对于后续应用中的大部分代码和只读数据，ESP32-C3 不会全部复制到内部 RAM，而是通过 MMU 将外部 Flash 的物理页面映射到 CPU 虚拟地址空间。程序代码通常映射到 IROM 的 `0x420xxxxx` 地址范围，只读数据通常映射到 DROM 的 `0x3C0xxxxx` 地址范围。CPU 通过这些虚拟地址取指或读取数据，Cache 和 MMU 再把访问转换到实际的 Flash 物理位置。因此，Flash 物理偏移、CPU 虚拟地址和镜像 Entry address 必须分开理解。

当前构建的 `bootloader.bin` 信息为：

```text
Flash 存储位置：0x000000
镜像入口地址：  0x403CC710
入口符号：      call_start_cpu0
```

当前 Bootloader 镜像的主要段包括：

```text
DRAM 段加载到 0x3FCD5810
IRAM 段加载到 0x403CC710
IRAM 段加载到 0x403CE710
```

ROM 将这些段加载到内部 RAM 后，再跳转到 `0x403CC710` 执行 `call_start_cpu0`。

> `0x000000` 表示 Bootloader 存在哪里，`0x403CC710` 表示当前构建中 CPU 从哪里开始执行 Bootloader。两者属于不同地址空间。

#### 2.3 二级 Bootloader 初始化

二级 Bootloader 获得控制权后，会继续建立完整的启动环境，主要包括：

- 设置 Bootloader 自己的栈；
- 完成早期 C 运行环境初始化；
- 初始化 SPI Flash；
- 配置 Flash 模式、频率等参数；
- 初始化 Cache 和 MMU；
- 根据配置检查 Flash 加密和 Secure Boot；
- 准备读取分区表并选择应用。

ROM 和二级 Bootloader 使用的栈不是应用任务栈：

```text
ROM 临时栈
    ↓ ROM 跳转到二级 Bootloader
Bootloader 栈
    ↓ Bootloader 跳转到应用
应用启动栈和 FreeRTOS 任务栈
```

> ROM 栈、Bootloader 栈和 Arduino `loopTask` 栈互相独立，不存在一个栈从上电一直沿用到 `loop()`。

#### 2.4 读取分区表

二级 Bootloader 从 Flash 物理偏移 `0x008000` 读取二进制分区表。分区表告诉 Bootloader：

- NVS 位于哪里；
- OTA 启动信息位于哪里；
- 应用分区位于哪里；
- 每个分区有多大；
- 文件系统和 Core Dump 分区位于哪里。

当前默认分区表中的主要地址是：

```text
nvs      → 0x009000，大小 0x05000
otadata  → 0x00E000，大小 0x02000
ota_0    → 0x010000，大小 0x140000
ota_1    → 0x150000，大小 0x140000
spiffs   → 0x290000，大小 0x160000
coredump → 0x3F0000，大小 0x10000
```

Bootloader 不需要把 `ota_0` 和 `ota_1` 的地址固定写在程序逻辑中，而是通过分区表取得这些地址。

应用分区通常需要按照 `0x10000`，也就是 64 KiB 边界对齐。这是后续 Flash 映射能够保持页内偏移一致的重要条件。

> 应用分区位置由分区表决定，不是由 `firmware.bin` 自己决定。修改分区表后，应用地址和最大固件大小都可能改变。

#### 2.5 读取 OTA 信息并选择应用

找到 `otadata` 分区后，Bootloader 从当前工程的 Flash `0x00E000` 读取 OTA 启动信息。

`otadata` 不保存应用程序，它只记录 OTA 选择状态，例如下一次应该从哪个应用分区启动。

```text
Bootloader
    ↓
读取 otadata
    ↓
选择应用
    ├── ota_0：Flash 0x010000
    └── ota_1：Flash 0x150000
```

第一次完整烧录后，通常从 `ota_0` 启动。第一次 OTA 成功后，新固件位于 `ota_1`，Bootloader 会在下一次复位时选择 `ota_1`。再下一次 OTA 则可以写回 `ota_0`。

如果所选应用镜像无效，Bootloader 会拒绝正常启动它。是否可以自动回退到上一版本，还取决于 OTA 回滚功能及相关配置。

> OTA 切换不是旧程序直接跳转到新程序，而是更新启动选择后复位，再由 Bootloader 重新选择应用。

#### 2.6 验证应用镜像

确定目标应用分区后，Bootloader 从分区起始地址读取 ESP 应用镜像头：

```text
选择 ota_0 → 从 Flash 0x010000 读取
选择 ota_1 → 从 Flash 0x150000 读取
```

验证内容主要包括：

- ESP 镜像 Magic 是否正确；
- 镜像目标芯片是否为 ESP32-C3；
- 镜像段数量和长度是否合法；
- Entry address 是否有效；
- Checksum 或附加 Hash 是否正确；
- 启用 Secure Boot 时，数字签名是否有效。

ESP 应用镜像不是简单的裸机器码。它包含镜像头、多个程序段、加载地址、入口地址以及校验信息。

> 应用分区中必须放置合法的 ESP 应用镜像，不能把任意二进制文件写进去并期待 Bootloader 启动它。

#### 2.7 加载与映射应用程序

应用镜像中的程序段分为两种主要处理方式。

##### 2.7.1 IRAM 和 DRAM 段

需要在内部 RAM 中运行或保存的内容，会从应用分区复制到镜像段头指定的地址：

```text
应用分区中的程序段
        ↓ Bootloader复制
内部 IRAM 或 DRAM
```

这类内容通常包括：

- 早期启动代码；
- 必须从 IRAM 执行的函数；
- 已初始化的可写全局数据；
- 部分系统运行数据。

##### 2.7.2 IROM 和 DROM 段

大部分普通代码和只读常量不需要全部复制到内部 RAM，而是通过 Cache 和 MMU 映射：

```text
外部 Flash 物理地址
        ↓ MMU和Cache
CPU 虚拟地址
```

常见地址范围可以概括为：

```text
0x420xxxxx：映射后的程序代码 IROM
0x3C0xxxxx：映射后的只读数据 DROM
0x403xxxxx：内部 IRAM
0x3FCxxxxx：内部 DRAM
```

##### 2.7.3 同一个固件为什么能运行在两个分区

假设应用代码在 `firmware.bin` 中的相对数据偏移为 `0x10020`，CPU 期望的虚拟地址为 `0x42000020`。

从 `ota_0` 启动时：

```text
应用分区起始地址：0x010000
镜像内部相对偏移：0x010020
Flash物理地址：   0x020020

Flash 0x020020 ──MMU──→ CPU 0x42000020
```

从 `ota_1` 启动时：

```text
应用分区起始地址：0x150000
镜像内部相对偏移：0x010020
Flash物理地址：   0x160020

Flash 0x160020 ──MMU──→ CPU 0x42000020
```

因此：

```text
ota_0中的代码 ─┐
                ├──MMU──→ 相同CPU虚拟地址
ota_1中的代码 ─┘
```

同一个 `firmware.bin` 不需要分别为 `ota_0` 和 `ota_1` 重新链接。

> 应用中的函数和只读数据主要使用 CPU 虚拟地址，不直接使用 `ota_0` 或 `ota_1` 的 Flash 物理地址。这是同一应用镜像可以在两个 OTA 分区运行的关键。

#### 2.8 跳转到应用入口

应用段加载和映射完成后，Bootloader 从应用镜像头取得 Entry address，并把控制权交给应用的早期启动代码。

当前构建的应用信息为：

```text
应用存储位置：0x010000或0x150000
应用入口示例：0x40381FC2
入口符号：    call_start_cpu0
```

这里同样需要区分：

- `0x010000`或`0x150000`是整个应用镜像的 Flash 物理起始地址；
- `0x40381FC2`是当前应用镜像的 CPU 入口地址。

Bootloader 跳转到应用入口，而不是直接寻找 `setup()`或`loop()`。

> Entry address 是构建结果，代码变化或工具链变化后可能改变，不应把示例入口地址作为固定常量写入 OTA 程序。

#### 2.9 ESP-IDF 与 FreeRTOS 启动

应用的 `call_start_cpu0` 开始运行后，ESP-IDF 启动代码继续建立应用环境：

```text
应用 call_start_cpu0
    ↓
完成应用早期运行环境初始化
    ↓
初始化静态数据、BSS和堆
    ↓
运行C/C++全局构造函数
    ↓
初始化ESP-IDF系统组件
    ↓
启动FreeRTOS调度环境
    ↓
创建main task
    ↓
调用app_main()
```

当前框架配置中的 ESP-IDF main task 栈大小为：

```text
CONFIG_ESP_MAIN_TASK_STACK_SIZE = 4096字节
```

`app_main()` 是 ESP-IDF 应用的用户层入口。Arduino Framework 会提供自己的 `app_main()`，然后由它继续创建 Arduino 运行任务。

> ESP-IDF 应用的入口形式是 `app_main()`，不是传统桌面C/C++程序中的 `main()`。

### 3. 从 ESP-IDF 进入 Arduino 程序

#### 3.1 Arduino `app_main()`

Arduino Core 提供的 `app_main()` 可以简化理解为：

```cpp
extern "C" void app_main()
{
    initArduino();

    xTaskCreateUniversal(
        loopTask,
        "loopTask",
        getArduinoLoopTaskStackSize(),
        nullptr,
        1,
        &loopTaskHandle,
        ARDUINO_RUNNING_CORE);
}
```

它先调用 `initArduino()` 初始化 Arduino 运行环境，然后创建一个 FreeRTOS 任务执行用户程序。

#### 3.2 `setup()` 和 `loop()`

Arduino 的 `loopTask` 可以简化为：

```cpp
void loopTask(void*)
{
    setup();

    for (;;) {
        loop();
    }
}
```

因此：

- `setup()`在`loopTask`中执行一次；
- `loop()`在同一个任务中不断执行；
- Bootloader并不知道`setup()`和`loop()`的存在；
- Bootloader只负责启动完整的ESP-IDF应用镜像。

#### 3.3 Arduino 任务栈

当前 Arduino Core 配置的默认 `loopTask` 栈大小为：

```text
CONFIG_ARDUINO_LOOP_STACK_SIZE = 8192字节
```

整个启动过程中使用的栈可以概括为：

```text
ROM临时栈
    ↓
Bootloader栈
    ↓
ESP-IDF main task栈
    ↓
Arduino loopTask栈
```

每个用户创建的 FreeRTOS 任务还会拥有自己的独立任务栈。

> `setup()`和`loop()`运行在 Arduino 创建的 `loopTask` 中。任务栈大小与应用位于`ota_0`还是`ota_1`没有关系。

### 4. OTA 后的再次启动

假设设备当前从`ota_0`运行：

```text
当前应用：ota_0，Flash 0x010000
备用分区：ota_1，Flash 0x150000
```

一次完整OTA过程对启动状态的影响是：

```text
当前从ota_0运行
    ↓
将新firmware写入ota_1
    ↓
验证新应用镜像
    ↓
更新otadata中的启动选择
    ↓
ESP.restart()
    ↓
CPU重新进入片内ROM
    ↓
ROM重新加载二级Bootloader
    ↓
Bootloader读取分区表和otadata
    ↓
选择ota_1
    ↓
加载并启动新应用
    ↓
重新执行setup()和loop()
```

普通启动和OTA后的启动对比如下：

```text
普通首次启动：ROM → Bootloader → ota_0 → 应用
第一次OTA后： ROM → Bootloader → ota_1 → 应用
第二次OTA后： ROM → Bootloader → ota_0 → 应用
```

虽然应用所在Flash物理分区发生变化，但MMU会把相应程序段映射到应用所需的CPU虚拟地址，因此同一个`firmware.bin`可以在两个OTA分区中运行。

> OTA完成后不是从旧应用直接跳转到新应用，而是经过一次完整的复位和启动流程。真正选择新应用的是二级Bootloader。

### 5. 启动相关文件说明

#### 5.1 `bootloader.bin`

`bootloader.bin`是二级Bootloader镜像，负责：

- 初始化Flash、Cache和MMU；
- 读取分区表和OTA启动信息；
- 选择、验证并启动应用镜像。

当前ESP32-C3工程将它写入：

```text
Flash 0x000000
```

#### 5.2 `partitions.bin`

`partitions.bin`是CSV分区表编译后的二进制形式，记录每个分区的类型、子类型、起始地址和大小。

当前写入地址：

```text
Flash 0x008000
```

#### 5.3 `boot_app0.bin`

`boot_app0.bin`用于初始化OTA启动数据区域，使首次启动能够选择初始应用分区。

当前写入地址：

```text
Flash 0x00E000
```

#### 5.4 `firmware.bin`

`firmware.bin`是完整的ESP应用镜像，包含：

- 用户的`setup()`、`loop()`和其他业务代码；
- Arduino Core；
- 被链接进应用的ESP-IDF组件；
- 被使用的库；
- 应用镜像头和程序段；
- 校验信息。

但它不包含：

- 二级Bootloader；
- 分区表；
- OTA数据分区；
- SPIFFS文件系统镜像。

第一次完整烧录时，它通常写入：

```text
ota_0：Flash 0x010000
```

普通OTA升级时，同一个`firmware.bin`会被写入当前未运行的OTA分区。

#### 5.5 `firmware.elf`

`firmware.elf`包含更完整的链接和调试信息，例如：

- 函数和变量符号；
- CPU地址；
- 程序段；
- 调试信息。

它主要用于调试、反汇编和地址解析，普通OTA不会传输`firmware.elf`。

#### 5.6 第一次完整烧录

PlatformIO第一次烧录时，会把多个独立镜像分别写到对应地址：

```text
bootloader.bin → Flash 0x000000
partitions.bin → Flash 0x008000
boot_app0.bin  → Flash 0x00E000
firmware.bin   → Flash 0x010000
```

这些文件默认是独立镜像，不是先合并成一个文件再烧录。

#### 5.7 普通OTA使用的文件

应用OTA通常只传输：

```text
firmware.bin
```

设备中已有的Bootloader会根据分区表，把它写入当前未运行的OTA应用分区。

修改分区表或Bootloader后，不能假设普通应用OTA会自动更新它们。此时通常需要通过USB/UART重新执行完整烧录。

> 第一次完整烧录负责建立Bootloader、分区表、OTA状态和初始应用；后续普通OTA只更新应用`firmware.bin`。

#### 5.8 查看镜像信息

可以使用`esptool.py`查看镜像中的程序段和入口地址：

```bash
python ~/.platformio/packages/tool-esptoolpy/esptool.py \
    --chip esp32c3 \
    image_info .pio/build/adafruit_qtpy_esp32c3/bootloader.bin

python ~/.platformio/packages/tool-esptoolpy/esptool.py \
    --chip esp32c3 \
    image_info .pio/build/adafruit_qtpy_esp32c3/firmware.bin
```

镜像的Entry address会随着代码、构建配置和工具链变化。文中使用的`0x403CC710`和`0x40381FC2`只表示当前构建结果。

## 第二部分：Arduino `Update` API 详解

Arduino-ESP32 的 `Update` 类对 ESP-IDF 的 OTA 分区操作进行了封装。应用程序不需要自己擦除 Flash、计算
Flash 物理地址或直接修改 `otadata`，只需要按照规定顺序调用 `begin()`、`write()` 和 `end()`。

### 1. `Update` 整体调用时序

一次应用固件更新的基本流程如下：

```text
获得 firmware.bin 及其大小
    ↓
Update.begin(size, U_FLASH)
    ↓
选择当前未运行的 OTA 应用分区
    ↓
循环接收固件数据
    ↓
Update.write(data, length)
    ↓
检查 write() 的实际写入长度
    ↓
所有数据接收完成
    ↓
Update.end()
    ↓
检查镜像并设置下次启动分区
    ↓
向发送端返回升级成功响应
    ↓
ESP.restart()
    ↓
Bootloader 选择并启动新应用
```

任意阶段失败时，应停止本次更新：

```text
begin()、write() 或 end() 失败
    ↓
读取 getError() 或 errorString()
    ↓
Update.abort()
    ↓
放弃未完成的新分区
    ↓
当前应用继续运行
```

> `Update` 只负责把已经取得的字节写入目标分区并提交启动选择。它不负责 Wi-Fi、HTTP、UART 或自定义协议传输，也不会在成功后自动调用 `ESP.restart()`。
>
> `Update.end()` 成功只表示新分区已经成为下次启动目标。CPU 此时仍在旧应用中执行，必须经过复位并重新运行 ROM 和 Bootloader，才会进入新应用。

### 2. `Update` 对象与使用条件

使用 Arduino OTA API 时需要包含头文件：

```cpp
#include <Update.h>
```

Arduino Core 默认提供一个全局对象：

```cpp
UpdateClass Update;
```

通常直接使用 `Update`，不需要自己创建 `UpdateClass` 实例。

应用 OTA 的基本条件包括：

- 分区表中至少存在两个 OTA 应用分区；
- 存在 `otadata` 分区，用于保存启动选择；
- 新的 `firmware.bin` 不大于目标 OTA 分区；
- 设备当前已经运行包含 OTA 接收逻辑的应用；
- 传输的是 ESP 应用镜像，而不是 Bootloader、分区表或整个 Flash 的合并镜像。

本工程当前主要分区关系为：

```text
otadata → 记录下次启动选择
ota_0   → 一个应用分区
ota_1   → 另一个应用分区
```

当设备从 `ota_0` 运行时，`Update.begin(..., U_FLASH)` 通常选择 `ota_1`；当设备从 `ota_1` 运行时，通常选择
`ota_0`。

`Update` 支持的常用更新类型为：

```cpp
U_FLASH   // 更新应用固件
U_SPIFFS  // 更新 SPIFFS 或兼容的数据分区
```

> 本工程执行的是应用 OTA，因此使用 `U_FLASH`。`firmware.bin` 不能使用 `U_SPIFFS` 写入，文件系统镜像也不能使用 `U_FLASH` 当作应用启动。

### 3. API 分阶段详解

#### 3.1 `Update.begin()`：开始更新

当前 Arduino-ESP32 中的函数声明为：

```cpp
bool begin(
    size_t size = UPDATE_SIZE_UNKNOWN,
    int command = U_FLASH,
    int ledPin = -1,
    uint8_t ledOn = LOW,
    const char* label = nullptr);
```

应用固件更新通常写成：

```cpp
if (!Update.begin(firmwareSize, U_FLASH)) {
    Serial.println(Update.errorString());
    return;
}
```

各参数含义如下：

| 参数 | 含义 |
| --- | --- |
| `size` | 本次准备写入的总字节数 |
| `command` | 更新应用 `U_FLASH` 或数据分区 `U_SPIFFS` |
| `ledPin` | 可选的写入指示灯引脚，`-1` 表示不使用 |
| `ledOn` | 指示灯点亮时对应的电平 |
| `label` | 更新数据分区时可指定分区标签 |

使用 `U_FLASH` 时，`begin()` 会通过底层 OTA API 查找当前未运行的下一应用分区。它不会覆盖当前正在执行的应用分区。

`begin()` 主要完成：

- 检查是否已有更新正在进行；
- 检查固件大小是否有效；
- 查找目标 OTA 分区；
- 检查固件是否能放入目标分区；
- 分配一个 Flash 扇区大小的内部缓冲区；
- 初始化本次更新的进度和 MD5 状态。

如果调用时还不知道最终大小，可以传入：

```cpp
Update.begin(UPDATE_SIZE_UNKNOWN, U_FLASH);
```

在当前实现中，这会暂时把目标分区大小作为最大更新大小，最后通常需要配合 `end(true)` 按实际写入量结束。

> 已知固件准确大小时，应优先传入真实大小并使用默认的 `end(false)`。这样 `Update` 能在开始阶段检查空间，并在结束阶段拒绝长度不完整的固件。

#### 3.2 `Update.write()`：写入固件数据

函数声明为：

```cpp
size_t write(uint8_t* data, size_t len);
```

基本用法如下：

```cpp
const size_t written = Update.write(data, dataLength);
if (written != dataLength) {
    Serial.println(Update.errorString());
    Update.abort();
    return;
}
```

返回值表示本次实际接收并处理的字节数。不能只调用 `write()` 而忽略返回值，否则 Flash 写入失败时，上层仍可能错误地继续发送下一块。

`write()` 内部会先把数据放入缓冲区，缓冲区达到 Flash 扇区大小或已经接收到最后一段数据时，再执行实际擦除和写入。因此：

- 每次调用的数据长度不需要正好等于 Flash 扇区大小；
- 可以按协议允许的较小数据块多次调用；
- 所有数据必须按照固件中的原始顺序连续写入；
- 写入长度不能超过 `remaining()`。

`Update.write()` 本身不接收文件偏移参数，它默认下一次数据紧接在上一次数据之后。自定义通信协议如果可能出现重发、乱序或重复数据，必须由协议层检查 `offset`，确认顺序正确后才能调用 `write()`。

> `write()` 的“顺序写入”与通信协议的“可靠传输”是两件事。`Update` 不知道一个数据块是否为重复帧，也不会根据网络包序号自动重新排列数据。

#### 3.3 `Update.writeStream()`：从流中写入

函数声明为：

```cpp
size_t writeStream(Stream& data);
```

它可以从 Arduino `Stream` 对象持续读取数据，例如串口流或某些网络客户端：

```cpp
const size_t written = Update.writeStream(stream);
```

`writeStream()` 会尝试读取 `remaining()` 指定的剩余数据，并在长时间读取不到数据时设置 Stream 错误。它适合“输入流本身已经连续、可靠且只包含固件”的场景。

自定义帧协议通常更适合使用 `write()`，原因是上层还需要逐帧检查：

- 消息类型；
- 文件索引；
- 数据偏移；
- 数据长度；
- CRC；
- 请求和响应是否对应。

#### 3.4 状态和进度 API

`Update` 提供以下状态查询：

```cpp
bool isRunning();
bool isFinished();
size_t size();
size_t progress();
size_t remaining();
```

它们的含义为：

| API | 含义 |
| --- | --- |
| `isRunning()` | 当前是否存在已经开始且尚未重置的更新 |
| `isFinished()` | 已处理字节数是否等于计划总大小 |
| `size()` | `begin()` 确定的计划总大小 |
| `progress()` | 当前已经写入处理的字节数 |
| `remaining()` | 计划总大小减去当前进度 |

轮询进度可以写成：

```cpp
const size_t current = Update.progress();
const size_t total = Update.size();
const unsigned percent = total == 0 ? 0 : current * 100U / total;
```

也可以注册进度回调：

```cpp
Update.onProgress([](size_t current, size_t total) {
    Serial.printf("OTA: %u/%u\n", static_cast<unsigned>(current), static_cast<unsigned>(total));
});
```

进度回调在内部缓冲区完成实际 Flash 写入后触发，不一定与每次上层调用 `write()` 一一对应。

> 进度日志不要混入承载二进制 OTA 协议的同一路串口。本工程使用默认 `Serial` 输出日志，使用 `Serial1` 传输协议数据。

#### 3.5 `Update.end()`：校验并提交更新

函数声明为：

```cpp
bool end(bool evenIfRemaining = false);
```

已知固件大小时，通常使用：

```cpp
if (!Update.end()) {
    Serial.println(Update.errorString());
    return;
}
```

默认的 `end(false)` 要求：

- 更新过程没有出现错误；
- 已写入数据量等于 `begin()` 指定的总大小；
- 如果配置了期望 MD5，实际 MD5 必须一致；
- 目标应用分区能够被识别为可启动镜像；
- 底层成功把目标分区设置为下次启动分区。

`end(true)` 表示即使还有计划中的剩余空间，也按当前实际写入量结束。它主要用于一开始不知道准确文件大小、使用了 `UPDATE_SIZE_UNKNOWN` 的情况。

> 不要为了绕过固件接收不完整的问题随意使用 `end(true)`。通信中已知文件大小时，长度不足应该视为升级失败，而不是把残缺内容强制提交。
>
> `end()` 会提交下次启动分区，但不会在当前调用栈中跳转到新固件。成功响应发送完成后，再调用 `ESP.restart()`。

#### 3.6 中止更新和读取错误

主动放弃更新可以调用：

```cpp
Update.abort();
```

常见使用场景包括：

- 通信连接中断；
- 数据块顺序错误；
- 文件索引发生变化；
- `write()` 返回长度不足；
- 整个固件的哈希不匹配；
- 收到新的更新请求，需要终止旧会话。

错误查询 API 包括：

```cpp
uint8_t getError();
bool hasError();
void clearError();
const char* errorString();
void printError(Print& out);
```

两种常见打印方式为：

```cpp
Serial.println(Update.errorString());
Update.printError(Serial);
```

当前版本中的主要错误码如下：

| 错误码 | 宏 | 含义 |
| ---: | --- | --- |
| 0 | `UPDATE_ERROR_OK` | 没有错误 |
| 1 | `UPDATE_ERROR_WRITE` | Flash 写入失败 |
| 2 | `UPDATE_ERROR_ERASE` | Flash 擦除失败 |
| 3 | `UPDATE_ERROR_READ` | 分区读取或最终验证失败 |
| 4 | `UPDATE_ERROR_SPACE` | 写入数据超过剩余空间 |
| 5 | `UPDATE_ERROR_SIZE` | 固件大小无效或超过分区大小 |
| 6 | `UPDATE_ERROR_STREAM` | 从 Stream 读取超时 |
| 7 | `UPDATE_ERROR_MD5` | MD5 不匹配 |
| 8 | `UPDATE_ERROR_MAGIC_BYTE` | 应用镜像首字节不是 `0xE9` |
| 9 | `UPDATE_ERROR_ACTIVATE` | 设置新启动分区失败 |
| 10 | `UPDATE_ERROR_NO_PARTITION` | 找不到可用目标分区 |
| 11 | `UPDATE_ERROR_BAD_ARGUMENT` | 更新类型等参数无效 |
| 12 | `UPDATE_ERROR_ABORT` | 更新被主动中止 |

> `abort()` 会让本次 `Update` 会话结束，但不会切换启动分区。已经写入备用分区的数据可以在下一次更新时被覆盖，不影响当前正在运行的应用。

#### 3.7 固件完整性校验

`Update` 支持为本次更新设置期望 MD5：

```cpp
bool setMD5(const char* expectedMd5);
```

使用时应在 `begin()` 成功后、`end()` 之前设置：

```cpp
if (!Update.setMD5(expectedMd5Text)) {
    Update.abort();
    return;
}
```

`expectedMd5Text` 必须是长度为 32 的十六进制字符串。`end()` 会比较整个接收内容的 MD5，不匹配时返回失败。

一个完整 OTA 系统中可以同时存在多层校验：

```text
每个协议帧的 CRC-32
    ↓ 检查单帧传输错误
整个 firmware.bin 的 SHA-256 或 MD5
    ↓ 检查文件是否完整一致
Update 对 ESP 镜像头和目标分区的检查
    ↓ 检查目标是否可以作为应用分区
Bootloader 启动时的镜像验证
    ↓ 决定是否真正执行新应用
```

这些校验解决的问题不同，不能简单互相替代。例如 CRC-32 能发现单个通信帧损坏，但不能确认所有分块是否属于同一个固件；镜像 Magic `0xE9` 正确也不能证明整个文件没有被修改。

> 本工程由通信协议检查每帧 CRC-32，并在 Linux 和 ESP32 两端计算整个 `firmware.bin` 的 SHA-256；只有 SHA-256 一致后才调用 `Update.end()`。

#### 3.8 回滚相关 API

`Update` 提供两个简单的分区切换接口：

```cpp
bool canRollBack();
bool rollBack();
```

`canRollBack()` 检查另一个 OTA 应用分区中是否存在可启动镜像。`rollBack()` 将另一个可启动应用分区设置为下次启动目标，之后仍然需要重启：

```cpp
if (Update.canRollBack() && Update.rollBack()) {
    ESP.restart();
}
```

这两个 API 不能在另一次 `Update` 正在进行时使用。

> `Update.rollBack()` 可以理解为主动切换到另一个可启动分区。它不等同于完整的“新固件首次启动后进行自检、未确认则由 Bootloader 自动回退”机制。需要自动回滚时，还要结合 ESP-IDF 的 OTA 镜像状态和回滚配置进行设计。

### 4. API 调用关系速查

| 阶段 | 主要 API | 成功条件 | 失败处理 |
| --- | --- | --- | --- |
| 设置进度回调 | `onProgress()` | 回调保存成功 | 可以不设置 |
| 开始更新 | `begin()` | 找到目标分区且空间足够 | 输出错误，不进入写入阶段 |
| 写入数据 | `write()` | 返回值等于传入长度 | `abort()` 并终止传输 |
| 查询状态 | `progress()`、`remaining()` | 数值与协议接收量一致 | 检查是否乱序或长度错误 |
| 设置 MD5 | `setMD5()` | MD5 文本格式正确 | `abort()` |
| 提交固件 | `end()` | 完整性和目标分区检查通过 | 输出错误并放弃更新 |
| 返回响应 | 自定义协议 API | 发送端收到成功确认 | 不应立即重启 |
| 启动新固件 | `ESP.restart()` | Bootloader 选择新分区 | 查看启动日志和分区状态 |
| 主动切换旧固件 | `rollBack()` | 另一个分区存在可启动镜像 | 保持当前应用运行 |

推荐的最小调用关系为：

```cpp
if (!Update.begin(firmwareSize, U_FLASH)) {
    return;
}

while (还有固件数据) {
    if (Update.write(data, dataLength) != dataLength) {
        Serial.println(Update.errorString());
        Update.abort();
        return;
    }
}

if (!Update.end()) {
    Serial.println(Update.errorString());
    return;
}

// 先把升级成功响应发送给主机，再重启。
ESP.restart();
```

这段代码只表示 `Update` API 的调用顺序。文件传输、超时、重试、分块编号、哈希和响应发送仍然需要由应用协议实现。

## 第三部分：本工程 OTA 示例讲解

本工程使用自定义 Protocol 协议完成 Linux 与 ESP32 之间的通信。Linux 作为 Master，ESP32 作为 Slave；
协议通过固定长度帧承载业务数据，并提供事务匹配、CRC-32、超时和重试功能。OTA 只是其中一种业务，Slave
后续还可以处理控制、传感器读取等其他消息。Protocol 的完整实现见：
[fazhehy/protocol](https://github.com/fazhehy/protocol)。

### 1. 工程整体架构

本工程可以分为主机、通用协议、应用消息和 OTA 四层：

```text
Linux主机
    ↓
OtaSender
    ↓
Protocol Master
    ↓ 固定帧、事务ID、CRC-32、超时重试
UART / USB-UART
    ↓
Protocol Slave
    ↓ 业务回调
AppProtocol
    ↓ 按MessageType分发
OtaReceiver
    ↓
Arduino Update
    ↓
ESP32-C3备用OTA应用分区
```

各层职责如下：

| 层次 | 主要职责 |
| --- | --- |
| Linux `OtaSender` | 读取固件、计算 SHA-256、构造 OTA 消息并显示发送进度 |
| Protocol Master/Slave | 组帧、收发、CRC-32、事务匹配、超时和重试 |
| `AppProtocol` | 根据业务消息类型把请求分发给对应模块 |
| `OtaReceiver` | 检查 OTA 会话、顺序写入固件、校验 SHA-256 并提交启动分区 |
| Arduino `Update` | 选择备用应用分区、擦写 Flash 并设置下次启动分区 |
| Bootloader | 设备重启后读取 `otadata`，验证并启动新应用 |

> `OtaReceiver` 只是 `AppProtocol` 的一个业务处理模块，并没有独占 Protocol Slave。以后增加控制、参数配置或传感器读取时，只需要扩展 `MessageType` 并在 `AppProtocol::handleRequest()` 中增加分发分支。

### 2. 完整 OTA 时序

设备第一次使用前，必须先通过 USB 完整烧录：

```text
bootloader.bin
partitions.bin
boot_app0.bin
初始firmware.bin
    ↓ USB首次烧录
ESP32具备Bootloader、OTA分区和OTA接收程序
```

之后的应用 OTA 时序如下：

```text
Linux读取firmware.bin
    ↓
检查0xE9镜像Magic、文件名和文件大小
    ↓
计算整个文件的SHA-256
    ↓
启动Protocol Master
    ↓
发送FileIndex
    ↓
ESP32检查元数据并调用Update.begin(size, U_FLASH)
    ↓
ESP32返回FileIndex响应
    ↓
Linux循环发送FileBlock
    ↓
ESP32检查fileIndex、offset和length
    ↓
Update.write()写入当前未运行的OTA分区
    ↓
ESP32返回nextOffset
    ↓
Linux确认后继续下一块
    ↓
全部数据发送完成
    ↓
Linux发送FileComplete
    ↓
ESP32检查总长度和SHA-256
    ↓
Update.end()提交新启动分区
    ↓
ESP32返回成功响应
    ↓
延迟后调用ESP.restart()
    ↓
ROM重新加载Bootloader
    ↓
Bootloader读取otadata并启动新应用
```

失败流程为：

```text
消息格式、fileIndex、offset、SHA-256或Update操作失败
    ↓
ESP32返回MessageError
    ↓
必要时调用Update.abort()
    ↓
不提交新的启动分区
    ↓
当前应用继续运行
```

> OTA 成功的关键不是“数据已经从 Linux 发出”，而是 ESP32 已经完整写入、校验并成功执行 `Update.end()`。只有这一步完成后，新的应用分区才会成为下次启动目标。

### 3. OTA 消息设计

所有业务载荷的第一个字节都是 `MessageType`。多字节整数使用小端序，由
`writeUint16Le()`、`writeUint32Le()` 和对应的读取函数进行编解码。

当前 Protocol 单帧最多承载 500 字节业务载荷。OTA 使用三种消息：

```text
FileIndex     开始一次文件传输
FileBlock     传输一块固件数据
FileComplete  请求校验并提交整个固件
```

#### 3.1 `FileIndex`：开始 OTA 会话

请求格式为：

```text
[type:1]
[fileIndex:4]
[fileSize:4]
[sha256:32]
[nameLength:1]
[fileName:N]
```

各字段含义如下：

| 字段 | 作用 |
| --- | --- |
| `fileIndex` | 标识本次固件，后续所有消息必须携带相同值 |
| `fileSize` | `firmware.bin` 的完整字节数，同时传给 `Update.begin()` |
| `sha256` | Linux 计算的整个固件 SHA-256 |
| `nameLength` | 文件名长度 |
| `fileName` | 文件名，不包含路径和字符串结尾 `\0` |

Linux 当前取 SHA-256 的前 4 字节作为 `fileIndex`；如果结果为 0，则改为 1。它不是安全认证凭据，只用于区分传输会话。

ESP32 接受请求后返回：

```text
[type:1][fileIndex:4]
```

只有 Linux 收到相同的 `fileIndex`，才会进入分块发送阶段。

#### 3.2 `FileBlock`：传输固件数据

请求格式为：

```text
[type:1]
[fileIndex:4]
[offset:4]
[dataLength:2]
[data:N]
```

固定头部占 11 字节，因此单帧最多携带：

```text
500 - 11 = 489字节固件数据
```

ESP32 只接受满足以下条件的数据块：

```text
OTA会话已经开始
fileIndex == 当前会话fileIndex
offset == nextOffset
0 < dataLength <= 489
offset + dataLength <= fileSize
```

成功写入后返回：

```text
[type:1][fileIndex:4][nextOffset:4]
```

Linux 必须确认响应中的 `nextOffset` 等于本块结尾，才能发送下一块。

> `nextOffset` 表示 ESP32 已经确认接收并交给 `Update.write()` 的位置。它使 Linux 不会仅凭“串口发送函数返回成功”就误认为 Flash 写入成功。

#### 3.3 `FileComplete`：完成并提交

请求格式为：

```text
[type:1][fileIndex:4]
```

ESP32 收到后依次检查：

1. OTA 会话是否存在；
2. `fileIndex` 是否匹配；
3. `nextOffset` 是否等于 `fileSize`；
4. 实际 SHA-256 是否等于 `FileIndex` 中的期望值；
5. `Update.end()` 是否成功。

全部成功后返回：

```text
[type:1][fileIndex:4]
```

#### 3.4 OTA 错误码

ESP32 通过 Protocol 响应帧中的 `errorCode` 返回应用错误：

| 错误 | 含义 |
| --- | --- |
| `InvalidType` | 不支持的业务消息类型 |
| `InvalidLength` | 载荷或数据块长度不合法 |
| `InvalidOffset` | 数据块偏移不等于期望位置，或写入将越界 |
| `FileMismatch` | 接收长度或整个固件 SHA-256 不匹配 |
| `FileTooLarge` | 固件无法放入目标 OTA 分区 |
| `FileNotStarted` | 尚未收到有效 `FileIndex` |
| `InvalidFileIndex` | 消息不属于当前传输会话 |
| `InvalidFileName` | 文件名为空、过长或包含非法字符 |
| `OtaBeginFailed` | `Update.begin()` 失败 |
| `OtaWriteFailed` | `Update.write()` 失败 |
| `OtaFinalizeFailed` | `Update.end()` 失败 |

Linux 会把这些错误转换为可读文本，并停止本次发送。

### 4. Linux 主机实现

#### 4.1 程序入口和固定配置

Linux 程序不读取命令行参数。串口和固件路径直接配置在 `linux/main.cpp`：

```cpp
static const char* SERIAL_DEVICE = "/dev/ttyUSB0";
static const char* FIRMWARE_PATH =
    "/path/to/ota/.pio/build/adafruit_qtpy_esp32c3/firmware.bin";
```

`main()` 只完成三件事：

```text
设置Linux协议串口
    ↓
创建OtaSender
    ↓
调用otaSender.send(FIRMWARE_PATH)
```

这样固件处理和协议传输不会继续堆积在程序入口中。

#### 4.2 固件预检查

`OtaSender::send()` 首先完成：

- 从路径中提取文件名；
- 检查文件名长度；
- 以二进制方式打开固件；
- 检查第一个字节是否为 ESP 镜像 Magic `0xE9`；
- 检查文件大小能否用 `uint32_t` 表示；
- 计算整个文件的 SHA-256；
- 生成本次传输使用的 `fileIndex`。

计算 SHA-256 会读取完整文件，因此计算结束后必须清除文件流状态并把读取位置恢复到文件开头，才能继续发送固件内容。

#### 4.3 启动 Protocol Master

Linux 通过以下代码启动协议主机：

```cpp
Master master;
const MasterStatus status = master.begin();
```

每个 OTA 请求设置为高优先级，并使用协议默认重试次数：

```cpp
request.priority = RequestPriority::High;
request.maximumRetries = MASTER_DEFAULT_RETRY_COUNT;
```

`transactAndValidate()` 统一检查：

- 协议事务是否成功；
- 重试后是否仍然失败；
- ESP32 是否返回应用错误；
- 响应消息类型是否正确；
- 响应载荷长度是否正确。

#### 4.4 分阶段发送固件

Linux 发送流程分为三个函数：

```text
sendFileIndex()
    ↓
sendFileBlock()循环
    ↓
sendFileComplete()
```

分块循环每次最多从文件读取 `FILE_BLOCK_DATA_SIZE`，即 489 字节。最后一块可以小于 489 字节。

每块的确认过程为：

```text
读取offset位置的数据
    ↓
发送FileBlock
    ↓
等待ESP32响应
    ↓
检查fileIndex
    ↓
检查nextOffset == offset + dataLength
    ↓
更新本地offset
```

#### 4.5 Linux 进度输出

Linux 只在 ESP32 成功确认一个数据块后增加 `offset`，然后根据以下关系计算进度：

```text
percent = confirmedOffset × 100 / fileSize
```

因此 Linux 显示的是“ESP32 已确认处理”的进度，而不是单纯的文件读取进度：

```text
Progress: 100% (274970/274970 bytes)
```

### 5. ESP32 接收端实现

#### 5.1 Arduino 程序入口

`src/main.cpp` 的 `setup()` 初始化日志、LED 和协议 Slave：

```cpp
const SlaveStatus status = protocol.begin();
```

当前 `loop()` 负责翻转 GPIO8 LED，并检查 OTA 是否已经允许重启：

```cpp
if (protocol.restartReady()) {
    ESP.restart();
}
```

Protocol Slave 自己拥有接收任务，因此 `loop()` 中的 LED 延时不会停止协议任务接收数据。

#### 5.2 `AppProtocol` 分发业务请求

`AppProtocol::begin()` 把静态回调和当前对象指针交给 Slave：

```cpp
return slave_.begin(requestCallback, this);
```

静态回调通过 `context` 找回对象，再调用普通成员函数：

```text
Slave收到请求
    ↓
requestCallback(request, response, context)
    ↓
AppProtocol::handleRequest()
    ↓
根据MessageType分发
```

当前三个 OTA 消息被转发给：

```cpp
otaReceiver_.handleRequest(request, response);
```

`Control` 和 `ReadSensor` 已经预留在消息枚举中，但当前示例还没有实现，因此返回 `InvalidType`。

#### 5.3 `handleFileIndex()`：打开目标分区

ESP32 收到 `FileIndex` 后检查消息长度、文件名、文件索引和文件大小。如果已经有未完成的 `Update` 会话，则先调用：

```cpp
Update.abort();
```

然后开始新的应用更新：

```cpp
Update.begin(fileSize, U_FLASH);
```

这里的 `U_FLASH` 是 Arduino `Update` 的更新类型，表示输入数据是 ESP 应用固件。它不是 Flash 地址，也不是分区名称。使用 `U_FLASH` 时，`Update` 会通过 ESP-IDF OTA API 查找当前未运行的下一个 OTA 应用分区，例如当前运行 `ota_0` 时选择 `ota_1`。

与它对应的另一个常见类型是：

```cpp
U_SPIFFS
```

`U_SPIFFS` 用于更新 SPIFFS 或兼容的数据分区，不能用于启动 `firmware.bin`。

> `Update.begin(fileSize, U_FLASH)` 只负责选择备用应用分区并建立写入会话。它不会开始串口接收，也不会自动重启设备。

`begin()` 成功后，`OtaReceiver` 保存：

- 当前 `fileIndex`；
- 固件总大小；
- 下一个期望偏移 0；
- Linux 传来的 SHA-256；
- 当前进度百分比。

#### 5.4 `handleFileBlock()`：顺序写入固件

每个数据块写入前都要检查会话状态、`fileIndex`、数据长度和 `offset`。核心写入代码为：

```cpp
const size_t written = Update.write(data, dataLength);
if (written != dataLength) {
    // 中止会话并返回OtaWriteFailed
}
```

写入成功后，同时更新整个文件的 SHA-256 和下一偏移：

```text
sha256.update(data, dataLength)
nextOffset += dataLength
```

ESP32 使用 `Update.progress()` 和 `Update.size()` 查询写入进度，并且只在百分比变化时通过日志串口打印：

```text
OTA progress: 1% (3072/274970 bytes)
...
OTA progress: 100% (274970/274970 bytes)
```

最后把 `nextOffset` 放入响应，使 Linux 确认本块已经处理。

#### 5.5 `handleFileComplete()`：校验并提交

`FileComplete` 不包含固件数据，它表示 Linux 已经发送完全部内容。ESP32 依次执行：

```text
检查fileIndex
    ↓
检查nextOffset == fileSize
    ↓
结束SHA-256计算
    ↓
比较实际SHA-256与期望值
    ↓
Update.end()
```

SHA-256 不匹配时调用 `abortSession()`，不会提交目标分区。

`Update.end()` 成功后，目标应用分区被设置为下次启动分区，但当前 CPU 仍然在旧应用中运行。此时
`OtaReceiver` 先构造成功响应，然后设置延迟重启请求。

#### 5.6 延迟重启和原子状态

如果在 `handleFileComplete()` 中直接调用 `ESP.restart()`，Slave 可能还没有来得及把成功响应发送给 Linux。因此代码先记录：

```text
restartAtMs = millis() + 500
restartRequested = true
```

Arduino `loop()` 再通过 `restartReady()` 判断是否到达重启时间。

`restartRequested_` 和 `restartAtMs_` 使用 `std::atomic`，因为它们可能分别由 Protocol Slave 任务和 Arduino
`loopTask` 访问。原子读写避免一个任务读取到另一个任务尚未完整更新的状态。

> 这种设计把“OTA 请求重启”和“应用执行重启”分开。`OtaReceiver` 不需要创建额外常驻任务，也不会在协议回调中抢先复位。

### 6. 校验层次和安全边界

本工程不是只依赖一种校验，而是分层检查：

```text
Protocol帧CRC-32
    ↓ 检查单帧传输损坏
Transaction ID和响应类型
    ↓ 检查请求与响应是否匹配
fileIndex
    ↓ 检查消息是否属于当前固件会话
offset和nextOffset
    ↓ 检查分块顺序、重复和越界
整个firmware.bin的SHA-256
    ↓ 检查所有分块组合后的文件是否一致
Update.end()
    ↓ 检查并提交目标应用分区
Bootloader镜像验证
    ↓ 重启后决定是否真正执行新应用
```

各层解决的问题不同：

- CRC-32 只能说明当前协议帧在传输中没有明显损坏；
- `fileIndex` 和 `offset` 防止把其他会话或错误位置的数据写入当前固件；
- SHA-256 检查 Linux 文件和 ESP32 实际接收内容是否完全一致；
- `Update.end()` 和 Bootloader 负责应用分区与 ESP 镜像层面的检查。

当前示例没有实现固件签名和发送端身份认证。CRC-32 和 SHA-256 可以发现损坏，但不能阻止攻击者主动替换固件。需要安全 OTA 时，还应启用 Secure Boot、Flash Encryption 或应用层数字签名验证。

### 7. 接线、构建和运行

#### 7.1 两路串口的用途

本工程同时使用两路串口：

| 设备 | 用途 | 波特率 |
| --- | --- | ---: |
| `/dev/ttyACM0` | 开发板 USB 烧录和 ESP32 日志 | 115200 |
| `/dev/ttyUSB0` | 外接 USB-UART，传输 OTA 协议 | 921600 |

当前 OTA 协议 UART 配置位于 `include/protocol/config.h`，接线为：

```text
Linux USB-UART TX → ESP32-C3 GPIO20（RX）
Linux USB-UART RX ← ESP32-C3 GPIO21（TX）
Linux USB-UART GND ↔ ESP32-C3 GND
```

USB-UART 必须使用 3.3 V 逻辑电平，并且双方必须共地。日志使用默认 `Serial`，协议使用 `Serial1`，不要把日志文本发送到协议串口。

#### 7.2 配置 Linux 路径

运行前修改 `linux/main.cpp`：

```cpp
static const char* SERIAL_DEVICE = "/dev/ttyUSB0";
static const char* FIRMWARE_PATH =
    "/path/to/ota/.pio/build/adafruit_qtpy_esp32c3/firmware.bin";
```

建议为 `FIRMWARE_PATH` 使用绝对路径，避免从不同工作目录启动时找不到固件。

#### 7.3 第一次完整烧录

第一次必须通过 USB 烧录 Bootloader、分区表、OTA 初始状态和应用：

```bash
cd /path/to/ota
pio run -t upload
```

`platformio.ini` 当前使用 `/dev/ttyACM0` 作为上传口和日志口。烧录完成后查看 ESP32 日志：

```bash
pio device monitor
```

#### 7.4 构建 Linux 主机

在项目根目录执行：

```bash
cmake -S linux -B linux/build
cmake --build linux/build
```

生成的可执行文件为：

```text
linux/build/protocol_linux
```

修改 `linux/main.cpp` 中的固定路径后，需要重新执行 `cmake --build linux/build`。

#### 7.5 编译新的 OTA 固件

修改 ESP32 应用后，只编译新的应用镜像：

```bash
cd /path/to/ota
pio run
```

生成的 OTA 文件为：

```text
.pio/build/adafruit_qtpy_esp32c3/firmware.bin
```

这里不要执行 `pio run -t upload`，否则仍然是通过 USB 烧录，而不是测试 Linux OTA 流程。

#### 7.6 执行 OTA

确认 USB-UART 接线和两个固定路径正确后，直接运行：

```bash
./linux/build/protocol_linux
```

程序不需要附加命令行参数。Linux 会显示固件路径、大小、SHA-256、文件索引和发送进度；ESP32 日志串口会显示 OTA 开始、写入百分比、完成和重启信息。

#### 7.7 传输中断后重新升级

如果传输中断，可以重新执行：

```bash
./linux/build/protocol_linux
```

新的 `FileIndex` 会让 ESP32 中止未完成的旧会话，并从偏移 0 开始。当前示例没有实现断点续传。

在长度、SHA-256 和 `Update.end()` 全部成功之前，新的启动分区不会被提交，因此失败的传输不会替换当前正常运行的应用。

### 8. 工程文件关系

主要文件关系如下：

```text
ota/
├── platformio.ini                 ESP32构建、上传口和日志口配置
├── partitions.csv                 ESP32 Flash分区表
├── include/
│   ├── app/
│   │   ├── messages.h             应用消息、OTA字段和错误码
│   │   └── app_protocol.h         ESP32业务分发接口
│   ├── ota/
│   │   └── ota_receiver.h         ESP32 OTA接收器接口和会话状态
│   └── protocol/                  通用Protocol头文件
├── src/
│   ├── main.cpp                   Arduino入口、LED和延迟重启
│   ├── app/
│   │   └── app_protocol.cpp       Slave回调和业务消息分发
│   ├── ota/
│   │   └── ota_receiver.cpp       Update写入、SHA-256校验和提交
│   └── protocol/                  通用Protocol实现
└── linux/
    ├── main.cpp                   Linux固定路径和程序入口
    ├── CMakeLists.txt             Linux构建配置
    └── ota/
        ├── ota_sender.h           Linux OTA发送器接口
        └── ota_sender.cpp         固件检查、分块发送和进度输出
```

调用关系可以简化为：

```text
linux/main.cpp
    ↓
OtaSender::send()
    ↓
Master::transact()
    ↓ UART
Slave任务和回调
    ↓
AppProtocol::handleRequest()
    ↓
OtaReceiver::handleRequest()
    ↓
Update.begin() / write() / end()
    ↓
src/main.cpp检查restartReady()
    ↓
ESP.restart()
```
[源码](https://github.com/fazhehy/esp32-c3-ota)
