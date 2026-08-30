---
title: STM32H747学习笔记--MPU与Cache
published: 2026-08-31
description: 'STM32H747 Cortex-M7 MPU分区、内存属性及保护实验'
image: ''
tags: [embedded, stm32]
category: 'embedded'
draft: false 
lang: 'zh-CN'
---

# STM32H747 MPU与Cache

## 1. MPU基础

MPU (Memory Protection Unit, 内存保护单元) 是 CPU 内部的内存访问检查模块。它不分配内存，也不改变物理地址，而是为不同地址范围规定访问规则：

- 是否允许读写；
- 是否允许执行指令；
- 属于普通内存还是设备内存；
- 是否允许使用 Cache 和写缓冲；
- 特权程序和用户程序各自拥有什么权限。

CPU 每次访问内存时，MPU 都会检查地址和访问类型。访问违反规则时，会触发 `MemManage Fault`，从而尽早发现野指针、越界写入和错误的函数跳转。

> CM7 的错误处理函数是 `MemManage_Handler()`，位于 `CM7/Core/Src/stm32h7xx_it.c`。发生 MPU 访问违规后，内核通过启动文件中的异常向量表跳转到该函数。

> CubeMX 默认会生成 `MemManage_Handler()`，但复位后 MemManage 异常默认关闭，需要设置 `SCB->SHCSR` 的 `MEMFAULTENA` 位才能打开。否则 MPU 违规会升级为 HardFault。MPU 本身是否打开则由程序有没有执行 `HAL_MPU_Enable()` 决定。

```text
CPU发起访问
    ↓
MPU检查地址、权限和内存属性
    ├─ 不允许 → MemManage Fault
    └─ 允许   → 按指定的Cache和Buffer属性访问总线
```

MPU、Cache 和 Buffer 的关系可以概括为：

```text
MPU    ：制定每段地址的访问规则
Cache  ：保存指令或数据的高速副本
Buffer ：暂存尚未完成的写请求
```

MPU 将区域设为 Cacheable，只表示该区域允许缓存，并不会自动打开 Cache。Cortex-M7 复位后 L1 Cache 默认关闭，通常需要在启动阶段由软件显式调用：

```c
SCB_EnableICache();  // 打开指令Cache
SCB_EnableDCache();  // 打开数据Cache
```

一般先配置 MPU，再启用 Cache：

```c
MPU_Config();
SCB_EnableICache();
SCB_EnableDCache();
HAL_Init();
```

> CubeMX 新建并生成的当前默认工程没有打开 Cache：生成代码中没有 `SCB_EnableICache()` 和 `SCB_EnableDCache()`。CubeMX 生成并调用 `MPU_Config()` 也只是在配置内存属性，不等于启用了 Cache。
>
> “手动启用”是指程序中必须有代码执行上述函数，不一定要求开发者每次亲自调用。如果 Bootloader、BSP 或系统初始化代码已经启用，就不需要再次启用。可以读取 `SCB->CCR` 的 `IC` 和 `DC` 位判断当前状态。当前工程没有调用这两个函数，因此虽然部分 MPU Region 被设置为 Cacheable，CM7 的 L1 Cache 实际仍然关闭。

STM32H747 是双核芯片，CM7 和 CM4 各有独立的 MPU：

- CM7 支持 Region 0～15，并带有 L1 I-Cache 和 D-Cache；
- CM4 支持 Region 0～7，没有 CM7 这种 L1 I-Cache 和 D-Cache；
- 配置 CM7 的 MPU 不会自动配置 CM4。

## 2. MPU内存分区

MPU 使用 Region 管理地址空间。一个 Region 主要由以下内容组成：

```text
Region编号 + 基地址 + 区域大小 + 访问权限 + 内存属性
```

### Region大小和地址对齐

Region 大小必须是 2 的幂，最小为 32B，例如：

```text
32B、64B、128B、256B、1KB、2KB……512KB……4GB
```

基地址还必须按 Region 大小对齐。以 128B Region 为例：

```text
0x20000180：合法，能够被0x80整除
0x20000140：不合法，没有按0x80对齐
```

这也是实验数组需要使用 `alignas(128)` 的原因：

```cpp
alignas(128) static volatile uint8_t mpu_test_data[128];
```

> `alignas(128)` 是 C++ 的对齐说明符，要求编译器把 `mpu_test_data` 的起始地址放在 128B 边界上，也就是地址必须能被 128 整除。它只改变变量的对齐要求，不会把数组大小改成其他值；这里数组本身仍然是 128B。
>
> 如果不使用 `alignas(128)`，链接器可能把数组放在任意合法地址，例如 `0x20000140`。这个地址虽然可以正常存放数组，却不能作为 128B MPU Region 的基地址。使用 `alignas(128)` 后，实验中的数组地址为 `0x20000180`，既满足 Region 对齐要求，又因为数组大小正好为 128B，不会把相邻变量包含进只读保护区域。

### Subregion

大小不小于 256B 的 Region 可以平均分成 8 个 Subregion。`SubRegionDisable` 的每一位控制一个子区域：

```text
0：启用该子区域
1：关闭该子区域
```

工程中的 Region 0 配置为 4GB，并使用：

```c
mpu_region.SubRegionDisable = 0x87U;
```

4GB 被分成 8 个 512MB 子区域。`0x87` 关闭子区域 0、1、2、7，Region 0 最终覆盖：

```text
0x60000000～0xDFFFFFFF
```

该范围被设置为禁止访问，用作外部地址空间的背景保护区。

### Region重叠和优先级

Region 可以重叠，编号越大，优先级越高。例如：

```text
Region 0：0x60000000～0xDFFFFFFF，禁止访问
Region 6：0xD0000000～0xD1FFFFFF，允许访问SDRAM
```

访问 SDRAM 时，Region 6 覆盖 Region 0，因此 SDRAM 可以正常使用；未被高编号 Region 覆盖的外部地址仍然禁止访问。这就是“默认拒绝，按需开放”的配置方法。

## 3. 内存属性

### 访问权限

常用访问权限如下：

| 配置                     | 含义                     |
| ------------------------ | ------------------------ |
| `MPU_REGION_NO_ACCESS`   | 特权级和用户级都不能访问 |
| `MPU_REGION_PRIV_RW`     | 仅特权级可读写           |
| `MPU_REGION_PRIV_RW_URO` | 特权级可读写，用户级只读 |
| `MPU_REGION_FULL_ACCESS` | 特权级和用户级都可读写   |
| `MPU_REGION_PRIV_RO`     | 仅特权级可读             |
| `MPU_REGION_PRIV_RO_URO` | 特权级和用户级都只读     |

`DisableExec` 用于控制区域中的内容能否作为指令执行：

```c
MPU_INSTRUCTION_ACCESS_ENABLE   // 允许执行
MPU_INSTRUCTION_ACCESS_DISABLE  // 禁止执行，即Execute Never
```

堆、栈、DMA 缓冲区和外设地址通常不应该执行代码，可以设置为禁止执行。

### Normal、Device和Strongly Ordered

MPU 使用 `TEX`、`C`、`B` 和 `S` 描述一个 Region 的内存属性。它们在 HAL 结构体中的对应关系如下：

| 字段  | 全称           | HAL成员        | 作用                                                     |
| ----- | -------------- | -------------- | -------------------------------------------------------- |
| `TEX` | Type Extension | `TypeExtField` | 与C、B组合，选择内存类型和Cache策略                      |
| `C`   | Cacheable      | `IsCacheable`  | 表示该区域是否允许缓存                                   |
| `B`   | Bufferable     | `IsBufferable` | 表示是否允许写缓冲；在Normal Memory中也参与选择Cache策略 |
| `S`   | Shareable      | `IsShareable`  | 表示该区域是否可能被其他总线主设备共享                   |

其中 `TEX/C/B` 必须组合起来查表，不能简单地理解为三个独立开关。常用组合如下：

| TEX | C   | B   | 得到的内存类型                           |
| --- | --- | --- | ---------------------------------------- |
| 0   | 0   | 0   | Strongly Ordered，不缓存、不缓冲         |
| 0   | 0   | 1   | Device Memory，不缓存、允许写缓冲        |
| 0   | 1   | 0   | Normal Memory，Write-Through             |
| 0   | 1   | 1   | Normal Memory，Write-Back                |
| 1   | 0   | 0   | Normal Memory，Non-cacheable             |
| 1   | 1   | 1   | Normal Memory，Write-Back并支持读/写分配 |

> **Write-Through（写直达）**：CPU 写入 Cache 的同时，也把写请求发送到真实内存。内存能较快获得新数据，但每次写入都会占用总线，性能通常低于 Write-Back。该组合采用 No Write Allocate，写未命中时不会专门加载一条新的 Cache Line。
>
> **Write-Back（写回）**：CPU 先修改 Cache，并把该 Cache Line 标记为 Dirty，直到执行 Clean 或发生 Cache Line 替换时才写回真实内存。表中的 `TEX/C/B=0/1/1` 支持读分配，但采用 No Write Allocate：读取未命中会加载 Cache Line，写入未命中不会因此分配新的 Cache Line。
>
> **Write-Back并支持读/写分配**：读取未命中和写入未命中都会分配 Cache Line。写入未命中时，硬件先把对应内存块加载进 Cache，再修改 Cache 中的数据，适合会被连续或重复访问的普通内存。
>
> **Non-cacheable 与“不缓存”的区别**：从结果看，两者都不会把数据放进 Cache；但 `Normal Non-cacheable` 仍然属于普通内存，允许普通内存的突发传输、访问合并和一定程度的重排，适合 SRAM、SDRAM 或 DMA 共享缓冲区。Device 和 Strongly Ordered 的“不缓存”则带有设备访问语义，会限制推测访问、访问合并和重排，用于具有读写副作用或严格顺序要求的外设。选择 MPU 属性时不能因为它们都“不缓存”就互相替换。

三种主要内存类型的用途是：

| 类型             | 典型用途           | 特点                           |
| ---------------- | ------------------ | ------------------------------ |
| Normal Memory    | SRAM、SDRAM        | 可以使用Cache，允许CPU优化访问 |
| Device Memory    | 外设寄存器         | 不缓存，保证必要的设备访问顺序 |
| Strongly Ordered | 严格时序的设备区域 | 不缓存、不缓冲，访问约束最严格 |

以当前工程的 AXI SRAM 为例：

```c
mpu_region.TypeExtField = MPU_TEX_LEVEL0;             // TEX=0
mpu_region.IsCacheable = MPU_ACCESS_CACHEABLE;        // C=1
mpu_region.IsBufferable = MPU_ACCESS_BUFFERABLE;      // B=1
mpu_region.IsShareable = MPU_ACCESS_NOT_SHAREABLE;    // S=0
```

逐项组合后得到：

```text
TEX/C/B = 0/1/1 → Normal Write-Back Cacheable Memory
S       = 0     → Non-shareable
```

因此 CPU 可以缓存 AXI SRAM 中的数据。CPU 写入时通常先修改 D-Cache，并把对应 Cache Line 标记为 Dirty，之后再写回真实 SRAM。需要注意，只有调用 `SCB_EnableDCache()` 后，Cacheable 属性才会真正产生 D-Cache 效果。

> `S` 表示地址区域的共享属性，不代表硬件会自动处理 CPU 与 DMA 之间的 Cache 一致性。CM7 使用可缓存内存与 DMA 交换数据时，仍然需要执行 Clean 或 Invalidate。

### Cache和Buffer

Cache 是位于 CPU 内核和主存之间的一小块高速存储器，用来保存近期使用过的指令和数据副本。SRAM、SDRAM 和 Flash 的访问速度比 CPU 执行速度慢，Cache 可以减少 CPU 真正访问这些存储器的次数。

```text
              ┌─ I-Cache：缓存程序指令
CPU内核 ──────┤
              └─ D-Cache：缓存程序数据
                       ↓
                 SRAM/SDRAM/Flash
```

Cache 不会改变数据原本属于哪块内存。例如变量仍然位于 AXI SRAM，D-Cache 中只是暂时保存了它的副本。

Cortex-M7 以 Cache Line 为基本管理单位，每条 Cache Line 为 32B。CPU 即使只读取一个字节，也会把包含该地址的一整组 32B 数据加载到 Cache。

例如 CPU 读取 `0x24000004`：

```text
请求地址：0x24000004
加载范围：0x24000000～0x2400001F，共32B
```

这样继续读取 `0x24000008`、`0x2400000C` 等相邻地址时，就可以直接使用已经加载的数据。每条 Cache Line 除了保存数据，还会记录地址标记以及 `Valid`、`Dirty` 等状态。

CPU读取数据时，硬件自动执行以下过程：

```text
CPU读取地址
    ↓
检查D-Cache中是否有对应Cache Line
    ├─ Cache Hit：直接返回Cache中的数据
    └─ Cache Miss：从内存读取32B并放入Cache，再返回所需数据
```

程序通常不需要主动把数据加载进 Cache。只要满足以下两个条件，第一次访问发生 Miss 时，硬件就会自动加载：

1. MPU 将该地址配置为 Cacheable；
2. 已经调用 `SCB_EnableDCache()` 或 `SCB_EnableICache()`。

CPU写数据时还要看写入策略：

- Write-Through：同时修改 Cache 和真实内存，数据比较及时地对外可见，但总线写入次数较多；
- Write-Back：先修改 Cache，并将 Cache Line 标记为 Dirty，之后才写回真实内存，性能更高；
- Write Allocate：写入发生 Miss 时，先为该地址分配并加载 Cache Line；
- No Write Allocate：写入发生 Miss 时，不为它建立新的 Cache Line。

当前工程的 `TEX/C/B=0/1/1` 使用 Write-Back。此时可能出现：

```text
CPU通过D-Cache看到新数据
真实内存中仍然是旧数据
```

Cache 容量有限。需要加载新的 Cache Line 时，硬件会选择旧的 Cache Line 进行替换：

```text
旧Cache Line是Clean → 可以直接丢弃
旧Cache Line是Dirty → 先写回内存，再进行替换
```

> Cache 的加载和替换主要由硬件自动完成；程序负责正确设置 MPU 内存属性，并在 CPU 与 DMA、LTDC 等设备共享数据时执行必要的 Cache 维护。

Write Buffer 可以近似理解为 CPU 到总线之间的写请求队列：

```text
CPU提交写请求 → Write Buffer → 总线 → 内存或外设
```

CPU 将写请求交给 Buffer 后可以继续执行，硬件会自动完成后续写入。需要等待访问完成或约束顺序时，可以使用 `__DSB()`、`__DMB()` 和 `__ISB()`。

如果 DMA、LTDC 等外设与 CPU 共享可缓存内存，还需要维护 D-Cache 一致性：

```text
CPU写、外设读：启动外设前Clean D-Cache
外设写、CPU读：外设完成后Invalidate D-Cache
```

## 4. STM32H747配置实例

### 配置接口

工程使用 `mpu_set_protection()` 配置单个 Region：

```c
void mpu_set_protection(uint32_t base_address,
                        uint32_t size,
                        uint32_t region_number,
                        uint32_t disable_exec,
                        uint32_t access_permission,
                        uint32_t shareable,
                        uint32_t cacheable,
                        uint32_t bufferable);
```

内部配置顺序为：

```c
HAL_MPU_Disable();
HAL_MPU_ConfigRegion(&mpu_region);
HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);
```

`MPU_PRIVILEGED_DEFAULT` 表示：特权程序访问未命中任何 Region 的地址时，继续使用处理器默认内存映射。

### 工程Region表

CM7 启动时通过 `mpu_memory_protection()` 配置以下区域：

| Region | 基地址       | 大小        | 用途             | 权限               | Cache/Buffer              |
| ------ | ------------ | ----------- | ---------------- | ------------------ | ------------------------- |
| 0      | `0x00000000` | 4GB部分子区 | 外部空间背景保护 | 禁止访问、禁止执行 | 禁止                      |
| 1      | `0x20000000` | 128KB       | DTCM             | 完全访问           | C+B，TCM实际不经过D-Cache |
| 2      | `0x24000000` | 512KB       | AXI SRAM         | 完全访问           | C+B                       |
| 3      | `0x30000000` | 512KB       | D2 SRAM窗口      | 完全访问           | C+B                       |
| 4      | `0x38000000` | 64KB        | SRAM4            | 完全访问           | C+B                       |
| 5      | `0x60000000` | 64MB        | FMC/LCD          | 完全访问           | 禁止                      |
| 6      | `0xD0000000` | 32MB        | SDRAM            | 完全访问           | C+B                       |
| 7      | `0x80000000` | 256MB       | NAND             | 完全访问、禁止执行 | 禁止                      |

> 上述配置只针对 STM32H747 的 CM7，不能直接套用到 CM4。两个内核拥有彼此独立的 MPU，CM7 执行 `HAL_MPU_ConfigRegion()` 只会修改 CM7 的 MPU 寄存器，不会影响 CM4。
>
> CM7 支持 Region 0～15，而 CM4 只支持 Region 0～7，因此后面实验使用的 Region 8 在 CM4 上不存在。CM4 的链接脚本和本地内存地址也不同，例如当前 CM4 工程把 D2 SRAM 映射到 `0x10000000`，不能照搬 CM7 的 DTCM 地址 `0x20000000`。另外，CM4 没有 CM7 的 L1 I-Cache和D-Cache。若要在 CM4 使用 MPU，应根据 CM4 的链接脚本重新规划地址和 Region，并在 CM4 自己的启动流程中单独调用配置函数。

当前工程只配置了 Cacheable 属性，没有调用 `SCB_EnableICache()` 或 `SCB_EnableDCache()`，因此 L1 Cache 尚未真正开启。

### 只读保护实验

实验在 DTCM 中创建一个 128B 对齐数组。Region 1 原本允许读写整个 DTCM，因此第一次写入可以成功：

```cpp
mpu_test_data[0] = 0x5AU;
```

然后使用优先级更高的 Region 8 覆盖数组所在的 128B，并设置为只读：

```cpp
mpu_set_protection(
    (uint32_t)mpu_test_data,           // 数组地址
    MPU_REGION_SIZE_128B,              // Region大小
    MPU_REGION_NUMBER8,                // 高于Region 1
    MPU_INSTRUCTION_ACCESS_DISABLE,    // 禁止执行
    MPU_REGION_PRIV_RO_URO,            // 只读
    MPU_ACCESS_NOT_SHAREABLE,
    MPU_ACCESS_NOT_CACHEABLE,
    MPU_ACCESS_NOT_BUFFERABLE);
```

MemManage Fault 复位后默认未使能，需要手动开启：

```c
SCB->SHCSR |= SCB_SHCSR_MEMFAULTENA_Msk;
__DSB();
__ISB();
```

> MPU 错误处理函数是 `MemManage_Handler()`，位于 CM7 工程的 `CM7/Core/Src/stm32h7xx_it.c` 文件中。启动文件的异常向量表会在发生 MemManage Fault 时跳转到该函数。

> CubeMX 默认会生成 `MemManage_Handler()`，但复位后 `MEMFAULTENA` 默认关闭，仅仅存在处理函数并不代表异常已经打开。当前工程在启动时调用 `MPU_Config()`，所以 MPU 已经打开；实验代码还需要设置 `SCB->SHCSR` 来打开 MemManage 异常，否则 MPU 违规会升级为 HardFault。

保护后的读取仍然成功，再次写入则触发异常：

```cpp
mpu_test_data[0] = 0xA5U;
```

实际串口输出如下：

```text
[=LOG=]: MPU experiment started on CM7
[=LOG=]: Before protection: write OK, value=0x5A
[=LOG=]: Protection enabled: read OK, value=0x5A
[=LOG=]: Writing again; the next message should come from MemManage_Handler
[ERROR]: MemManage Fault: protected write blocked, CFSR=0x00000082, MMFAR=0x20000180
```

`CFSR=0x82` 表示：

- `DACCVIOL=1`：发生数据访问权限违规；
- `MMARVALID=1`：`MMFAR` 中的错误地址有效。

`MMFAR=0x20000180` 正好是测试数组地址，说明 Region 8 成功覆盖 Region 1，并阻止了只读区域的写入。

## 5. 注意事项与总结

配置 MPU 时需要重点检查：

1. Region 大小必须是 2 的幂，基地址必须按大小对齐；
2. 重叠时高编号 Region 优先，修改配置前先确认覆盖关系；
3. 外设寄存器不能配置成普通 Cacheable Memory；
4. DMA 缓冲区应设为不可缓存，或者在 DMA 前后正确执行 Clean/Invalidate；
5. Cache 维护按 32B Cache Line 工作，DMA 缓冲区最好按 32B 对齐；
6. `HAL_MPU_Disable()` 只关闭 MPU，不会自动清除已经配置的 Region；
7. CM7 与 CM4 的 MPU 相互独立，必须分别配置；
8. 必须使能 `MEMFAULTENA`，否则 MPU 违规会升级为 HardFault。

MPU 的核心作用不是提高速度，而是定义每段地址“能不能访问、能不能执行、应该怎样访问”。Cache 和 Buffer 根据这些内存属性工作，在提高性能的同时，也带来 DMA 一致性和访问顺序问题。实际项目中应先规划内存用途，再决定每个 Region 的权限和属性。

[工程源码](https://github.com/fazhehy/STM32-HAL-Drivers/tree/main/STM32H747/mpu)
