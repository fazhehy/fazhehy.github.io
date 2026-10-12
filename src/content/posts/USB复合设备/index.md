---
title: STM32H747学习笔记--USB复合设备
published: 2026-10-12
description: 'USB 设备结构、枚举与端点分配，以及一个 USB 口同时提供虚拟串口和读卡器的实现'
image: ''
tags: [embedded, stm32]
category: 'embedded'
draft: false
lang: 'zh-CN'
---

# 第一部分：USB 基础

## 1. 主机、设备与分层拓扑

USB 是**分层星形**拓扑，不是一条线挂满器件：

```text
           主机（Host）
               │
           根集线器
          ┌────┼────┐
       设备  集线器  设备
               │
           ┌───┴───┐
         设备     设备
```

主机只有一个，设备靠集线器一层层分叉，每个设备一个 7 位地址，一条总线上最多 127 个。

关键的是主从关系：

- **主机是唯一的发起者**。每次传输都由主机先发令牌指名"哪个端点、哪个方向"，设备只能应答。
- **设备不能主动发起**。设备想说"我有数据了"，只能把数据准备好等主机来取。

设备永远没有"推"的能力，只有"准备好了"和"还没准备好"两种状态。

## 2. 包、事务与传输类型

### 包与事务

最小单位是**包**（Packet），三类：

| 包                      | 谁发 | 装什么                   |
| ----------------------- | ---- | ------------------------ |
| **令牌包**（Token）     | 主机 | 目标设备地址、目标端点号 |
| **数据包**（Data）      | 双方 | 要传的字节               |
| **握手包**（Handshake） | 双方 | 收到 / 没收到 / 出错     |

一次完整的"主机发起 → 数据 → 应答"叫**事务**（Transaction），按令牌包分 OUT（主机写）、IN（主机读）、SETUP（控制命令，只在枚举和配置时用）三种。

### 四种传输类型

传输由事务拼成。USB 定义了四种，各自的保证不同：

| 类型                    | 保证                 | 典型用途           | 本工程         |
| ----------------------- | -------------------- | ------------------ | -------------- |
| **控制**（Control）     | 一定送到             | 枚举、配置、类请求 | EP0，必用      |
| **批量**（Bulk）        | 一定送到，不保证时间 | 大块数据           | CDC 数据、MSC  |
| **中断**（Interrupt）   | 保证延迟上限         | 少量且要及时       | CDC 的通知端点 |
| **同步**（Isochronous） | 保证带宽，不保证正确 | 音视频             | 不用           |

U 盘的数据流和虚拟串口的字节流都是批量传输 —— 不能丢，但慢一点没关系。

"一定送到"靠重传：数据包带 CRC，接收方校验不过就不回应答，发送方超时重发。重发导致的重复靠 **DATA0 / DATA1** 标记轮流翻转去重。

**批量传输在线上没有长度字段**，一批数据靠**短包**收尾：一包小于端点的最大包长就表示结束。发 100 字节会拆成 64 + 36，36 < 64 即结束。

主机每 1 ms 发一个 SOF 包开启一帧，传输在帧里排队。中断传输按帧预约，本工程的 CDC 通知端点是每 16 帧轮询一次（`bInterval = 0x10`）。

## 3. 端点：一条方向一条通道

### 端点地址：一个字节装两件事

**端点**（Endpoint）是设备内部的可寻址通道，也是软件和 USB 硬件之间真正的接口。地址是一个字节：

```text
bit7      方向：1 = IN（设备→主机），0 = OUT（主机→设备）
bit6-4    保留
bit3-0    端点号（0 ~ 15）
```

- `0x83` = `0x80 | 0x03`：端点 3 的 IN 方向；
- `0x03` = `0x00 | 0x03`：端点 3 的 OUT 方向。

这是**两个不同的端点**。硬件上各占一套寄存器：IN 端点从 `0x900` 起、OUT 从 `0xB00` 起，每个占 `0x20` 字节，端点 3 的 IN 在 `0x960`、OUT 在 `0xB60`。

端点都是单向的，唯一例外是 **EP0** —— 双向，规范强制要求存在，专跑控制传输。

方向信息每层的表达方式不同：

| 层                 | 方向靠什么表示                                     |
| ------------------ | -------------------------------------------------- |
| 描述符 / ST 的 API | 端点地址的 bit7（`0x83` / `0x03`）                 |
| USB 设备库内部     | 进 `IN_ep[]` 还是 `OUT_ep[]`                       |
| DWC2 寄存器        | 写 `DIEP*`（`0x900` 起）还是 `DOEP*`（`0xB00` 起） |
| 线上               | 令牌包的 PID 是 IN 还是 OUT                        |

线上那层最彻底，令牌包格式是：

```text
PID(8) + 设备地址(7) + 端点号(4) + CRC5(5)
```

端点号字段只有 4 位，装不下方向；方向由 PID 表达 —— 主机发 IN 令牌还是 OUT 令牌，硬件据此决定查哪半边寄存器。`0x83` 这个字节不会被发到线上，也不是硬件字段，在 HAL 里只活两行：

```c
if ((ep_addr & 0x80U) == 0x80U) {
    ep = &hpcd->IN_ep[ep_addr & 0x0FU];     /* bit7 决定哪半边 */
    ep->is_in = 1U;
} else {
    ep = &hpcd->OUT_ep[ep_addr & 0x0FU];
    ep->is_in = 0U;
}
ep->num = ep_addr & 0x0FU;                  /* 抹掉 bit7 拿到号 */
```

往下只剩 `IN_ep[3]`，写寄存器时是 `DIEPCTL(3)`。**bit7 是"一个字节同时带号 + 方向"的省参数写法**，软件约定而已。

**端点是设备自己定的**。

- 规范只规定范围（号 0~15、类型、包长上限），不规定用哪几个；
- 设备在**端点描述符**里声明自己有哪些端点，主机照着读；
- 主机**没有分配权**，只有**否决权** —— 声明不合规范它就沉默，枚举超时，设备在电脑上不出现。PCIe 反过来，地址和中断号是主机分配的。

硬规则：**号 + 方向不能重复**。复合改造第一处被逼着改的就是这个 —— 两个类的默认值完全重叠：

| 端点         | CDC 类库默认 | MSC 类库默认 |
| ------------ | ------------ | ------------ |
| 1 IN         | `0x81`       | `0x81`       |
| 1 OUT        | `0x01`       | `0x01`       |
| 2 IN（中断） | `0x82`       | ——           |

必须有一个让路。本工程让 MSC 搬到 `0x83` / `0x03`。

### 端点号的选择范围

可选范围是几层约束的交集：

| 约束来自  | 限制                                                                                                  |
| --------- | ----------------------------------------------------------------------------------------------------- |
| 规范      | EP0 锁定给控制传输；号只有 0~15；类型决定包长上限（全速批量只能是 8/16/32/64，中断 1~64，等时 ≤1023） |
| 芯片      | `dev_endpoints = 9`，只有 EP0~EP8；`use_dedicated_ep1` 一开 EP1 就被独占                              |
| FIFO 预算 | 所有 IN 端点共享一块 FIFO RAM，每个至少 16 字                                                         |
| 库        | 一个类最多 5 个端点槽位                                                                               |
| 设备自己  | 号 + 方向不能重复                                                                                     |

最容易被忽略的是 FIFO 预算。DWC2 的 FIFO RAM 是共享的：一个 RX FIFO 给所有 OUT 端点共用，每个 IN 端点再从同一块里切一份 TX FIFO。本工程的账是：

```text
RX 128 │ EP0 64 │ EP1 64 │ EP2 16 │ EP3 32   = 304 字，按 320 字预算
                  └─ CDC 数据 └─ CDC 中断 └─ MSC
```

只有 IN 半边要 TX FIFO，OUT 端点不占。

TX FIFO 的偏移是把前面的 FIFO 大小累加算出来的，所以**跳号有代价**：用 EP5 而 EP4 空着，空掉的 EP4 也必须配 16 字，否则后面所有 FIFO 地址错位、数据踩到别人的地盘。320 字预算里跳一个号白扔 16 字。IN 端点必须挨着排、不留洞。

回到 `0x83` / `0x03` 的来历：

- **`bit7` 是写法** —— 0x80 是 IN，0x00 是 OUT；
- **号 3 是资源分配** —— 0 被 EP0 占，1、2 被 CDC 占，剩下第一个空号就是 3。可选 3~8，选 3 是为了连续：1、2、3 挨着排，FIFO 里没有空洞。

这个值写在这一行：

```c
static uint8_t msc_endpoints[] = {0x83U, 0x03U};         /* 批量IN, 批量OUT */
static uint8_t cdc_endpoints[] = {0x81U, 0x01U, 0x82U};  /* 批量IN, 批量OUT, 中断IN */
```

**数组只给号，类型和方向由构建器按类写死**：MSC 的 `EpAdd[0]/[1]` 是批量 IN / 批量 OUT；CDC 的 `EpAdd[0]/[1]/[2]` 是批量 IN / 批量 OUT / 中断 IN。元素先后有意义 —— `{0x83U, 0x03U}` 写成 `{0x03U, 0x83U}` 会把两个方向调个个儿，能编译、能枚举，但一个字节都传不对。

### 本工程的端点表

| 端点   | 方向 | 类型 | 属于       | 最大包长 |
| ------ | ---- | ---- | ---------- | -------- |
| `0x00` | 双向 | 控制 | 枚举与配置 | 64       |
| `0x81` | IN   | 批量 | CDC 数据   | 64       |
| `0x01` | OUT  | 批量 | CDC 数据   | 64       |
| `0x82` | IN   | 中断 | CDC 通知   | 8        |
| `0x83` | IN   | 批量 | MSC        | 64       |
| `0x03` | OUT  | 批量 | MSC        | 64       |

EP0 之外 5 个业务端点：CDC 3 个（一进一出传数据 + 一个中断端点发状态通知），MSC 2 个（一进一出传扇区数据）。CDC 的通知端点只有 8 字节、16 ms 轮询一次，运的是"串口线状态变了"这类几个字节的通知。

## 4. 设备、配置、接口、类

四层结构：

```text
设备（Device）
 └─ 配置（Configuration）
     └─ 接口（Interface）
         └─ 端点（Endpoint）
```

- 一个设备可以有多个**配置**，同一时刻只能用其中一个。本工程只有一个。
- **接口**是一组功能的单位：一套端点 + 一个类，编号从 0 开始。
- **类**（Class）是规范定义好的"这一类设备的工作方式"：**CDC** 是通信设备（虚拟串口），**MSC** 是大容量存储（U 盘、读卡器），另有 HID、Audio 等。

**类是按接口绑的，不是按设备绑的**。主机枚举完后，拿每个接口的 `bInterfaceClass` 去挑驱动：

```text
接口 0   类 = MSC   →  挂 U 盘驱动
接口 1   类 = CDC   →  挂 CDC 驱动
接口 2   类 = CDC   →  同一个 CDC 驱动（数据接口）
```

设备这一侧没有标志位说"我有两个功能"，它只是"一条配置里挂了好几个接口"。主机数完接口、各自绑驱动，才得出结论：这是复合设备。**复合是主机侧的结论，不是设备侧的声明**。

设备描述符里的 `bDeviceClass` 配合这件事：本工程填 `0x00`，意思是"别看我，去看每个接口的类"。整台设备都属于某一个类时才填那个类的编号。

**类不等于接口**：CDC 占**两个**接口 —— 一个通信接口管控制命令，一个数据接口管字节流。所以本工程接口数是 3：MSC 一个，CDC 两个。

## 5. 描述符与枚举

### 描述符是设备里写死的字节

设备靠**描述符**（Descriptor）向主机介绍自己，是格式固定的字节数组，烧在固件里：

| 描述符           | 内容                                             |
| ---------------- | ------------------------------------------------ |
| **设备描述符**   | VID、PID、USB 版本、厂商/产品/序列号字符串的索引 |
| **配置描述符**   | 这条配置总长多少、有几个接口                     |
| **接口描述符**   | 接口号、类、子类、协议、用几个端点               |
| **端点描述符**   | 端点地址、类型、最大包长、轮询间隔               |
| **字符串描述符** | 厂商名、产品名、序列号（UTF-16 编码）            |

**这些字节是预先写好的，不是和主机协商出来的**，主机只把它们读走。要把产品名从 `STM32 Mass Storage` 改成 `STM32 Composite`，只能改固件重新烧写。

### 枚举流程

从插上设备到能收发数据，中间这一串对话叫**枚举**（Enumeration），全程走 EP0 的控制传输：

| 步骤 | 主机                                                 | 设备                                   |
| ---- | ---------------------------------------------------- | -------------------------------------- |
| 1    | 检测到 D+ 被上拉，知道插进来一个全速设备             | 上电后把 D+ 拉高                       |
| 2    | 等约 100 ms，然后发一次 USB 复位                     | 复位，回到默认状态                     |
| 3    | `GET_DESCRIPTOR` 读设备描述符，先只要**前 8 个字节** | 返回 18 字节描述符的前 8 个            |
| 4    | 从这 8 字节里知道了最大包长，再读完整的 18 字节      | 返回全部                               |
| 5    | `SET_ADDRESS`（比如 4）                              | 记下自己的新地址                       |
| 6    | 读配置描述符、接口描述符、端点描述符、字符串描述符   | 逐个返回                               |
| 7    | `SET_CONFIGURATION`（1）                             | **分水岭**：各类驱动初始化，端点被打开 |
| 8    | 按每个接口的类，逐个绑定驱动                         | 正常工作                               |

第 3 步分两次读，是因为主机这时还不知道 EP0 能收多大一包，不敢多要 —— `bMaxPacketSize0` 正好在这前 8 字节里。

第 7 步是分界：

- 在这之前设备只是"能被读描述符"，业务端点一个都没开，收也收不了、发也发不了；
- 处理 `SET_CONFIGURATION` 时，库会把每个类的初始化函数挨个调一遍（本工程 MSC、CDC 各一次），端点被打开、FIFO 被分配、缓冲区被登记；
- 成功后设备状态变成 `USBD_STATE_CONFIGURED` —— 应用层那句"现在能不能发数据"等的就是它。

# 第二部分：复合设备的构成

## 1. 接口的划分与编号

接口号由**类的注册顺序**决定。构建器把已激活类占的接口数累加，得到下一个空号：先注册的 MSC 得 0，后注册的 CDC 得 1 和 2。每个类占几个接口是写死的 —— MSC 一个，CDC 两个，一个管控制命令，一个管字节流。

代码里就是 `usb_device.c` 里两次注册调用，MSC 在前。两句对调也能跑，接口号整体互换，没有一处硬编码会对不上。

## 2. IAD

两个接口属于同一个串口这件事，需要设备明确告诉主机。CDC 规范自己的办法是在通信接口的描述符组里放一条 Union Functional Descriptor（子类型 0x06），指明 master 和 slave 接口，但它藏在类专有描述符里，主机得先认出 CDC 类才会去解析。跨类的通用机制是 IAD，排在接口描述符之前，固定 8 字节：

| 字段                | 值     | 含义                     |
| ------------------- | ------ | ------------------------ |
| `bLength`           | 8      | 长度                     |
| `bDescriptorType`   | `0x0B` | IAD                      |
| `bFirstInterface`   | 1      | 起点接口                 |
| `bInterfaceCount`   | 2      | 连续两个接口属于同一功能 |
| `bFunctionClass`    | `0x02` | CDC                      |
| `bFunctionSubClass` | `0x02` | ACM                      |
| `bFunctionProtocol` | `0x01` | AT 命令                  |
| `iFunction`         | 0      | 无字符串                 |

主机的 USB 核心先读描述符，看到 IAD 就把从这里起的连续两个接口交给同一个驱动。Linux 的 `cdc_acm` 靠这条合成一个 `/dev/ttyACM`。没有它，两个接口各成一个设备，串口起不来。

这一段要写的是一个宏。生成 IAD 的代码由 `#if USBD_COMPOSITE_USE_IAD == 1` 包着，宏库里**只读不定义**，不写就按 0 处理，整段不生成。要在 `usbd_conf.h` 里自己加上并填 1，漏了的表现是编译通过、硬件不报错、主机那边串口不出现。

## 3. 配置描述符的拼装

拼装发生在**注册那一刻**，不是枚举时现算。每个类把段落追加到静态缓冲末尾，再把已写长度推后，下一个类接着写；配置描述符头的总长和接口数由每个类写完后重写一次，MSC 加一，CDC 加二。缓冲全速和高速各一块，上限 300 字节（`USBD_CMPST_MAX_CONFDESC_SZ`）。

本工程拼出来 98 字节、十四个段落：

| #   | 段落              | 长  | 关键字段                             |
| --- | ----------------- | --- | ------------------------------------ |
| 1   | 配置描述符        | 9   | 总长 98、接口数 3                    |
| 2   | MSC 接口          | 9   | 接口 0，0x08 / 0x06 / 0x50，2 个端点 |
| 3   | MSC 批量 IN 端点  | 7   | `0x83`，批量，64                     |
| 4   | MSC 批量 OUT 端点 | 7   | `0x03`，批量，64                     |
| 5   | IAD               | 8   | 接口 1 起 2 个，0x02 / 0x02 / 0x01   |
| 6   | CDC 通信接口      | 9   | 接口 1，0x02 / 0x02 / 0x01，1 个端点 |
| 7   | Header            | 5   | 子类型 0x00，`bcdCDC` 0x0110         |
| 8   | Call Management   | 5   | 子类型 0x01，数据接口 2              |
| 9   | ACM               | 4   | 子类型 0x02                          |
| 10  | Union             | 5   | 子类型 0x06，master 1 / slave 2      |
| 11  | CDC 中断 IN 端点  | 7   | `0x82`，中断，8 字节，间隔 0x10      |
| 12  | CDC 数据接口      | 9   | 接口 2，0x0A，2 个端点               |
| 13  | CDC 批量 IN 端点  | 7   | `0x81`，批量，64                     |
| 14  | CDC 批量 OUT 端点 | 7   | `0x01`，批量，64                     |

9 + 9 + 7 + 7 + 8 + 9 + 5 + 5 + 4 + 5 + 7 + 9 + 7 + 7 = 98。

IAD 的起始接口、Union 的 master/slave、Call Management 的数据接口号，全部从分配结果读出来，没有写死的常量。

## 4. 端点与接口的对应

接口描述符里的 `bNumEndpoints` 只说数量，没说是哪几个。归属靠位置 —— 端点描述符跟在所属接口描述符后面，下一个接口描述符一出现，前一组就结束，所以书写顺序就是归属关系：

```text
MSC: 接口 0（2 个端点） → 0x83 → 0x03

CDC: IAD
     → 接口 1（1 个端点）→ Header → CallMgm → ACM → Union → 0x82
     → 接口 2（2 个端点）→ 0x81 → 0x01
```

`bInterval` 也在这里定：中断端点 `0x10`，即 16 帧、约 16 ms；批量端点 0。宏的最后两个参数就是高速和全速各一份的间隔值。

## 5. 复合模式与注册顺序

前四步的前提是 `usbd_conf.h` 里四个宏都填 1：`USE_USBD_COMPOSITE`、`USBD_CMPSIT_ACTIVATE_MSC`、`USBD_CMPSIT_ACTIVATE_CDC`、`USBD_COMPOSITE_USE_IAD`。第一个一开，库里的 `USBD_MAX_SUPPORTED_CLASS` 从 1 变 4，`classId`、`tclasslist[]`、`pClassDataCmsit[]`、`pUserData[]` 这些按类编号的表才存在。

`USBD_MAX_NUM_INTERFACES` 不在这四个里，但要改：默认 1，改成 3。库只在分派类请求时拿请求里的接口号跟它比较，值小了不报错，超出的请求被 EP0 应答成 STALL —— 设备侧没有提示，主机只表现为请求失败。主机的 `SET_LINE_CODING`（设波特率）落在接口 1 上，填 1 就被挡住，串口照样收发，只是波特率设不上。

**库里的 classId 指向下一个空槽**：注册一个类，构建器先往那个槽写描述符再自增；而注册回调的两个函数（MSC 的存储回调、CDC 的接口回调）写的是**当前**槽。所以 `usb_device.c` 里分两轮：先注册完两个类，再用 `USBD_CMPSIT_SetClassID` 把 classId 拨到目标类上注册回调，查不到返回 `0xFF`。

写成一轮交错（注册类 → 拨号 → 注册回调 → 下一个类）会坏在中间：CDC 那次注册落在 MSC 的槽 0 上，构建器见槽已激活，跳过描述符生成，但类指针和端点数组已被覆盖，classId 和类计数照样自增。最后拨号找不到 CDC，返回 `0xFF` 卡在 `Error_Handler`，D+ 上拉拉不开，主机连设备都看不到。

运行时不用管：核心在每类的 `Init` 和 `DeInit` 之前会把 classId 设成那个类的编号。只有应用层没有这个上下文，主循环里 classId 可能停在任何一个类上，CDC 那边必须按类类型查自己的实例编号。

# 第三部分：代码改动

改动分三处，重新生成时命运不同。

`usbd_conf.c` 的分配器和 FIFO 配置、`usb_device.c` 的端点表写在 `USER CODE` 区，Generate 之后还在。

`usbd_conf.h` 的四个宏和 `USBD_MAX_NUM_INTERFACES`、`usbd_desc.c` 的 `USBD_PID_FS` 与字符串、`usb_device.c` 里 `MX_USB_DEVICE_Init` 的注册段在生成区，点一次 Generate 就退回原样。`.ioc` 里仍是单 MSC，CubeMX 生成不出复合设备，USB 这一层改完就不再点 Generate；以后调别的外设必须重新生成时，这三处按这一部分重做。

`usbd_cdc_if.{c,h}`、`BSP/USB_CDC` 和 `CM7/CMakeLists.txt` 里的挂载不在 CubeMX 的管理范围（`.ioc` 里没有 CDC 类），不受影响。

## 1. 工程与构建

工程从 `usb_msc` 复制而来，那里只有 MSC 一个类，缺两个目录，从 Cube 仓库拷到 `Middlewares/ST/STM32_USB_Device_Library/Class/` 下：

- `Class/CDC/{Inc,Src}` —— 虚拟串口类；
- `Class/CompositeBuilder/{Inc,Src}` —— 复合构建器，管接口编号和配置描述符拼装。

新加的源文件和头文件路径挂到 `CM7/CMakeLists.txt`：

```cmake
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    ../Middlewares/ST/STM32_USB_Device_Library/Class/CDC/Inc
    ../Middlewares/ST/STM32_USB_Device_Library/Class/CompositeBuilder/Inc
)

target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    ../Middlewares/ST/STM32_USB_Device_Library/Class/CDC/Src/usbd_cdc.c
    ../Middlewares/ST/STM32_USB_Device_Library/Class/CompositeBuilder/Src/usbd_composite_builder.c
)
```

同一组 include 还要再给 `USB_Device_Library` 这个目标加一遍，因为 `usbd_core.c`、`usbd_ctlreq.c` 编在那个目标里，复合模式下它们也要 include 构建器的头：

```cmake
target_include_directories(USB_Device_Library PUBLIC
    ../Middlewares/ST/STM32_USB_Device_Library/Class/CDC/Inc
    ../Middlewares/ST/STM32_USB_Device_Library/Class/CompositeBuilder/Inc
)
```

`usbd_cdc_if.c` 也加在这里而不是 `mx-generated.cmake`，CubeMX 重新生成时不会被覆盖。

## 2. 配置层

`usbd_conf.h` 是生成区。加四个宏，接口数从 1 改成 3：

```c
/*---------- -----------*/
/* 复合模式总开关，库用 #ifdef 判断，必须定义 */
#define USE_USBD_COMPOSITE     1U
/*---------- -----------*/
/* 允许复合构建器生成这两个类的描述符片段，它用 == 1U 判断 */
#define USBD_CMPSIT_ACTIVATE_MSC     1U
/*---------- -----------*/
#define USBD_CMPSIT_ACTIVATE_CDC     1U
/*---------- -----------*/
/* 给 CDC 生成 IAD。这个宏库里只被读、从没被定义，不写就没有 IAD */
#define USBD_COMPOSITE_USE_IAD     1U
/*---------- -----------*/
#define USBD_MAX_NUM_INTERFACES     3U
```

`usbd_conf.c` 改两处。第一处是类句柄的内存：CubeMX 生成的是单块静态缓冲，大小按 MSC 一个类算。

```c
void *USBD_static_malloc(uint32_t size)
{
  UNUSED(size);
  static uint32_t mem[(sizeof(USBD_MSC_BOT_HandleTypeDef)/4)+1];
  return mem;
}
```

两个类各要一块，会拿到同一个地址，MSC 的句柄（内含 512 字节 `bot_data`）盖掉 CDC 的。`USER CODE BEGIN 0` 区换成计数式线性分配，`USER CODE BEGIN Includes` 里补 `#include "usbd_cdc.h"`（`sizeof(USBD_CDC_HandleTypeDef)` 要用）：

```c
#define USBD_CLASS_MEM_SIZE (sizeof(USBD_CDC_HandleTypeDef) + sizeof(USBD_MSC_BOT_HandleTypeDef))

static uint8_t class_mem[USBD_CLASS_MEM_SIZE] __attribute__((aligned(4)));
static uint32_t class_mem_used;
static uint32_t class_mem_live;
```

```c
void *USBD_static_malloc(uint32_t size)
{
  uint32_t aligned = (size + 3U) & ~3U;

  if ((class_mem_used + aligned) > sizeof(class_mem))
  {
    /* 两个类的 Init 都会检查 NULL 并返回 USBD_EMEM */
    return NULL;
  }

  void *mem = (void *)&class_mem[class_mem_used];
  class_mem_used += aligned;
  class_mem_live++;

  return mem;
}

void USBD_static_free(void *p)
{
  UNUSED(p);

  if (class_mem_live > 0U)
  {
    class_mem_live--;
  }
  if (class_mem_live == 0U)
  {
    class_mem_used = 0U;
  }
}
```

指针在活块数归零时回到起点，靠的是 `USBD_free` 总是成对出现 —— `SET_CONFIGURATION` 和每次总线复位都会把所有活动类 `DeInit` 一遍。

第二处是发送 FIFO。原来只分了 EP0 和 EP1：

```c
  HAL_PCDEx_SetRxFiFo(&hpcd_USB_OTG_FS, 0x80);
  HAL_PCDEx_SetTxFiFo(&hpcd_USB_OTG_FS, 0, 0x40);
  HAL_PCDEx_SetTxFiFo(&hpcd_USB_OTG_FS, 1, 0x80);
```

那段是单 MSC 工程留下的，EP1 就是 MSC 的批量 IN。复合后 IN 端点变成 EP1（CDC 数据）、EP2（CDC 中断）、EP3（MSC 批量），`USER CODE BEGIN TxRx_Configuration` 区改成五行：

```c
  HAL_PCDEx_SetRxFiFo(&hpcd_USB_OTG_FS, 0x80);
  HAL_PCDEx_SetTxFiFo(&hpcd_USB_OTG_FS, 0, 0x40);   /* EP0：控制，最大 64 字节 */
  HAL_PCDEx_SetTxFiFo(&hpcd_USB_OTG_FS, 1, 0x40);   /* EP1 IN：CDC 数据，最大 64 字节 */
  HAL_PCDEx_SetTxFiFo(&hpcd_USB_OTG_FS, 2, 0x10);   /* EP2 IN：CDC 中断，最大 8 字节（16 字是下限）*/
  HAL_PCDEx_SetTxFiFo(&hpcd_USB_OTG_FS, 3, 0x20);   /* EP3 IN：MSC 批量，最大 64 字节 */
```

EP2、EP3 没有 FIFO 就不能发送 —— MSC 的 CSW 和 INQUIRY 都走 EP3，一个字节都出不去，而且往深度为 0 的 FIFO 写会踩到 RX FIFO，连枚举都保不住。总预算 128 + 64 + 64 + 16 + 32 = 304 字，按 320 字（OTG_FS 最小可能的 FIFO RAM）配，不顶格。

## 3. usb_device.c

includes 加三个头：`usbd_cdc.h`、`usbd_cdc_if.h`，以及 `usbd_composite_builder.h`（`USBD_CMPSIT_SetClassID` 的声明）。

`USER CODE BEGIN PV` 区加两张端点表，元素顺序就是构建器分配端点的顺序：

```c
/* 复合构建器只读这两张表，但形参是 uint8_t*，不能写 const。
   顺序即接口号：MSC 先注册占接口 0，CDC 后注册占接口 1 和 2。 */
static uint8_t msc_endpoints[] = {0x83U, 0x03U};         /* 批量IN, 批量OUT */
static uint8_t cdc_endpoints[] = {0x81U, 0x01U, 0x82U};  /* 批量IN, 批量OUT, 中断IN */
```

`MX_USB_DEVICE_Init` 里，CubeMX 生成的注册段是这样：

```c
  if (USBD_RegisterClass(&hUsbDeviceFS, &USBD_MSC) != USBD_OK)
  {
    Error_Handler();
  }
  if (USBD_MSC_RegisterStorage(&hUsbDeviceFS, &USBD_Storage_Interface_fops_FS) != USBD_OK)
  {
    Error_Handler();
  }
```

换成两次复合注册加两轮拨号：

```c
  if (USBD_RegisterClassComposite(&hUsbDeviceFS, &USBD_MSC, CLASS_TYPE_MSC, msc_endpoints) != USBD_OK)
  {
    Error_Handler();
  }
  if (USBD_RegisterClassComposite(&hUsbDeviceFS, &USBD_CDC, CLASS_TYPE_CDC, cdc_endpoints) != USBD_OK)
  {
    Error_Handler();
  }
  if (USBD_CMPSIT_SetClassID(&hUsbDeviceFS, CLASS_TYPE_MSC, 0U) == 0xFFU)
  {
    Error_Handler();
  }
  if (USBD_MSC_RegisterStorage(&hUsbDeviceFS, &USBD_Storage_Interface_fops_FS) != USBD_OK)
  {
    Error_Handler();
  }
  if (USBD_CMPSIT_SetClassID(&hUsbDeviceFS, CLASS_TYPE_CDC, 0U) == 0xFFU)
  {
    Error_Handler();
  }
  if (USBD_CDC_RegisterInterface(&hUsbDeviceFS, &USBD_Interface_fops_FS) != USBD_OK)
  {
    Error_Handler();
  }
```

交错写会让 CDC 落到 MSC 的槽上：构建器见槽已激活就跳过描述符生成，最后拨号在整个表里找不到这个类，返回 `0xFF` 卡在 `Error_Handler`，D+ 上拉不拉开。这一段在生成区，重新生成会覆盖。

## 4. usbd_desc.c

生成区，只改四个宏：

```c
#define USBD_PID_FS     22337
#define USBD_PRODUCT_STRING_FS     "STM32 Composite"
#define USBD_CONFIGURATION_STRING_FS     "Composite Config"
#define USBD_INTERFACE_STRING_FS     "Composite Interface"
```

PID 不能用 MSC 的 22314 或 VCP 的 22336：主机按 VID/PID 缓存"这个设备该用哪个驱动"，撞了会被认成原来那个设备。VID 1155 与仓库其他工程一致，不动。

## 5. usbd_cdc_if.c

`usbd_cdc_if.c` 和 `.h` 从 `vcp` 工程拷来，改六处。

一、includes 加两个头：

```c
#include "usbd_composite_builder.h"
#include "usb_cdc.h"
```

二、`CDC_Init_FS`。`USBD_CDC_SetTxBuffer` 在复合模式下多一个实例号参数，另外清一次环形缓冲。原来是：

```c
  USBD_CDC_SetTxBuffer(&hUsbDeviceFS, UserTxBufferFS, 0);
  USBD_CDC_SetRxBuffer(&hUsbDeviceFS, UserRxBufferFS);
```

改成：

```c
  /* 这个回调由核心在 CDC 的 Init 里调用，此时 classId 就是 CDC 自己 */
  USBD_CDC_SetTxBuffer(&hUsbDeviceFS, UserTxBufferFS, 0, (uint8_t)hUsbDeviceFS.classId);
  USBD_CDC_SetRxBuffer(&hUsbDeviceFS, UserRxBufferFS);
  usb_cdc_reset();
  receive_armed = true;
```

`receive_armed` 是文件里新加的 `static bool`：核心马上会在 `USBD_CDC_Init` 里自己挂一次 OUT 接收，标志跟着置真，否则 `CDC_ResumeReceive_FS` 会以为没挂，在传输途中重复挂一次。

三、`CDC_Transmit_FS` 的句柄取法。原来取 `pClassData`：

```c
  USBD_CDC_HandleTypeDef *hcdc = (USBD_CDC_HandleTypeDef*)hUsbDeviceFS.pClassData;
  if (hcdc->TxState != 0){
    return USBD_BUSY;
  }
  USBD_CDC_SetTxBuffer(&hUsbDeviceFS, Buf, Len);
  result = USBD_CDC_TransmitPacket(&hUsbDeviceFS);
```

这个字段每初始化一个类就被覆盖，复合后指向最后初始化的那个。改成按类查实例号，并把这个号传给库函数：

```c
  uint8_t class_id = (uint8_t)USBD_CMPSIT_GetClassID(&hUsbDeviceFS, CLASS_TYPE_CDC, 0);
  if (class_id == 0xFFU) {
    return USBD_FAIL;
  }
  USBD_CDC_HandleTypeDef *hcdc = (USBD_CDC_HandleTypeDef*)hUsbDeviceFS.pClassDataCmsit[class_id];
  if (hcdc->TxState != 0){
    return USBD_BUSY;
  }
  USBD_CDC_SetTxBuffer(&hUsbDeviceFS, Buf, Len, class_id);
  result = USBD_CDC_TransmitPacket(&hUsbDeviceFS, class_id);
```

主循环里调用时 `classId` 可能停在别的类上（比如刚处理完 MSC 的批量传输），所以要按类查，不能用 `hUsbDeviceFS.classId`。

四、新增 `CDC_IsTransmitReady_FS`：

```c
uint8_t CDC_IsTransmitReady_FS(void)
{
  uint8_t class_id = (uint8_t)USBD_CMPSIT_GetClassID(&hUsbDeviceFS, CLASS_TYPE_CDC, 0);
  if (class_id == 0xFFU) {
    return 0U;
  }

  USBD_CDC_HandleTypeDef *hcdc = (USBD_CDC_HandleTypeDef*)hUsbDeviceFS.pClassDataCmsit[class_id];
  if (hcdc == NULL || hcdc->TxState != 0U) {
    return 0U;
  }

  return 1U;
}
```

`USBD_CDC_SetTxBuffer` 只记指针不拷数据，数据要一直有效到发送完成，循环复用同一块缓冲区的地方必须先问它一次。

五、`CDC_Receive_FS`。原来是原样重挂：

```c
  USBD_CDC_SetRxBuffer(&hUsbDeviceFS, &Buf[0]);
  USBD_CDC_ReceivePacket(&hUsbDeviceFS);
```

改成先入队、再按空位决定挂不挂：

```c
  if (usb_cdc_receive(Buf, *Len) && (usb_cdc_get_free_space() >= CDC_DATA_FS_OUT_PACKET_SIZE)) {
    USBD_CDC_SetRxBuffer(&hUsbDeviceFS, &Buf[0]);
    USBD_CDC_ReceivePacket(&hUsbDeviceFS);
    receive_armed = true;
  } else {
    receive_armed = false;
  }
```

中断里只入队，回环的耗时不会压在 OTG_FS 中断里（MSC 也在同一个中断里跑）。放不下一整包时端点保持 NAK，主机这一包会重发。

六、新增 `CDC_ResumeReceive_FS`，主循环每轮调用：

```c
void CDC_ResumeReceive_FS(void)
{
  if (!receive_armed && (usb_cdc_get_free_space() >= CDC_DATA_FS_OUT_PACKET_SIZE)) {
    (void)USBD_LL_PrepareReceive(&hUsbDeviceFS, CDC_OUT_EP, UserRxBufferFS, CDC_DATA_FS_OUT_PACKET_SIZE);
    receive_armed = true;
  }
}
```

这里直接调 `USBD_LL_PrepareReceive` 而不调 `USBD_CDC_ReceivePacket`：后者从 `pdev->classId` 取自己的句柄和端点，而主循环里这个字段可能停在 MSC 上。

`usbd_cdc_if.h` 里补两个声明：`CDC_IsTransmitReady_FS` 和 `CDC_ResumeReceive_FS`。

## 6. BSP/USB_CDC

新驱动，`usb_cdc.c`、`usb_cdc.h` 加一个 `usb_cdc_example.c`（example 的内容整段在注释里，不进编译）。对外五个：

```c
bool usb_cdc_receive(const uint8_t* data, uint32_t length);   // USB 中断里调用
uint32_t usb_cdc_read(uint8_t* buffer, uint32_t length);      // 主循环里调用
uint32_t usb_cdc_get_free_space(void);
uint32_t usb_cdc_get_dropped_count(void);
void usb_cdc_reset(void);
```

128 字节环形缓冲，生产者是 USB 中断、消费者是主循环，单生产单消费，`volatile` 头尾指针就够，不需要临界区。反压交给 USB 本身：

```c
bool usb_cdc_receive(const uint8_t* data, uint32_t length)
{
    if (data == NULL || length == 0U) {
        return false;
    }

    // 装不下就一个字节都不放：调用方收到 false 后不重新挂接收，主机重发这一包。
    if (length > free_space()) {
        rx_dropped += length;
        return false;
    }

    for (uint32_t index = 0U; index < length; ++index) {
        rx_buffer[rx_head] = data[index];
        rx_head = (rx_head + 1U) % USB_CDC_RX_BUFFER_SIZE;
    }

    return true;
}
```

整包收下或者一个不放，所以 `rx_dropped` 正常恒为 0，它不为 0 说明挂接收时的空位判断漏了。源文件和 include 路径按第 1 节的方式加进 `CM7/CMakeLists.txt`。

## 7. cm7_main.cpp

SD 卡自检那一段不动，结尾的 `for (;;)` 换成日志、回环、挂接收三件事：

```cpp
    static uint8_t echo_buffer[64];
    static uint8_t log_buffer[48];
    uint32_t tick = 0U;
    uint32_t counter = 0U;

    for (;;) {
        const bool configured = (hUsbDeviceFS.dev_state == USBD_STATE_CONFIGURED);

        // 每秒一行日志，带出环形缓冲的退回计数。
        if (++tick >= 100U) {
            tick = 0U;
            if (configured && CDC_IsTransmitReady_FS()) {
                int length = std::snprintf((char*)log_buffer, sizeof(log_buffer), "CM7 USB CDC OK %lu drop=%lu\r\n",
                                           (unsigned long)counter, (unsigned long)usb_cdc_get_dropped_count());
                if (length > 0) {
                    CDC_Transmit_FS(log_buffer, (uint16_t)length);
                    ++counter;
                }
            }
        }

        // 主机发来的字节原样发回。
        if (configured && CDC_IsTransmitReady_FS()) {
            uint32_t count = usb_cdc_read(echo_buffer, sizeof(echo_buffer));
            if (count > 0U) {
                CDC_Transmit_FS(echo_buffer, (uint16_t)count);
            }
        }

        // 环形缓冲腾出空位后重新挂 OUT 接收。
        if (configured) {
            CDC_ResumeReceive_FS();
        }

        delay_ms(10);
    }
```

两块缓冲区分开是必须的：`CDC_Transmit_FS` 只把指针记下来、不拷贝数据，发送期间写同一块会把正在发的内容改掉。日志排在回环前面 —— 回环字节晚一轮只是延迟，日志这一秒错过了就真没了。

# 第四部分：实际现象

## 1. 上电与枚举

STM32H747 的 Cortex-M7，`USB_OTG_FS`（`PA11` 为 D−、`PA12` 为 D+，全速 12 Mbit/s），卡座上一张 FAT32 的 TF 卡，主机是 Ubuntu。日志走 USART1，115200 8N1；下载用工程根目录的 `./flash.sh cm7`。

完整工程：[STM32H747/usb_composite](https://github.com/fazhehy/STM32-HAL-Drivers/tree/main/STM32H747/usb_composite)。

复位后串口先打印自检结果：

```text
[=LOG=]: SD: block_number=62333952 block_size=512
[=LOG=]: FatFs test passed: 0:/TSTnnnnn.TXT
```

这两行过了，`f_mount(NULL, SDPath, 0)` 卸载 FatFs、`usb_msc_set_media_ready(true)` 开门闸，卡从此归主机，设备侧不再访问。

主机上核对描述符，看四处：`bNumInterfaces = 3`、`bDeviceClass = 0`、CDC 数据接口前面那条 IAD（`bDescriptorType` 为 `0x0B`）、配置里 5 个业务端点。这几项对上，说明复合结构立起来了。

## 2. 两个功能并存

主机上同时出现 `/dev/ttyACM1` 和 `/dev/sdc`。本机的 ST-LINK 探针占了 `ttyACM0` 和 `sdb`，认节点时别看错。

`cat /dev/ttyACM1` 每秒一行 `CM7 USB CDC OK n drop=0`，`n` 是日志行号，`drop` 是环形缓冲退回的字节数。

回环往里写几个字节会原样回来。测的时候用**同一个一直开着的句柄**先写后读 —— 每写一次就新开一个句柄，会让设备重新走一遍初始化，写进去的字节丢掉，看起来像固件卡死。终端里先把回显关掉（`stty -F /dev/ttyACM1 -echo`），否则看到的是终端自己回显的那份。

盘挂上后卷标是 `STM32`，根目录里是自检写下的 `TSTnnnnn.TXT`。

## 3. 已知限制

- MSC 在 `OTG_FS` 中断里同步读卡，期间 CDC 的收发被推迟：往盘里拷大文件时回环会变慢，不丢功能。
- `CDC_Transmit_FS` 忙时返回 `USBD_BUSY`，那一轮的回环字节不重发。
- 环形缓冲只有 128 字节，主机灌得比主循环快时会退回，日志行里的 `drop` 就是退回字节数。
- Windows 没实测；CDC + MSC 复合在 Win10+ 上应走 in-box 的 `usbser` 和 `USBSTOR`。
- `usbd_conf.h` 的宏、`usbd_desc.c` 的身份字符串、`usb_device.c` 的注册段在 CubeMX 生成区，点了 Generate 要按第三部分重做。

> 上面的现象来自一块板子、一台 Linux 主机上的几次插拔，长期插拔和大文件持续读写没测。编译通过与硬件验证是两回事。

