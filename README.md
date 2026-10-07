# Ternary-Bonsai-2-27B 本地部署（llama-prism / CUDA）

已在本机实测调优并跑通。模型：`Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf`（5.54 GiB，PTQ1_0 三值量化，1.75 bpw g128，基于 Qwen3.8 27B 的混合 SSM+注意力架构）。

服务当前监听 `http://127.0.0.1:8080`，提供 OpenAI 兼容 API。

---

## 0. 这一轮的记录与取舍（先看这一节）

### 0.1 优先级

调这台模型时按此排序取舍，冲突时高位胜出：

| 优先级 | 目标 | 现状 | 底线 |
|---|---|---|---|
| **1** | **上下文** | **96K（98304）**，在我们这里接近满速能跑到的长度 | 再长只能牺牲速度，而卸载带来的下降比较明显（见 0.4） |
| **2** | **质量** | 受 **1.75 bpw 量化**本身限制 | 工具使用实测良好（见 0.3） |
| **3** | **速度** | 短上下文 48.5 t/s；**96K 满上下文仍有 15.9 t/s** | 有保底，故最不重要 |

### 0.2 最终配置

```
-c 98304  -ctk q4_0 -ctv q4_0  -b 2048 -ub 512  -fa on  -ngl 99  -np 1  -kvu
-lm none  --jinja  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05
```

| 指标 | 实测 |
|---|---|
| 显存 | 7622 MiB 空载 / 7730 MiB 峰值（8 GB 卡，余约 170 MiB） |
| 短上下文中位解码 | **48.5 t/s**（4/4 确定性） |
| 93K 深上下文 | 预填充 464.9 t/s，解码 15.91 t/s |
| DSH 冷启动（22.3K 提示） | 预填充约 32.8 秒 |

几个影响较大的点：**K/V 用同一种量化类型**（混用会静默把注意力踢到 CPU，预填充 751→79.6 t/s）；`-b 4096` 在我们这里变慢；`-lm none` 三项都略有好处（省 5 GB 内存 / 加载更快 / 解码略升）。

### 0.3 质量：以"能不能用工具"为判据

四轮真实任务实测（不是基准分，是让它真去用）：

| 探测 | 结果 |
|---|---|
| 任务看板 `task_board_create` | ✅ 一次调对，卡片经看板账本独立核对确实建出 |
| 记忆宫殿 `engram_save` + `engram_search` | ✅ 存取闭环，带对 `kind`/`entities`，检索回原文一字不差 |
| Office `univer_*` 五步流程 | ✅ 新建→工作树→写 A1→读回→合并，`A1` 值独立核对正确 |
| 浏览器 `browser_*` | ✅ 2 次调用拿到正确标题（此前失败是 Electron 未安装，非模型问题） |

四轮下来工具都完成得不错，所以没有做更激进的裁剪（仅对本地模型丢掉 browser/engram/task_board 的长尾 25 个工具，省约 5,900 token）。

### 0.4 试过、这次没有采用的方向

完整清单（40+ 项，每项带实测依据）见 **[IDEAS.md](IDEAS.md)**。几个大项：

| 方向 | 我们测到的 |
|---|---|
| KVMem 分层 KV | 在我们这里略慢（48.50 vs 43.94 t/s），93K 深度会溢出 |
| 分层卸载 `-ngl` / `-ot` | 卸载后下降明显：ngl 56→9.7 t/s、48→6.3 t/s |
| `--swa-full` | 没有变化（该模型没有 SWA 层） |
| `--slot-save-path` | 恢复不了 prompt cache，且此 build 无 `--prompt-cache` |
| `--cache-reuse` | 没有变化（默认缓存已能同时命中多条序列） |
| `--no-repack` / `--no-op-offload` | 没有测出差异 |
| `GGML_CUDA_FORCE_MMQ` | 大约 +2.5%，在我们的噪声范围内 |
| **MTP 投机解码** | 公开数据里有明显加速，但要从 96K 里让出约 74K 上下文，与我们的优先级冲突，所以暂时没做 |

### 0.5 DSH 集成（三处本地改动）

1. **守护代理** `tools\llm-guard-proxy.mjs`（8090 → 8080）：只对 `ternary-bonsai-2-27b` 把输出预算抬到 ≥4096，规避"思考被截断→HTTP 500"；同时按模型裁工具。
2. **工具裁剪**：本地模型丢 browser/engram/task_board 长尾 25 个 → 提示词 31,625 → 22,339 token（−29%），**对话空间 64.4K → 73.7K**（对话空间 = 98304 − 提示 − 约 2.3K 的每轮固定开销）。
3. **压缩策略**：`compaction-basic` 默认 `headroomTokens: 65536` 对 96K 窗口会让预算归零 → 已在 profile 里给 `bonsai-local` 单独配 `headroomTokens: 8192`（阈值约 57K）。

> ⚠️ 两处与 DSH 行为有关的已知点：
> - **主动压缩**若不生效，**被动恢复仍然可用**：写到 96K 时报一次上下文超限，DSH 会自动裁剪+压缩+重试（代价是浪费一轮，约 30 秒重新预填充）。
> - 退出 DSH 会连带结束模型服务与代理（Job Object 机制），**回来跑一条命令即可** ↓

### 0.6 一条命令恢复全部

```powershell
D:\dsh\DSH\new\scripts\start-local-model.ps1
```

已在跑则跳过、没跑则拉起，最后报告端口与显存。**退出 DSH 或重启电脑后跑它一次即可。**

---

## 1. 快速开始

| 操作 | 双击 | 命令行 |
|---|---|---|
| 启动（默认档）+ 打开 Web UI | `start.cmd` | `.\scripts\start-server.ps1` |
| 停止 | `stop.cmd` | `.\scripts\stop-server.ps1` |
| 查看状态 / 显存 / 测速 | `status.cmd` | `.\scripts\status-server.ps1 -Bench` |
| 命令行对话 | — | `.\scripts\chat.ps1 "你的问题"` |
| 全量自检（6 项） | — | `.\scripts\verify-server.ps1` |

换个档位或端口：

```powershell
.\scripts\start-server.ps1 -Profile balanced -Port 8081  # 64k + 4 槽，需要并发批处理时
.\scripts\start-server.ps1 -Profile safe                 # 32k，给别的程序留显存
.\scripts\start-server.ps1 -Profile ram                  # 256k，但只有 10.5 t/s
.\scripts\start-server.ps1 -Force                        # 替换正在跑的服务
.\scripts\start-server.ps1 -Foreground                   # 前台运行，看完整日志
.\scripts\start-server.ps1 -Bind 0.0.0.0                 # 对局域网开放（手机用，见第 10 节）
```

> **在 DSH 里使用还要起守护代理**：`.\scripts\start-guard-proxy.ps1`（原因与细节见第 9 节）。
> 只想快速上手看 **QUICKSTART.md**；本文档是完整技术日志，第 3 节按时间顺序记录了每一轮实验。

### OpenAI 兼容接口

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ternary-bonsai-2-27b","messages":[{"role":"user","content":"你好"}]}'
```

- Base URL：`http://127.0.0.1:8080/v1`
- 模型 id：`ternary-bonsai-2-27b`
- 内置 Web UI：浏览器打开 `http://127.0.0.1:8080`（**要求客户端支持 gzip**；curl/PowerShell 直接 GET 会得到 415，这是正常的，浏览器没这个问题）
- 健康检查：`/health`；Prometheus 指标：启动时加 `-Metrics`

---

## 2. 本机最后采用的配置

硬件：**RTX 5060 Laptop 8GB**（驱动 617.14 / CUDA 13.4，算力 12.0）、Ryzen 9 8940HX 16C/32T、31GB 内存。
前提：桌面与其它程序已全部走核显，独显基线占用 **0 MiB**，可用 **7899 MiB**。

默认档实际命令行：

```
llama-server.exe -m models\Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf
  --host 127.0.0.1 --port 8080
  -ngl 99 -fa on -c 98304 -b 2048 -ub 512
  -np 1 -kvu
  --alias ternary-bonsai-2-27b --jinja
  -lm none
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05
  -ctk q4_0 -ctv q4_0
```

实测：峰值显存 **7730 MiB**，余量 **169 MiB**，预填充约 **640–740 t/s**（随提示长度而变），短上下文生成约 **48.5 t/s**、92K 深度约 **15.9 t/s**，服务常住内存 **886 MiB**，加载耗时约 **3.5–5 s**。

### 五个档位

| Profile | 上下文 | KV 精度 | 槽位 | 峰值显存 | 余量 | PP / TG | 适用 |
|---|---|---|---|---|---|---|---|
| `safe` | 32768 | q8_0 | 4 | 7466 MiB | 433 MiB | 751 / 43.4 t/s | 别的程序要用独显时 |
| `balanced` | 65536 | q4_0 | 4 | 7464 MiB | 435 MiB | 751 / 43.1 t/s | 需要并发批处理时 |
| **`default`** | **98304** | **q4_0** | **1** | **7730 MiB** | **169 MiB** | **约 740 / 48.5 t/s** | **满速下最长上下文（推荐）** |
| `q8` | 49152 | q8_0 | 1 | 7576 MiB | 323 MiB | 753 / 43.4 t/s | 保守选项（见 3.11） |
| `ram` | 262144 | q4_0 | 4 | 6240 MiB | 1659 MiB | 555 / 10.5 t/s | **原生满上下文**，KV 放内存 |

（以上均为**实测值**，不是估算。`default` 的峰值是 91,801-token 提示下的最坏值。）

### 为什么默认是 96K 而不是 64K

96K 是**本机满速下能给出的最长上下文**——实测在约 92K 深度（91,801-token 提示）仍有 **464.9 t/s 预填充 / 15.91 t/s 解码**，没有溢出。选它而不是更短的上下文，代价有两个，都值得先了解一下：

1. **余量从约 435 MiB 降到 169 MiB**（最坏情况）。余量薄时若别的程序占用独显，会静默掉进 WDDM 溢出区、吞吐塌到 1/10。本机已把独显完全让给模型（基线 0 MiB），所以可接受。
2. **只能 1 槽。** 每槽位约 150 MiB（混合 SSM 的每序列循环状态），96K 配 4 槽是 8166 MiB，装不下。

第 2 点的代价是**失去并发批处理**（3 并发下聚合吞吐 1.5×）。**但对 DSH 无影响**：实测一个 agent 轮次只发一次请求（子代理走 deepseek-flash），并发收益用不到。

反过来，上下文对 DSH 是刚需：**DSH 的 agent 提示本身约 31,625 token**（system prompt + 全部工具 schema）。64K 只剩约 32K 给对话，96K 则剩约 64K。

需要并发批处理时用 `balanced`（65536 / 4 槽 / 435 MiB 余量），两者可随时切换。

上限参考：96K 满速已验证；104K 曾测得满速但峰值已达 7882 MiB（余量约 17 MiB，深提示下有风险）；112K 开始退化；120K 溢出。所以 **96K 是留有余地的满速上限**。

DSH 侧的 `contextWindow` 已同步为 **98304**——若不改这一项，DSH 仍会按 64K 触发上下文压缩，白费掉扩出来的空间。

槽位数是**按档位设定的**，因为每个槽位要额外占约 150 MiB 显存（这个混合 SSM 模型每个序列都要保留自己的循环状态）。`default` 和 `q8` 的上下文本身已占满显卡，所以只能 1 槽；`safe`/`balanced`/`ram` 用 4 槽。用 `-Slots` 覆盖时会给出余量预警。详见 3.17。

**为什么 KV 用 q4_0 而不是 q8_0**：实测两者精度无差异（见 3.7），而 q4_0 每 1024 token 只占约 18 MiB、q8_0 约 35 MiB——同样的显存能换到接近两倍的上下文。`q8` 档仅作为"宁可用 q8_0"的保守选择保留。

采样参数按模型卡与 GGUF 内嵌元数据设置：`temp 1.0 / top-p 0.95 / top-k 20 / min-p 0.05`。

### 显存与上下文的经济账

实测每 1024 tokens 上下文的 KV 开销：

| KV 类型 | 每 1024 tokens | 可用上限（~7730 MiB 天花板） |
|---|---|---|
| f16 | ~64 MiB | 约 32–40k |
| q8_0 | ~35 MiB | 48k |
| q4_0 | ~18 MiB | 96k |

权重固定占 5673 MiB，计算缓冲约 100 MiB，可用总量 7899 MiB。所以**任何额外显存开销都会直接吃掉上下文**：想加别的组件（例如视觉投影），先按上表换算要牺牲多少上下文。

模型原生最大上下文是 262144 tokens；受 8GB 显存限制，本机满速上限是 96k（`default`），要跑满原生 256k 只能走 `ram` 档，代价是降到 10.5 t/s。


---

## 3. 实测数据与调参依据

**怎么读这一节**：以下 17 个小节**按时间顺序**记录了调参过程中的每一轮实验。同一个问题（尤其是上下文上限）会被反复测很多次，因为配置在演进——**后面的小节会修正或细化前面的判断**。所以别把早期小节的数字当成现值。

**要当下结论，看第 1、2 节**。这一节是"为什么是这个值"的证据链。

### 参数速查（本机最终值）

| 问题 | 目前的取值 | 依据小节 |
|---|---|---|
| 满速上下文上限 | **98304**（96k） | 3.9、3.16 |
| 显存天花板 | 峰值 **7730 MiB**，超出即溢出（吞吐塌到 1/10） | 3.1、3.16 |
| KV 精度选哪个 | **q4_0**（与 q8_0 无差异，每 1024 token 省一半显存） | 3.7、3.11 |
| 官方偏置要不要用 | **这次没用**（反而差 0.074%） | 3.11 |
| K/V 能否混用类型 | **不能**（掉出快速内核，PP 掉 9 倍） | 3.3、3.12 |
| 批大小 | `-b 2048 -ub 512` | 3.2、3.14 |
| 加载模式 | **`-lm none`**（省 5.1 GB 内存且更快） | 3.13、3.14 |
| 槽位 | 默认 1 个（并发收益对本场景无用），需要时用 `balanced` | 3.17 |
| 思考预算 | **这次没设**；给客户端足够 `max_tokens` | 3.15、第 9 节 |

其余小节是过程记录，含失败的尝试与排查方法，留作参考。

以下全部为本机实测（2000-token 提示，`-ub 512`）。

### 3.1 上下文 × KV 精度 —— 显存边界在 7730 MiB

| 上下文 | KV | 峰值显存 | PP t/s | TG t/s | 结果 |
|---|---|---|---|---|---|
| 32768 | f16 | 7858 | 722.2 | 38.8 | 可用 |
| 32768 | q8_0 | 7026 | 751.4 | 43.4 | 可用 |
| **49152** | **q8_0** | **7586** | **753.2** | **43.4** | **← 默认** |
| 65536 | q8_0 | 7888 | **36.6** | **7.1** | 溢出 |
| 65536 | q4_0 | 7122 | 751.3 | 43.1 | 可用 |
| **98304** | **q4_0** | **7730** | **约 740** | **48.5** | **← default 档（现行）** |
| 131072 | q4_0 | 7888 | **36.1** | **6.2** | 溢出 |

**最重要的一条**：溢出时服务照样"启动成功"，但吞吐塌到 1/10。只有实测吞吐才能识破——`64k-q8` 峰值 7888 MiB，PP 从 753 掉到 36.6、TG 从 43.4 掉到 7.1。这就是为什么"能跑起来"不能作为部署成功的判据。

### 3.2 为什么 `-ub 512` 是最优点

| ubatch | 峰值显存 | PP t/s | TG t/s |
|---|---|---|---|
| **512** | 7595 | 751.3 | 43.5 |
| 1024 | 7776 | 748.8 | 40.1 |
| 2048 | 7880 | 311.1 | 0 | 

调大不但没提速，1024 反而略慢，2048 直接溢出。

### 3.3 K 与 V 用同一种量化类型更稳

固定 `-c 49152`，只改 K/V 类型（对照实验，其它参数完全一致）：

| 配置 | 峰值显存 | PP t/s | TG t/s | 结果 |
|---|---|---|---|---|
| K=q8_0 / V=q8_0 | 7586 | 735.2 | 43.6 | 满速 |
| **K=q8_0 / V=q4_0** | **7148** | **79.6** | **18.1** | 混合即掉速 |
| K=q4_0 / V=q4_0 | 6818 | 750.2 | 43.3 | 满速 |
| K=f16 / V=q8_0 | 7878 | 35.1 | 5.2 | 显存溢出 |

关键在第二行：混合精度**用得更少显存**（7148 < 7586），却慢了约 9 倍。显存压力无法解释，只能是 **K 与 V 类型不同时掉出 Flash-Attention 快速内核、走慢速回退路径**。第三行同类型 q4_0 满速，也排除了"低精度本身慢"。

注意区分两种失效模式：混合精度是**显存够但变慢**（7148 MiB），f16/q8 那行才是**显存溢出**（7878 MiB）。两者的修法完全不同——前者改回同类型，后者降上下文。

**这一轮的观察：K 和 V 各自可以是 `q8_0` 或 `q4_0`，两边保持一致时表现更好。** LM Studio 的 KV 档位天然是单一类型，所以不存在这个问题。

### 3.4 KV 放内存（`-nkvo`）—— 能跑，但别当默认

| 上下文 | KV | 峰值显存 | 峰值内存 | PP t/s | TG t/s |
|---|---|---|---|---|---|
| 32768 | f16 | 5922 | 8.6 GB | 589.8 | 10.1 |
| 65536 | f16 | 6030 | 10.7 GB | 585.6 | 10.8 |
| 131072 | f16 | 6350 | 14.9 GB | 563.5 | 9.8 |
| 196608 | f16 | 6670 | 19.0 GB | 562.6 | 10.1 |

196k 上下文 + 完整 f16 KV 确实能跑，但 **TG 从 43 掉到 10 t/s**。机理清楚：每个解码步都要把整个 KV 经 PCIe 拉一遍（32k 上下文时约 1 GiB/步 ≈ 0.1 s）。所以 KV 需要留在显存，`ram` 档只适合不要求交互速度的批处理。

### 3.5 深上下文真实速度（42,501-token 提示）

| 配置 | 峰值显存 | PP t/s | TG t/s |
|---|---|---|---|
| 49152 / q8_0 | 7586 | 579.3 | 25.7 |
| 98304 / q4_0 | 7730 | 580.1 | 26.1 |

即使塞满 42k 上下文，仍有 26 t/s 生成速度，完全可用。

### 3.6 与你在 LM Studio 上的数字对照

你给的 LM Studio 参考值 Q8_0: 30976 / Q4_0: 52736，方向与我的实测一致，只是更保守——LM Studio 要为自己 GUI 预留显存。我的边界是：q8_0 满速到 48k，q4_0 满速到 96k。

### 3.7 KV 精度实测：q4_0 与 q8_0 无差异

之前只测了速度和显存，没测精度。为此专门做了质量探针（`scripts\calibration\quality-probe.ps1`，三档各跑一遍，temperature=0）：

- **短任务 12 项**：多步算术、三段论、字符串反转、字符计数、质数、速度换算、精确格式输出、星期推算、事实题。
- **长上下文检索**：在填充文本中埋入唯一密钥 `TANGO-7741`，要求在指定深度取回（这是 KV 量化损害最先暴露的地方）。

| 配置 | 短任务 | 检索 |
|---|---|---|
| 49152 / q8_0 | 12/12 | 4/4 |
| 49152 / q4_0 | 12/12 | 4/4 |
| 98304 / q4_0 | 12/12 | 4/4（最深处 **94,266 tokens**） |

**q4_0 没有付出任何可测的精度代价。** 这有两方面支持：一是上面的实测全等；二是架构原因——本模型 `full_attention_interval: 4`，64 层里只有 **16 层**有逐 token 的 KV 缓存，其余 48 层是固定大小的 SSM 循环状态、根本不被量化。所以 KV 量化只影响 25% 的层。

因为这个观察，"上下文长度 vs 精度"在本机上**不是真实取舍**——真正的取舍是**上下文长度 vs 显存余量**（见第 2 节档位表）。这也是默认档从 48k q8_0 改为 64k q4_0 的依据。

> 探针的局限（如实说明）：短任务 12 项样本偏小，检索类任务对 KV 量化相对不敏感。若要更严格，可用 `llama-perplexity.exe` 对比两档的困惑度——那才是逐 token 级的判据。目前的观察足够支撑"日常用 q4_0"，但若你对精度极度保守，`q8` 档仍在。

### 3.8 深上下文满载验证

用 `deep-load-check.ps1` 推入近满载提示，实测峰值与深上下文解码速度：

| 档位 | 提示 tokens | 峰值显存 | 预填充 | 深上下文解码 |
|---|---|---|---|---|
| 65536 / q4_0 | 59501 | 7122 MiB | 524.3 t/s | 22.2 t/s |
| 98304 / q4_0 | 91801 | 7730 MiB | 458.5 t/s | 18.1 t/s |

两个要点：**峰值显存不随提示长度增长**（KV 在加载时预分配，计算缓冲受 `-ub 512` 约束），所以余量是可预测的；**解码速度随上下文深度下降**（43 → 22 → 18 t/s），这是 KV 读取量增长的必然结果，不是故障。

### 3.9 上下文上限在哪（Agent 场景的关键数字）

`suites.ps1 -Suite ceiling`，VRAM 内的 q4_0：

| 上下文 | 预填充 | 解码 | 峰值显存 | 结论 |
|---|---|---|---|---|
| 98304 | 753 | 43.5 | 7730 | ✅ 满速 |
| 106496 | 751 | 43.4 | 7882 | ✅ 满速（但要吃掉全部余量） |
| 114688 | 193.7 | 35.2 | — | ⚠️ 开始退化 |
| 122880 | 53 | 7.3 | — | 溢出 |

**在我们这里，满速能跑到的范围大约是 96k–106k，到 262144 不太现实。** 权重就占 5673 MiB，8GB 卡装不下原生上下文。所以 `default`（98304）是满速下能给出的最长上下文。

### 3.10 要满 262144 只有一条路：KV 放内存（`-nkvo`）

这里先修正我先前的一个说法。我曾说 `-nkvo` 慢是因为"每个解码步都要把整个 KV 经 PCIe 拉一遍，随上下文线性变慢"——**后来发现这个解释站不住**，算一下就矛盾：192k 上下文的 f16 KV 有 12 GiB，若每步真读 12 GiB，那是 120 GB/s，PCIe 做不到，可实测仍有 10 t/s。

用极小上下文做了判别实验：

| 配置 | 上下文 | KV 大小 | 预填充 | 解码 | 峰值显存 |
|---|---|---|---|---|---|
| VRAM f16（对照） | 4096 | 256 MiB | 753.1 | **44.5** | 6160 |
| `-nkvo` f16 | 4096 | 256 MiB | 603.8 | **10.6** | 5770 |
| `-nkvo` f16 | 16384 | 1 GiB | 601.1 | **9.4** | 5812 |
| `-nkvo` f16 | 65536 | 4 GiB | 591.2 | **9.8** | 6030 |
| `-nkvo` q4_0 | **262144** | 4.6 GiB | 554.7 | **10.5** | 6256 |

4k 上下文（KV 只有 256 MiB，带宽根本不是瓶颈）时 `-nkvo` 也只有 10.6 t/s，而 262144 时是 10.5 t/s——**惩罚是固定的约 4 倍，与上下文长度无关**。所以这是路径代价，不是带宽代价。修正后的结论：

**`ram` 档能跑满原生 262144 上下文，只占 6256 MiB 显存 + 11.4 GB 内存，解码恒定 10.5 t/s，而 prefill 几乎不打折（554.7 vs 753）。**

顺带排除了另一条看起来可行的路——把部分层留在 CPU 换上下文：

| 配置 | 上下文 | 预填充 | 解码 |
|---|---|---|---|
| `-ngl 56`（8 层在 CPU） | 131072 | 539.1 | **9.7** |
| `-ngl 48`（16 层在 CPU） | 163840 | 468.6 | **6.3** |

代价和 `-nkvo` 一样是掉到 ~10 t/s，但 `-nkvo` 给的是完整 262144。**这条路能拿到的上下文被 `-nkvo` 覆盖，所以这次没有采用。**

### 3.11 两个精度的真实能力差距（配对分析）

判据一：困惑度。`llama-perplexity.exe`，moby.txt，ctx 4096 × 16 chunks = 65536 tokens，只变 KV 类型。

⚠️ **这里有个统计陷阱。** 工具报的是 `PPL = 16.1217 +/- 0.25148`，标准误 ±0.25，而我们关心的差异只有 0.02 量级——**直接看这个误差会误判为"全部无差别"**。但那个 ± 是 **chunk 之间的难度离散度**，而所有配置跑的是**同一批 chunk**，且同一配置重复运行得到**逐 chunk 完全一致**的结果（`ppl-q4_0.log` 与 `ppl-ctl-roton.log` 逐位相同，可验证确定性）。所以正确做法是**逐 chunk 配对比较**：

| 对比 | 平均 PPL 差 | 相对 | 更差的 chunk | 判定 |
|---|---|---|---|---|
| q4_0 vs f16 | **+0.0228** | **+0.14%** | **16 / 16** | 高度显著（p ≈ 1.5e-5） |
| q8_0 vs f16 | +0.0022 | +0.014% | 14 / 16 | 显著 |
| 关旋转 vs 开旋转 | −0.0135 | −0.084% | 11 / 16 | 不显著 |

所以**精度差距是真实存在的，量级是 q4_0 约 +0.14%、q8_0 约 +0.014%**（q8_0 比 q4_0 靠近无损约 10 倍）。但 0.14% 的困惑度差异远低于日常可感知的程度——参考坐标：模型权重本身已被压到 1.75 bit/权重。

判据二：任务级。用"硬能力测试"（`hard-battery.ps1`）在**两者都满速的 49152 上下文**下对比（q8_0 在 64k 会溢出，会导致不公平比较——见 3.17）：

| 任务（40k tokens 深度） | q8_0 | q4_0 |
|---|---|---|
| multi（三处密钥全部取回） | PASS | PASS |
| **count（统计全文出现 7 次的词）** | **FAIL → 答 4** | **FAIL → 答 4** |
| distr（8 个相似密钥中挑指定项目） | PASS | PASS |
| combine（合并相距 80% 的两条信息） | PASS | PASS |
| numeric（两个相距很远的数求和） | PASS | PASS |
| **合计** | **4 / 5** | **4 / 5** |

**两个精度在任务级上没能区分出来**，唯一失败的一次还是同一道题、同一个错法。所以那个 `count` 失败更像是**模型在 40k 深度附近的能力边界**，与精度关系不大——两个精度的结果一样。这对 Agent 设计有实际意义：**不要让这个模型做需要全上下文聚合的任务（如"统计全文里 X 出现几次"），它在 40k 深度下不可靠，且换精度救不了。**

判据三：官方偏置。PrismML 为 q4_0 提供 `llama-kv-mean-center` 均值中心化偏置，我按官方流程生成并测了两种校准设置：

| 配置 | 困惑度 | 配对结论 |
|---|---|---|
| q4_0（开旋转，无偏置） | 16.1217 | — |
| q4_0（关旋转，无偏置） | 16.1170 | 关旋转略好，但不显著 |
| q4_0 + 偏置（校准旋转不匹配） | 16.1220 | — |
| q4_0 + 偏置（校准旋转匹配） | **16.1220** | 与不匹配时**逐 chunk 完全相同** |

两种校准设置产出**逐位相同**的偏置，且都**让 PPL 变差 +0.074%（15/16 chunk 更差）**。

**这一轮的观察：这个偏置在我们这里没有帮助。** 我起初推测是"关旋转的损失抵消了偏置收益"，配对数据与这个推测不符——关旋转本身影响不大，加了偏置反而更差。原始 q4_0 的 +0.14% 差距本来就已可忽略，加偏置反而更差。

> 复现脚本：`quality-probe.ps1`（任务级）、`hard-battery.ps1`（判别性任务）、`ppl-compare.ps1`（困惑度）、`kv-bias.ps1` 与 `kv-bias-rotmatched.ps1`（偏置）。结论仅针对当前模型与 build 10754；换模型或换语料应重跑。

### 3.12 混合 K/V 掉速的根因（有官方出处）

第 3.3 节测到的混合精度掉速，根因在 upstream llama.cpp，不在本机配置：

- `ggml/src/ggml-cuda/fattn.cu` 里 `ggml_cuda_fattn_kv_type_supported()` 有编译期开关 `GGML_CUDA_FA_ALL_QUANTS`，**默认 OFF**；
- 默认路径还有第二道检查，**要求 `K->type == V->type`**；
- 任一触发即返回 `BEST_FATTN_KERNEL_NONE`，调度器**把整个 `FLASH_ATTN_EXT` 放到 CPU 后端**；
- 全程**无警告、无日志**——这就是它难以诊断的原因。

诊断特征：**prefill 塌陷**，而 decode 受影响较小。我的实测（PP 735→79.6，TG 43.6→18.1）正是此特征。参考：[llama.cpp PR #27150](https://github.com/ggml-org/llama.cpp/pull/27150)、以及一篇对该现象的分析（[KingsClaw](https://blog.kingsclaw.org/llama-cpp-quantized-kv-cache-silent-cpu-fallback-prefill-fix/)）。

修复方式是重新编译加 `-DGGML_CUDA_FA_ALL_QUANTS=ON`，但**本部署不需要**——K/V 同类型时本来就满速。

### 3.13 内存去哪了：为什么"看起来吃了 15 G"，以及怎么让它降到 886 MB

先给结论：**系统内存没有被吃爆。** 稳态实测（32 GB 机器）：

```
Available      21,464 MiB
Committed      25,198 / 50,398 MiB
Pagefile 已用     401 MiB（峰值 6,557 MiB）   ← 几乎没动，说明没发生交换
```

排查过程与真实构成：

| 状态 | 系统已用 | 系统可用 | 服务 WorkingSet | 服务 Private |
|---|---|---|---|---|
| 完全没有服务 | 9,329 MiB | 22,637 MiB | 0 | 0 |
| mmap 加载后 | 15,268 | 16,698 | 5,998 | 8,300 |
| `-lm none` 加载后 | — | **21,464** | **886** | 8,576 |

三个容易误判的点：

1. **"吃了 15 G"其实是"系统已用 15 G"**，其中 9.3 GB 在服务启动前就被系统/其它程序占着，服务只贡献约 5.9 GB——而那 5.9 GB 是 5.54 GiB 模型文件被 **mmap 钉在页缓存**里的部分，属**可回收**，不是泄漏。
2. **任务管理器里的 Private（8.3 GB）会误导人**：那是"已提交但未驻留"的地址空间承诺，不占物理内存。真正占用物理内存看 **WorkingSet**。
3. **提示缓存不是元凶。** 我一开始怀疑 `-cram, --cache-ram`（默认 8192 MiB）——用 `-cram 1024` 实测占用**逐项完全一致**（Private 都是 8300/8901 MiB），假设被自己的数据否定。

**真正的优化**：模型已全部在显存里，主机端那份拷贝只在加载时需要，mmap 却把它一直钉住。改用 `-lm none`（即 `--no-mmap` 的现代写法）：

| 指标 | mmap | `-lm none` |
|---|---|---|
| 服务 WorkingSet | 6,015 MiB | **886 MiB** |
| 系统可用 | 16,493 MiB | **21,464 MiB** |
| 加载耗时 | 6.6 s | **3.5–5.0 s** |
| 解码 | 43.2 t/s | **45.6 t/s** |

**省 5.1 GB 常住内存，同时加载更快、解码更快**——三项全赢，已设为默认。

> 说明：曾经量到"只剩 2,476 MiB 可用"，那是我多个基准任务叠加运行时的瞬时状态（测试服务 + 脚本 + 页缓存），不是部署的稳态表现。

#### 曾观察到内存占用超过 95% —— 原因已定位

调参期间确实一度冲到 95% 以上，原因是**我自己的基准实验**，不是日常配置：

- `-nkvo`（KV 放内存）配 **f16 @ 196608 上下文**时，服务峰值内存实测 **19,037 MiB**；
- 加上系统基线 9.3 GB ⇒ 约 28 GB（88%）；
- 再叠加并行的测试进程与页缓存 ⇒ 突破 95%。

日常档位离这个量级很远（`default` 服务常住内存 886 MiB）。比较值得留意的是 `ram` 档，它也因 `-lm none` 而大幅改善：

| `ram` 档（262144，KV 放内存） | 服务内存 | 系统占用率 |
|---|---|---|
| mmap | 11,275 MiB | 66.5% |
| `-lm none` | **6,146 MiB** | **50.5%** |

并已加**启动保护**：`ram` 档启动前会估算 KV 所需内存（按 KV 精度换算 18/34/64 KiB per token），若超过当前可用内存的 70% 则**拒绝启动**并给出替代建议，而不是让机器去交换：

```
RAM note  : KV lives in system memory (~16384 MiB for 262144 tokens)
            available right now: 21127 MiB
refusing to start: KV needs ~16384 MiB of RAM but only 21127 MiB is available.
Lower -Ctx (e.g. -Ctx 131072), or use -Profile default which keeps KV in VRAM.
```

### 3.14 速度：还试过什么、哪些没用

| 尝试 | 结果 | 结论 |
|---|---|---|
| `-lm none`（替代 mmap） | 加载 6.6→3.5s，解码 43.2→45.6 t/s，省 5.1 GB 内存 | ✅ **已采用** |
| `-b 4096`（逻辑批 2048→4096） | 解码 43.2 → **38.8** t/s，预填充无改善 | 变慢了，所以维持 2048 |
| `-ub 2048` | 显存 7880 MiB 溢出，PP 311 t/s | 未采用（3.2 节） |
| `-ub 1024` | 略慢于 512 | 维持 512 |
| 投机解码（DSpark drafter） | Bonsai 2 官方**尚未发布**配套 drafter | ⛔ 不可用 |
| 部分层 CPU 卸载换显存 | 解码掉到 9.7 t/s | 被 `-nkvo` 覆盖 |

**还有一个对 Agent 极有价值、不需要改任何参数的特性：提示缓存。** 实测第二轮对话在 30k tokens 的相同前缀上，只处理了 **8 个 token**（`pp_n=8`）——也就是多轮对话、工具调用回环里那些重复的长前缀**几乎免费**。前提是客户端回传时保持前缀稳定，并带上 `cache_prompt: true`（llama.cpp 默认对相同前缀复用 KV）。

### 3.15 思考长度：一个会导致"答案为空"的真实故障模式

这是实测踩到的，值得单独记：**客户端 `max_tokens` 太小时，模型会把整个预算用在推理上，最终返回空答案。**

第一轮语言测试（`max_tokens=2048`，无思考上限）六种条件里有四种 `AnsLen = 0`——思考产出 3970–4964 字符，`completion_tokens` 全部顶到 2048，可见答案一个字都没有。同一提示把预算提到 4096 就正常出答案。

> ⚠️ **这一条我改了两次，两次都不够准确，这里给出证据分级。**
>
> - **已实测**：客户端 `max_tokens` 太小 → 流式输出被截断 → 畸形文本让 fork 的解析器解析失败 → **HTTP 500**（第 9 节有两个触发条件）。
> - **已实测**：`--reasoning-budget 1024` 当时**修好了**空答案问题——上面那段"加上后所有条件都正常产出答案"就是证据（原文见本节历史版本）。
> - **未实测（我上一轮当成定论写了出来，后来发现不准确）**：服务端预算是否也会引发 500。
>   本 fork 有 `--reasoning-budget-message`（预算耗尽时**注入收尾消息**，默认 none），
>   也就是说服务端可以**干净地结束思考**，与客户端硬截断并非同一机制；
>   KVMem 那条 fork 线的启动器甚至**默认就设 `-ReasoningBudget 4096`**。
>
> 所以**这是待测项，不是"不要用"**。见 `IDEAS.md` 的 E1——这个测试很便宜，而且若成立，
> 它能同时缓解"思考吃光预算"与"绕圈浪费时间"两个问题。
>
> **正确做法：给客户端足够的 `max_tokens`。** 约 300 词的任务需要约 4000 token，按此比例放大即可。DSH 给的是 32768，所以那边不需要任何处理。

### 3.16 做精度对比时需要在"两者都满速"的上下文下比较

这一节记录一个我在调参中真实踩到的坑，因为它会直接毁掉对比结论。

我第一次跑硬能力测试时把上下文设成 65536，对 q4_0 没问题，但 **q8_0 在 65536 会溢出**（3.1 节已测：峰值 7888 MiB）。结果那组 q8_0 是在共享内存颠簸的状态下跑的，两个精度的比较被污染，还出现了假失败。

发现它的方式正是你在旁边看到的那句"GPU 好像 20 秒才动一下"——顺着查 GPU 才发现余量只剩 **12 MiB**。对照修正前后同一条命令：

| | 错误条件 | 修正条件 |
|---|---|---|
| 上下文 / KV | 65536 / q8_0 | 49152 / q8_0 |
| 显存占用 | 7888 MiB | 7580 MiB |
| 余量 | **12 MiB** | **320 MiB** |
| GPU 利用率 100% 的含义 | WDDM 换页空转 | 真实计算 |

**规则：跨配置对比前，先确认每个配置的余量都在 200 MiB 以上。** 否则你比的不是精度，而是谁先溢出。`default` 档（98304 / q4_0）余量 169 MiB，恰好低于这条线——它作为单独运行的档位可用（已在 92k 提示下验证满速），但**不适合作为对比实验的一侧**。

### 3.17 并发：Agent 场景的槽位与 unified KV

测试方法：3 个请求同时发出（HttpClient 异步任务），各约 2000-token 提示 + 256 token 生成，`ignore_eos` 强制生成满额。**注意**：.NET 默认 `ServicePointManager.DefaultConnectionLimit = 2`，会悄悄把并发串行化、让结果看起来"没有批处理"——第一轮就踩了这个坑，抬高后才得到真实数据。

| 配置 | 槽位 | **每槽上下文** | 顺序 3 请求 | 并发 3 请求 | **聚合加速比** | 30k 提示 |
|---|---|---|---|---|---|---|
| `-np 1` | 1 | 65536 | 27.2 s | 29.2 s | **0.93**（无批处理） | ✅ 接受 |
| `-np 3` 非 unified | 3 | **22016** | 27.3 s | 17.6 s | 1.55 | **HTTP 400** |
| `-np 3 -kvu` | 3 | 65536 | 26.7 s | 19.1 s | 1.40 | ✅ 接受 |
| **`-np 4 -kvu`（默认）** | 4 | **65536** | 27.6 s | 18.2 s | **1.51** | ✅ 接受 |

三个结论：

1. **`-np 1` 会把并发请求串行化**——加速比 0.93，3 个并发和 3 个顺序一样慢。Agent 并发发请求时这是在浪费。
2. **连续批处理真实有效**，多槽位带来 **1.4–1.55× 聚合吞吐**。
3. **需要配 `-kvu`（unified KV）**。不带它时 `-c` 会被**按槽位数等分**：`-np 3` 配 `-c 65536` 每槽只有 **22016** token，30k 提示直接被 400 拒绝。这是最容易踩的坑——上下文会随并发数悄悄缩水。unified KV 让所有槽共享一个池，单个会话仍可用满 `-c`。

**代价在显存，不在速度。** 这个模型是 SSM/注意力混合架构，**每个序列都要保留自己的循环状态**，所以每槽位额外约 **150 MiB 显存**（`-c 65536` 实测）：

| 槽位数 | 显存 | 余量 |
|---|---|---|
| 1 | 7010 MiB | 889 MiB |
| 2 | 7160 MiB | 739 MiB |
| **4** | **7460 MiB** | **439 MiB** |
| 8 | 7824 MiB | **75 MiB** ⚠️ 溢出边缘 |

因此槽位默认值是**按档位**给的（`safe`/`balanced`/`ram` 用 4，`default`/`q8` 用 1），用 `-Slots` 覆盖时会打印估算余量并在低于 250 MiB 时预警。

批处理的固有代价：单请求延迟上升（约 9.2 s → 18.2 s），换来总墙钟时间缩短。Agent 关注总时长，所以划算；纯单用户对话用 `-Slots 1` 可拿回最低延迟。

---

## 4. 关于 LM Studio：需要换上 fork 的后端，但 LM Studio 本身可以用

**必需条件**：`PTQ1_0` / `PQ2_0` 属于 Ternary Bonsai 2，需要由 PrismML fork（`prism-b10658+`）运行。prism-ml 官方文档明确：

- stock llama.cpp 会**直接拒绝**这些文件。
- 遗留的 `Q2_0` 文件在 stock 构建上**会静默加载并输出乱码**（比拒绝更危险）。
- 本工作区的 `llama-prism-b10754` 满足要求（b10754 > b10658）。

**不过"需要 fork 后端"不等于"LM Studio 用不了"。** LM Studio 允许把自定义后端二进制放进它的运行时目录，本机的这套 fork 正是这样在 LM Studio 上跑通的。换后端之后，LM Studio 用的就是 fork，该模型能正常工作。

真正的差异只在**可用上下文**，而且原因清楚：

| | KV=Q8_0 | KV=Q4_0 |
|---|---|---|
| LM Studio 报告上限 | 30976 | 52736 |
| 本次裸服务实测满速上限 | 49152 | 98304 |

LM Studio 要为自己 GUI 预留显存，报价天然保守；本次部署是在独显基线占用 0 MiB、GPU 完全让给模型的条件下实测的。两者都成立，不是谁对谁错。

验证 fork 是否真的在加载这个模型（输出里应出现 `ftype : PTQ1_0 - 1.75 bpw ternary (group 128)`）：

```powershell
.\runtime\llama-prism\llama-cli.exe -m .\models\Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf -st -rea off -p "test"
```

GGUF 里带 `prism.hadamard.*` 元数据（401 个权重名 + 28672 个符号值），需要由 fork 的 Hadamard 变换内核还原，stock 构建没有这些内核。

---

## 5. 接口用法细则

### 思考模式
这是思维模型，`reasoning_content` 与正文分开返回。默认「思考」由模板决定（`xhigh`）。可调：

```powershell
.\scripts\start-server.ps1 -ReasoningEffort medium    # xhigh | medium | low（其它值会 500）
```

按次关闭思考（OpenAI 风格）：

```json
{ "chat_template_kwargs": { "enable_thinking": false } }
```

⚠️ **客户端 `max_tokens` 不要设得太小**：思考会先占用预算，太小会导致**空答案**，更小还会触发 **HTTP 500**（机理见第 9 节）。约 300 词的任务需要约 4000 token。

至于**服务端** `-ReasoningBudget` 能否用来兜底：**未实测**（我上一轮的判断越界了）。本 fork 的 `--reasoning-budget-message` 能在预算处干净收尾，理论上与客户端硬截断不是一回事——待测项见 `IDEAS.md` E1。

### 原生工具调用
`--jinja` 已启用，返回标准 `tool_calls` 字段，实测：

```
tool_calls -> get_weather({"city":"Paris"})
```

### 流式输出
标准 SSE（`stream: true`），实测帧格式正确并以 `[DONE]` 结尾。

### 图像输入
**当前不可用。** 27B 是视觉模型，但需要额外的 `mmproj` 投影文件（约 +0.9 GB），工作区里没有。只有文本。

### 鉴权
```powershell
.\scripts\start-server.ps1 -ApiKey 'sk-你的密钥'
```
之后所有 `/v1/*` 请求需带 `Authorization: Bearer sk-你的密钥`。

### 局域网访问
```powershell
.\scripts\start-server.ps1 -Bind 0.0.0.0 -ApiKey 'sk-你的密钥'
```

---

## 6. 目录结构

```
D:\dsh\DSH\new\
├─ start.cmd / stop.cmd / status.cmd   双击启动器
├─ QUICKSTART.md                       一屏版上手文档（日常看这个）
├─ README.md                           本文档（完整技术日志）
├─ BUGREPORT-prism-fork-500.md         已提交 issue 的草稿备份
├─ models\
│   └─ Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf     5.54 GiB
├─ runtime\llama-prism\                fork 运行时（74 个文件）
│   ├─ llama-server.exe / llama-cli.exe / llama-bench.exe ...
│   ├─ ggml-cuda.dll                   180 MB CUDA 后端
│   └─ cublas64_13.dll / cublasLt64_13.dll / cudart64_13.dll
├─ scripts\
│   ├─ start-server.ps1 / stop-server.ps1 / status-server.ps1
│   ├─ start-guard-proxy.ps1           守护代理（模型感知的输出预算地板）
│   ├─ verify-server.ps1               6 项端到端自检
│   ├─ chat.ps1                        命令行对话
│   └─ calibration\                    本次调参用的压测脚本（可重跑）
│       ├─ calibrate.ps1 .. calibrate6-kvtypes.ps1   显存/吞吐/KV 类型标定
│       ├─ suites.ps1                  上下文上限 / KV 放内存 / 分层卸载三组
│       ├─ quality-probe.ps1           短任务 + 长上下文检索的精度对照
│       ├─ hard-battery.ps1            判别性任务（多针/计数/干扰项/合并/数值）
│       ├─ ppl-compare.ps1             困惑度对照（逐 chunk 配对分析用）
│       ├─ kv-bias.ps1 / kv-bias-rotmatched.ps1     官方 KV 均值中心化偏置评测
│       ├─ deep-load-check.ps1         任意档位的满载峰值与深上下文速度
│       ├─ ram-audit.ps1               内存构成审计（cache-ram 对照）
│       ├─ optimise.ps1                加载模式/批大小/提示缓存对照
│       ├─ concurrency.ps1             槽位 / unified KV / 并发批处理对照
│       ├─ mem-probe.ps1               单次调用内存探针（带看门狗）
│       ├─ thinking-probe.ps1          思考 token 用量测量
│       └─ lang-drift*.ps1             语言漂移量化（上一代模型的问题，本代正常）
├─ corpus\                             评测语料与中文提示词（UTF-8 纯文本）
├─ logs\
│   ├─ *.json                          各轮实测结果（README 里的表格来源）
│   ├─ guard-proxy.log                 守护代理日志（记录客户端实际发的 max_tokens）
│   ├─ benchmarks\                     173 个压测过程日志
│   ├─ lang\                           27 个语言漂移测试输出
│   └─ server-history\                 76 个历史服务日志（最新一对留在根目录）
└─ tools\
    ├─ llm-guard-proxy.mjs             守护代理源码
    ├─ read-asar.mjs / js-yaml.mjs     读 app.asar 内文件 / YAML 解析库
    ├─ validate-yaml.mjs               写 DSH 配置前先校验
    ├─ read-session.mjs                解压并解析 DSH 会话日志（多帧 zstd）
    └─ inspect_gguf.py                 纯 Python GGUF 元数据/张量解析器
```

`logs\` 下的 `calibration.json`、`bench*.json`、`kvtest.json`、`quality-probe.json`、`hard-battery.json`、`ppl-compare.json`、`ramtest.json`、`optimise.json`、`concurrency.json` 保存了每一步的原始实测数据（含每道题的对错明细）。

---

## 7. 故障排查

**启动失败** —— 脚本会打印日志末尾的 stderr。完整日志在 `logs\server-<profile>-<时间戳>.log` 与其 `.err`。

**怎么确认没有隐性溢出（最重要的一条）** —— 溢出到共享内存时服务照样"启动成功"，只是吞吐塌掉。用两条命令当场判定：

```powershell
# 1) 余量：低于约 200 MiB 就有风险，接近 0 基本已溢出
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv

# 2) 逐秒看服务自身 CPU 与 GPU 利用率
Get-Process llama-server | Select-Object Id,@{n='CPU%';e={[int]($_.CPU)}}
```

**溢出颠簸的指纹**是这两个信号同时出现：**显存余量只剩个位数 MiB，而 GPU 利用率长期 100%**。此时 GPU 并非在计算，而是在 WDDM 共享内存里换页，还会伴随约 20 秒一次的周期性节拍。本次调参中就实测到一次：ctx 65536 配 q8_0 时占用 7888 MiB、余量仅 **12 MiB**，GPU 显示 100% 却在空转。同样的 65536 配 q4_0 只用 7122 MiB，余量 777 MiB，正常满速——**同样的上下文长度，换个 KV 精度就从溢出变成宽裕**。

**速度突然变慢** —— 先按上面的方法查余量。多半是显存被别的程序挤了，或用了一个对该 KV 精度过大的上下文。跑 `status.cmd` 看当前档位，或降档。

**想换回 32k 省显存** —— `.\scripts\start-server.ps1 -Profile safe -Force`。

**重新验证 fork 是否还在正确加载模型** ——

```powershell
.\runtime\llama-prism\llama-cli.exe -m .\models\Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf -st -rea off -p "test"
```

输出里应出现 `ftype : PTQ1_0 - 1.75 bpw ternary (group 128)`。

**手动指定参数组合** —— 可覆盖 profile 的上下文与 KV 类型：

```powershell
.\scripts\start-server.ps1 -Ctx 40960 -KvType q8_0 -Force
```

注意：`-Ctx` 需要是 1024 的倍数；`-KvType` 请只用单一类型（见 3.3），可选 `f16` / `q8_0` / `q4_0`。

---

## 8. 关于开机自启

未配置。若需要，用任务计划程序新建任务，操作设为：

```
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\dsh\DSH\new\scripts\start-server.ps1"
```

触发器选「登录时」。不建议注册为 Windows 服务——服务账户拿不到桌面会话的 CUDA 上下文。

---

## 9. 接入 DSH 作为模型供应商

已接入。DSH 的模型供应商配在 profile 的补丁层里：

```
C:\Users\<用户名>\.dsh\profiles\desktop\cordis.patch.yml
```

在 `llm-pi-ai` 插件的 `config.providers` 下新增了 `bonsai-local`：

```yaml
      bonsai-local:
        displayName: Bonsai Local (llama-prism)
        api: openai-completions
        baseURL: http://127.0.0.1:8080/v1
        compat:
          thinkingFormat: deepseek
          supportsReasoningEffort: true
        models:
          - id: ternary-bonsai-2-27b
            name: Ternary Bonsai 2 27B (PTQ1_0, 96k)
            contextWindow: 98304
            reasoningEfforts:
              off:
              minimal: low
              low: low
              medium: medium
              high: xhigh
              xhigh: xhigh
              max: xhigh
```

原有的 `bendi`（指向 LM Studio 的 1234 端口）**保持不动**，两套并存，可在选择器里切换。

### 关键坑：`reasoning_effort` 会让服务 500

这是接进去最容易坏的地方，逐字段实测出来的：

| 请求体字段 | 结果 |
|---|---|
| `stream_options` / `store` / `user` / `parallel_tool_calls` / `tool_choice` / `tools` / `frequency_penalty` / `presence_penalty` / `top_p` / `max_completion_tokens` | ✅ 接受 |
| **`reasoning_effort` = `high`** | **HTTP 500** |
| **`reasoning_effort` = `default`** | **HTTP 500** |
| `reasoning_effort` = `xhigh` / `medium` / `low` | ✅ 接受 |

原因在模型的 chat template：它只接受 `xhigh | medium | low`，其它值直接抛异常；而 **DSH 默认发的是 `high`**（`agent-default-model` 里 `reasoningEffort: high`）。所以需要做映射——这正是上面 `reasoningEfforts` 把 `high → xhigh`、`minimal/max → low/xhigh` 的原因。**任何映射到 `high`/`default` 的写法都会 500。**

### 为什么这样配

- **`apiKeyEnv: BENDI_API_KEY`**：pi-ai **拒绝无密钥的供应商**——省略该字段会让整个供应商不可用（`No API key for provider: bonsai-local`，实测见下文"坑一"）。这里引用的是 LM Studio 那个密钥：服务没有 `--api-key` 会忽略它，且**全程只在本机回环传输**（8090 代理 → 8080 服务）。**更干净的做法**是在设置页给该供应商填一个占位密钥，然后删掉这一行（见 QUICKSTART 第 2 节）。
- **`compat.thinkingFormat: deepseek`**：本服务把思考放在 `reasoning_content`，与 deepseek 格式一致。它会让插件发送 `thinking: {type: enabled}`，实测被服务正常接受。
- **`supportsReasoningEffort: true`**：不开启的话 `reasoning_effort` 根本不会发送，思考档位就无法真正控制服务端。
- **`contextWindow: 98304`**：按 `default` 档的真实上下文填。**不填对会白费扩容**——DSH 会按你填的值触发上下文压缩，填小了扩出来的空间用不上。

### 生效方式：热加载，不需要重启

boot 源码里标注得很明确：

> `/** The user patch layer inside a profile directory (hot-reloaded on long-lived surfaces). */`

而且解析失败也**不会拖垮进程**——"unreadable or unparsable file logs a warning and keeps the last good tree: a hot-reload of a live app must never take the process down"。所以改坏了最坏结果是保持旧配置 + 一条警告。

生效后如界面没刷新，按 **Ctrl+R** 即可。

### 已知语义限制

- **选 `off` 无法真正关闭思考**：映射为空值后不发送任何 reasoning 字段，服务端仍按模板默认（思考开启）。要真正关闭只能由客户端发 `chat_template_kwargs.enable_thinking = false`；服务端 `-ReasoningBudget` **不是**可行办法（见下）。
- **思考内容会显示**：本模型思考较长，客户端 `max_tokens` 偏小会让思考占满预算、返回空答案（见 3.15）。**解法是给足客户端预算，不要设服务端预算。**

### ⚠️ 已知会导致 HTTP 500 的两个触发条件

接入后需要知道这两条，否则会看到莫名其妙的失败。

**触发条件一：客户端的 `reasoning_effort` 值不在 `xhigh|medium|low` 之内。**

模板直接抛异常，服务返回 500，日志里是：

```
Jinja Exception: Unexpected reasoning effort high.
Supported types are xhigh (default), medium, and low.
```

DSH 默认就发 `high`，所以需要靠 `reasoningEfforts` 映射（见上）。**任何映射到 `high`/`default` 的写法都会 500。**

**触发条件二：输出预算不足以装下思考时，思考被截断会让服务返回 500。**

这一条我一开始判断错了。实测把因果链摆正了——**同一个提示，只改输出预算**：

| 提示（目标 300 词） | 输出预算 | 结果 |
|---|---|---|
| `Explain attention in about 300 words.` | 2048 | **HTTP 500，2/2 复现** |
| 同上 | 3072 | ✅ 200（但思考吃光全部，正文 0 字符） |
| 同上 | **4096** | ✅ 200，**正好 300 词**，未顶上限 |
| `Explain attention.`（无字数要求） | 2048 | ✅ 200，思考仅 283 token |

**所以不是"字数要求"有毒，而是思考被过小的预算截断**：截断处产生的畸形文本让 fork 的解析器解析失败。日志里的原始错误是服务在解析自己生成的文本时失败：

```
[json.exception.parse_error.101] parse error at line 1, column 1373:
syntax error while parsing object - invalid literal;
last read: '"We need answer user: \"Explain attention in about 300 words.\" ...'
```

用户在正常使用时观察到的现象与此吻合：规定字数后，模型会"先生成、再逐个数字、不符就重新生成、再数一遍"，这个循环把预算烧光，于是撞上截断。

**工具化计数并不能解决它——实测反而更糟**（`count_words` 工具 + "用工具验证并修订"）：

| 工具方案 | 输出预算 | 结果 |
|---|---|---|
| 带 `count_words` 工具 | 2048 | 调用 1 次工具，**正文为空** |
| 同上 | 4096 | **一次都没调用**，正文为空 |
| 同上 | 8192 | 调用 **4 次**（轮次用尽），7521 token，**仍无正文** |

工具把这个循环**合法化**了：模型反复"草稿→计数→修订"，迟迟不交稿。而**不用工具、预算 4096，一次就给出正好 300 词**。

**应对（按效果排序）**：

1. **给足输出预算**：这是目前试过的办法里比较有效的一个。约 300 词的目标配 4096 token 刚好；更长文本按比例放大。**预算是这里的主导变量，不是提示词措辞。**
2. **服务端 `-ReasoningBudget` 这次没有设**：它同样会截断思考，我们怀疑它可能引发同类问题——但这一点**没有实测过**（见第 9 节末）。本 build 上保持 `0`（不限制）。
3. **为此加计数工具帮助不大**：我们实测没有改善，反而更慢。
4. 能不用精确字数就不用。

> 注：这是 build 10754 的问题（用户确认该分支即为最新版），不是本地配置写错；换其它版本前无从验证是否修复。

### 关于思考预算：真正的变量是客户端的输出预算

数据和最初的直觉相反，值得记下来：

- **正常提问下思考很便宜**：`Explain attention.` 只用 **283 token**。
- **但预算不是质量旋钮，而且过小会致命**：思考被截断会触发上面那个 500。所以"给 2048 就够"这个说法不太成立——2048 恰好就是出现 500 的那个配置。
- **加大客户端 `max_tokens` 才是解法**：同一提示 2048→500，4096→正好 300 词。
- **服务端 `-ReasoningBudget` 这次没有加**：理由与上面相同，不过这一点**尚未实测**。

结论：**让客户端给足输出预算，服务端不设思考上限。**

### 改动与回退

- 备份：`cordis.patch.yml.bak-manual-<时间戳>`（同目录）
- 改动仅为**新增** `bonsai-local` 块，不修改也不删除任何原有内容
- 回退：把备份复制回去，热加载后即恢复；或直接删掉 `bonsai-local` 那一块
- 校验工具：`tools\validate-yaml.mjs`（从 app.asar 提取的 js-yaml，写盘前可先校验）
- asar 读取工具：`tools\read-asar.mjs`（可列出/导出 app.asar 内任意文件）

### 守护代理：只对该模型生效的输出预算地板

因为 build 10754 在思考被小预算截断时会报 500（见上），加了一层**模型感知**的小代理：

```
DSH ──► 127.0.0.1:8090 (llm-guard-proxy) ──► 127.0.0.1:8080 (llama-server)
```

它的行为**只有一件事**：当请求里的 `model` 等于目标模型（`ternary-bonsai-2-27b`）且输出预算低于地板（默认 4096）时，把预算抬到地板值。**其它模型原样透传**，因此对 `bendi`（LM Studio）零影响。

**刻意不做的事**：不改写提示词。静默改写用户提示会改变语义，而且实测表明提示词侧的缓解（"不要数字数"、提供计数工具）并不能真正解决问题——工具反而让循环更长。

启用与停用：

```powershell
.\scripts\start-guard-proxy.ps1                 # 8090 -> 8080，地板 4096
.\scripts\start-guard-proxy.ps1 -Floor 8192     # 抬高地板
.\scripts\start-guard-proxy.ps1 -Stop
```

验证记录（2026-10-05）：

| 测试 | 结果 | 代理日志 |
|---|---|---|
| 目标模型 + `max_tokens=2048`（此前 4/5 报 500） | ✅ HTTP 200，295 词 | `RAISED max_tokens 2048 -> 4096` |
| 其它模型 | ✅ 原样透传 | `model not targeted` |

**注意**：DSH 的 `bonsai-local` baseURL 现指向 `http://127.0.0.1:8090/v1`。**代理未运行时该供应商会不可用**，此时把 baseURL 改回 8080 即可直连（代价是失去这层保护）。

代理日志在 `logs\guard-proxy.log`，它会记录每次请求的**实际 `max_tokens`**——这也是观察 DSH 到底给多少输出预算的一个途径。

### 端到端实测记录（2026-10-05，DSH 首次真正跑通）

接入过程中踩了两个坑，都已修复并实测验证。**这两条比前面的配置说明更重要**，因为它们决定"能不能真的用起来"。

**坑一：省略 `apiKeyEnv` 会让供应商完全不可用（我先前判断错了）。**

插件 README 说省略 `apiKeyEnv` 会进入"configured-but-keyless"状态。**实测不是**：配置合法、模型能在选择器里出现、模型发现（`GET /v1/models`）也成功，但**一发聊天请求就被 pi-ai 拒绝**，DSH 会话日志里是：

```
assistant/attempt → finish: {"kind":"error","failure":{
    "message":"No API key for provider: bonsai-local","code":"PI_AI_ERROR"}}
turn/end → error: No API key for provider: bonsai-local
```

整个失败只用了 3.2 秒，代理侧只看到 `GET /v1/models`、**没有任何 chat 请求**——因为请求在客户端就被拦下了。

修法：引用一个已存在的凭据即可（本服务没有 `--api-key`，会忽略该密钥）：

```yaml
        apiKeyEnv: BENDI_API_KEY
```

> 一点体会：**"配置合法 + 模型能列出"不等于"能调用"**。验证时最好真的发一次 chat 请求，否则会把"选择器里看得见"误当成"已经能用"。

**坑二：DSH 的"最大输出 token 数"留空 ≠ 没有上限。**

设置页那一栏留空时是灰字占位符 `32K`，那就是"不填时实际生效的值"。取值链是：

```
models[].maxTokens  ??  内置目录的 base.maxTokens  ??  defaultMaxTokens
     空 = undefined         我们这路由不在目录里          默认 32768
```

实测（守护代理日志）确认：

```
16:31:48 POST /v1/chat/completions model="ternary-bonsai-2-27b"
          stream=false -> 200 :: budget ok (max_completion_tokens=32768)
```

注意两点：**DSH 实际发的是 `max_completion_tokens` 字段**（不是 `max_tokens`，两者 llama-server 都接受）；数值就是 **32768**。所以守护代理的 4096 地板**在 DSH 场景永不触发**（日志写 `budget ok`），它保护的是其它客户端。

**首次成功调用的真实规模**（供性能预期参考）：

| 指标 | 实测 |
|---|---|
| 提示 token | **31,625**（DSH 的 system prompt + 全部工具 schema） |
| 预填充 | 约 **49 秒**（647 t/s） |
| 生成 | 约 **28 t/s**（比单请求空载的 46 t/s 低，因上下文深） |

含义：**DSH 每次冷启动一轮对话要先处理约 3.1 万 token 的前缀**。好在提示缓存在多轮里生效（相同前缀只处理增量），所以只有第一轮付这个代价；但若上下文被压缩或缓存失效，这 49 秒会重新出现。

### 关于"这是分支问题吗"——不是，更可能是上游遗留

需要分清两件事：

- **"是 bug"：确定。** 服务在解析自己生成的文本时抛错，配置与请求均无问题。
- **"是 PrismML 分支独有"：不确定，而且证据指向相反方向。** 排障中发现一组 upstream llama.cpp 的同类问题：
  - [#20193](https://github.com/ggml-org/llama.cpp/issues/20193)「**max_tokens 到达时**返回 500 而不是正常收尾」——已关闭（2026-03-08），复现用 Qwen3.5-4B，同样要求"加大 max_tokens 即正常"；
  - [#20708](https://github.com/ggml-org/llama.cpp/pull/20708)「新 chat parser 在解析失败时抛异常导致推理崩溃，在 llama-server 表现为 HTTP 500」；
  - [#20800](https://github.com/ggml-org/llama.cpp/pull/20800)「给 JSON_NATIVE 工具解析器加 content-only 回退」——正是这类崩溃的修法。

  我们的报错文本（`[json.exception.parse_error.101]`、并出现 **ill-formed UTF-8 byte**）与 #20193 不同，端点也不同（`/v1/chat/completions` vs `/v1/messages`），所以更像是**一个仍然存在的相邻缺陷**，而不是已被修掉的那个。

**因此报告位置的建议**：报给 **PrismML-Eng/llama.cpp**（这个模型只能跑他们的二进制，且已确认他们仓库里没有这条报告），并在报告里引用上面三个上游 issue，让维护者判断该修在 fork 还是上游。

已备好完整报告草稿：**`BUGREPORT-prism-fork-500.md`**（含环境、命令行、curl 复现、A/B 对照表、两种报错原文、机理假设与关联 issue），可直接粘贴为 GitHub issue。另：PrismML 仓库里 [issue #267](https://github.com/PrismML-Eng/llama.cpp/issues/267) 已记录"混合 K/V 导致 flash attention 跑在 CPU 上变慢"，与本 README 3.3 节的实测一致，属已知问题。

---

## 10. 怎么启动、以及在手机上使用

### 启动

模型不会开机自启，重启电脑后要手动起（两件东西，缺一不可）：

```powershell
.\start.cmd                 # 等价于 .\scripts\start-server.ps1 —— 起 llama-server（8080）
.\scripts\start-guard-proxy.ps1   # 起守护代理（8090）—— DSH 走这个端口
```

* `start.cmd` 是双击启动器，已绑定 GUI 里的"Bonsai Local"这个供应商所需的 8080。
* **守护代理只在 DSH 场景需要**（它把过小的输出预算抬到地板值、避免那个 500）。手机浏览器直连 8080，不需要它。
* 停：`.\stop.cmd`（只停 llama-server）；代理用 `.\scripts\start-guard-proxy.ps1 -Stop`。

### 在 DSH 里用它

模型选择器里选 **Bonsai Local (llama-prism)** → `Ternary Bonsai 2 27B (PTQ1_0, 96k)` 即可，不需要其它操作。DSH 侧已配好：baseURL 指向代理 8090、协议 OpenAI Chat Completions、上下文窗口 98304、`apiKeyEnv` 指向一个已存在凭据（本服务会忽略该密钥）。

### 在手机上用

**不需要装 App**：llama-server 自带一个完整的 Web 聊天界面（SvelteKit SPA，带手机图标与 manifest，可以"添加到主屏幕"当 PWA 用）。

1. 让手机连**同一个 Wi-Fi**；
2. 手机浏览器打开 `http://<电脑的局域网IP>:8080`（启动脚本末尾会直接列出可用地址，本机 WLAN 是 `<局域网IP>`）；
3. 想要 App 体验就点浏览器的"添加到主屏幕"。

也可以让手机上任何支持自定义 OpenAI 端点的聊天 App 连过来：API 地址填 `http://<IP>:8080/v1`，模型名 `ternary-bonsai-2-27b`，密钥留空。

启动时加 `-Bind 0.0.0.0` 才会对局域网开放（默认只绑 127.0.0.1）：

```powershell
.\scripts\start-server.ps1 -Bind 0.0.0.0            # 对局域网开放
.\scripts\start-server.ps1                          # 恢复只允许本机
```

**已实测**（2026-10-06，WLAN 地址）：

| 检查 | 结果 |
|---|---|
| `/health`、`/v1/models`、Web UI `/` | 全部 HTTP 200 |
| Web UI 的 JS 资源（2.6 MB） | HTTP 200 |
| 经局域网地址发一次推理 | HTTP 200，正常返回 |

**防火墙不需要额外操作**：`llama-server.exe` 已有 Public/Inbound/Allow 规则，而 WLAN 恰好是 Public 类别，正好匹配。

### 两个值得知道的点

**① Web UI 要求客户端支持 gzip。** 用 curl 或 PowerShell 直接 GET `/` 会得到 `415 Unsupported Media Type`，响应体是 `Error: gzip is not supported by this browser`。**这是正常的**——浏览器都会带 `Accept-Encoding: gzip`，所以手机上没问题。别把它当成故障（我一开始就误判成了"这个 build 没有 Web UI"）。

**② 局域网模式没有鉴权：同一网络里的任何人都能用这个模型、占用你的显卡。** 家里 Wi-Fi 通常可接受，但注意：

* 不要把它暴露到公网（路由器上别做端口转发）；
* 想加密钥就用 `-ApiKey 'sk-你的密钥'`，但**要注意 DSH 会因此失效**——DSH 发的是 `apiKeyEnv` 里那个凭据，与服务器密钥不一致就会被 401。要两者并存，得把 DSH 的 `apiKeyEnv` 换成同一个密钥（存进 `~/.dsh/.credentials.yaml`），这需要额外改一处配置。
