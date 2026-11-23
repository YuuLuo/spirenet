# SpireNet

[English](README.md)

SpireNet 是《杀戮尖塔 2》（*Slay the Spire 2*）的强化学习智能体。它只使用人类玩家在屏幕上能看到的信息，
在进阶 10（含第三幕双 Boss）下完成整局游戏。策略网络是实体 Transformer，在运行原版游戏代码的无界面宿主上
用异步 PPO 训练。

本仓库包含宿主程序、完整的训练与评测代码、各条已发布训练线的配置，以及用来测胜率的种子集。模型权重放在
GitHub Release 中。

> SpireNet 是独立项目，与 Mega Crit 没有任何关联，也未获其认可。仓库中不包含任何游戏文件、游戏代码或游戏文本。
> 运行需要你自己拥有的正版游戏。

## 结果

纯网络、贪心选择、对局时不做任何搜索。进阶 10，每个角色 1,000 个固定种子
（`eval/panels/<角色>-a10-dev1000.json`），每个种子打一局。

| 角色 | 模型 | 参数量 | 训练样本数 | 胜场 / 1,000 | 胜率（95% 置信区间） |
|---|---|---:|---:|---:|---|
| 静默猎手（Silent） | `silent-19m` | 18.2 M | 45.3 亿 | 854 | **85.4%**（83.1–87.5） |
| 故障机器人（Defect） | `defect-4m` | 4.2 M | 164.0 亿 | 780 | **78.0%**（75.3–80.5） |
| 铁甲战士（Ironclad） | `ironclad-4m` | 4.2 M | 152.3 亿 | 741 | **74.1%**（71.3–76.7） |
| 愚者（Fool） | `fool-19m` | 18.2 M | 31.3 亿 | 715 | **71.5%**（68.6–74.2） |

- 胜率是在 Windows 上用本仓库的代码和宿主测得的（`spire.habit_eval`，见[评测](#评测)）。
- 置信区间为 Wilson 区间。1,000 局读数的标准误约 1.3 个百分点。GPU 上的贪心对局不能逐位复现（bf16 下的近似平局），
  同一权重在同一批种子上重跑，胜率会有 1 个百分点左右的浮动。
- 这些种子集在训练期间用于监测进度，其中任何一局都从未被当作训练数据。
- 训练样本数是该网络的学习器消耗的决策数（19M 网络从蒸馏开始计，4M 网络从零开始计）。
- Necrobinder 和 Regent 两个角色尚未训练到同等水平，不发布。

## 工作原理

**宿主。** `host/` 是 SpireHost，一个 .NET 9 程序：加载游戏自带的 `sts2.dll`，以无界面、确定性的方式驱动整局游戏。
它的 Harmony 补丁只替换规则之外的引擎部分（不渲染、虚拟时钟、内存中的存档、输入注入），没有任何补丁改变游戏行为；
每个补丁及其用途都列在 `host/FIDELITY.md` 中。每个决策点，宿主发出一帧紧凑的二进制表格（卡牌、生物、能力、意图、
遗物、药水、地图、合法动作……）。训练时每个 actor 进程带 60 个宿主进程。

**信息边界。** 出牌和决策只看屏幕上能看到的信息。抽牌堆顺序、种子、随机数状态和未揭示的房间都不会发送。
玩家能学到的规则知识是允许的：宿主会根据敌人公开的行动模式，算出它接下来的行动预报（随机行动只给出可能的行动集合，
绝不给出实际抽中的结果），以及回合汇总（`FORECAST` / `CALC` 表）。

**模型。** 每一行表格变成一个 token：可学习的 ID 嵌入、对静态 catalog 特征的小型 MLP、symlog 缩放后的数值，
以及附属行的池化（能力并入其拥有者，地图边并入节点）。之后由不带位置编码的 pre-norm Transformer 处理。
所有 LayerNorm 都以角色、进阶等级和座位数为条件。指针头根据每个合法动作的来源、目标和全局 token 为其打分。
价值、胜率、到达层数和血量是同一主干上的辅助头。两种规模：`4m`（d 224，4 层）和 `19m`（d 384，8 层）。

**训练。** actor 用 EMA 权重做批量、CUDA graph、bf16 推理；学习器在一张或多张 GPU 上跑 PPO。已发布训练线用到的方法：

- 稠密奖励（每层 +0.01、精英 +0.02、Boss +0.1、胜利 +1、死亡 −0.1，外加血量项），战斗内外分别计算 GAE；
- 在战略性决策（地图、选牌奖励、商店、营火、事件）上把探索混入行为策略，学习器做精确的重要性修正；
- 进阶课程（`py/scripts/solo_curriculum.py`）：当前进阶的采样胜率达到 15% 时自动提高进阶；
- **kickstart 蒸馏**：把训练好的 4M 网络扩成 19M 网络；
- **ExIt**：搜索进程在重新抽样隐藏信息（抽牌顺序和未来的随机结果都重新抽取，从不读取真实值）的前提下评估局面，
  在搜索明显优于网络时生成策略标签；

更多细节（每个超参数、每个模型的来历）见 [docs/TRAINING.zh-CN.md](docs/TRAINING.zh-CN.md)。

## 目录结构

```
host/                 SpireHost（C#，.NET 9）：引擎驱动、观测编码、通信协议、分叉服务
host/tools/           prepare_game_dir.py、Publicizer
host/vocab/vocab.json 已发布模型使用的 ID 空间（只有名称；catalog.json 在本地生成）
py/spire/             Python 包：tokenizer、模型、actor、学习器、PPO、ExIt、评测
py/configs/           宿主启动配置（hosts/）、已发布训练线的配置（runs/）、removal-ratings.json
py/scripts/           进阶课程、权重精简、基准测试、迁移脚本
py/tests/             单元测试和集成测试
eval/panels/          上表使用的 1,000 种子集
docs/                 训练说明
```

## 快速开始

### 环境要求

- Windows 10/11 x64 或 Linux x64（WSL2 也可以；编译和评测已在 WSL2 的 Ubuntu 22.04 上验证）。
  训练在 Linux 上完成，评测两者都可以。搜索和 ExIt 进程需要 Linux
  （分叉服务使用 `fork(2)`）。
- .NET 9 SDK。
- Python 3.11+、带 CUDA 的 PyTorch 2.7+、NumPy。一张 NVIDIA 显卡（评测一张就够）。
- 《杀戮尖塔 2》**v0.113.1**（`sts2.dll` 的 SHA256 为
  `62ecbe91093e1fa862de4329aac865209dcb297804caf0f0520d92d8c0b4a9c3`）。宿主拒绝其他任何版本。
  如果你的游戏已经更新，可以用 Steam 的 depot 下载取回旧版本。

### 1–3. 用你的游戏编译宿主

每一步做什么、检查什么，见 `host/README.md` 的 "Build" 一节。在仓库根目录运行：

```
# Windows（PowerShell）
python host\tools\prepare_game_dir.py "<install>"
powershell -ExecutionPolicy Bypass -File host\make-publicized.ps1
dotnet build host\SpireHost\SpireHost.csproj -c Release -nodeReuse:false -p:UseSharedCompilation=false
host\out\bin\SpireHost\release\SpireHost.exe --export-vocab host\out\vocab-export --merge host\vocab\vocab.json
copy host\out\vocab-export\catalog.json host\vocab\
dotnet build host\SpireHost\SpireHost.csproj -c Release -nodeReuse:false -p:UseSharedCompilation=false

# Linux
python3 host/tools/prepare_game_dir.py "<install>"
bash host/make-publicized.sh
dotnet build host/SpireHost/SpireHost.csproj -c Release -nodeReuse:false -p:UseSharedCompilation=false
host/out/bin/SpireHost/release/SpireHost --export-vocab host/out/vocab-export --merge host/vocab/vocab.json
cp host/out/vocab-export/catalog.json host/vocab/
dotnet build host/SpireHost/SpireHost.csproj -c Release -nodeReuse:false -p:UseSharedCompilation=false
```

`<install>` 是包含 `SlayTheSpire2.pck` 的游戏目录（Steam：管理 → 浏览本地文件）。

1. `prepare_game_dir.py` 把游戏的托管程序集和 `release_info.json` 复制到 `host/game/`，并从游戏的 `.pck`
   中抽取英文本地化表，打包成宿主读取的格式。写入任何文件之前，它会先校验 `sts2.dll` 和本地化包的哈希。
2. `make-publicized` 生成 `sts2.dll` 的编译期引用，运行时从不加载它。
3. `--export-vocab` 从你的游戏生成 `catalog.json`（卡牌、遗物、怪物的静态数据，含卡牌文本）。输出必须是
   vocab 哈希 `0xB824481A`、catalog 哈希 `0x6D9889A2`，且 `merge_report` 为空；已发布的权重拒绝其他哈希。
   第二次编译会把 catalog 复制到可执行文件旁边。

`py/configs/` 中的启动配置使用 `host/out/bin/SpireHost/release/` 下的编译结果。重新编译会覆盖这些文件，
所以长时间训练时请生成不可变的部署目录（`make-dist`，见 `host/README.md`），并让启动配置指向它。

### 4. Python 环境

```
python -m venv .venv
.venv/bin/pip install -r py/requirements.txt     # Windows: .venv\Scripts\pip install -r py\requirements.txt
python py/native/build.py                         # 可选的 C 加速模块（Linux 用 gcc，Windows 用 MSVC）
```

如果 pip 会装成 CPU 版 PyTorch，请先安装与你的 CUDA 版本匹配的 PyTorch。所有命令都在 `py/` 下运行
（`python -m spire.<模块>`），不需要把它安装成包。C 加速模块只让 tokenize 更快，没有它时走 NumPy 路径，结果完全相同。

### 5. 下载权重

从 Release 下载 `*-ema.pt` 放到 `checkpoints/`，并用 `models/SHA256SUMS` 校验。文件中只有对局使用的 EMA 权重（`ema`）、
模型与 tokenizer 配置（`extra`）、训练计数器和 PPO 配置。

### 评测

在 `py/` 下运行（Windows 用 `configs/real_host.toml`，Linux 用 `configs/hosts/linux.toml`）：

```
python -B -m spire.habit_eval --checkpoint ../checkpoints/silent-19m-ema.pt \
  --host-config configs/real_host.toml --panel ../eval/panels/SILENT-a10-dev1000.json \
  --habit baseline --commit-select on --removal-rule configs/removal-ratings.json --rules-mode free \
  --envs 16 --gamelog-games 0 --out ../eval/results/silent-19m
```

`summary.json` 给出胜率及其区间、平均层数和失败情况；`games.jsonl` 每局一行。在 24 核 CPU 上用 16 个环境，
1,000 局约 15 分钟。

### 训练

在 `py/` 下运行：

```
python -B -m spire.train --config configs/runs/S25-silent-4m-ppo.json
python -B scripts/solo_curriculum.py S25-silent-v2a      # 另开一个终端：进阶课程
```

- 按你的机器修改 `actor.devices`、`actor.actors`、`hosts.envs_per_actor`、`learner.device` 和
  `learner.replica_device`。已发布配置是在 4× RTX 5090、约 180 个 CPU 核、约 300 GB 内存上运行的
  （每秒约 3–7 万样本）。
- `configs/runs/` 是各条已发布训练线的最终配置，用来说明模型是怎么训练出来的；其中大多是从更早的存档续训的
  （来历见 [docs/TRAINING.zh-CN.md](docs/TRAINING.zh-CN.md)）。`S25-silent-4m-ppo.json` 是纯 PPO，
  最适合作为起点：把 `episodes.lineup_ascensions` 设为 A0，让进阶课程逐步提高。
- ExIt（`ppo.exit_coef` > 0）需要搜索老师进程（`python -m spire.exit_teacher`，仅 Linux）把标签写进
  `ppo.exit_dir`。`exit_coef` 为 0 时学习器就是纯 PPO。
- `python -m spire.dashboard --run ../runs/<名称>` 会生成一次训练的 HTML 看板。

### 测试

```
cd py && python -m pytest -q
```

需要已编译的宿主、已生成的 catalog 或原生引擎的测试，在缺少这些条件时会自动跳过。

## 未包含的内容

- 游戏文件、反编译代码和游戏文本。
- 原生快速引擎（用 Rust 重新实现的游戏规则，用于加速搜索）。它源自游戏代码，因此不公开。本仓库的一切都通过
  SpireHost 运行在原版游戏代码之上。
- 带优化器状态的完整训练存档，以及实验记录本（代码注释和宿主文档有时会引用其中的 `notes/...` 文件）。

## 许可证

MIT，见 [LICENSE](LICENSE)。《杀戮尖塔 2》及其内容归 Mega Crit 所有。
