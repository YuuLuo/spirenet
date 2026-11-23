# Models / 模型

The weights are attached to the GitHub release, not stored in git. `manifest.json` lists for every file its
SHA256, the run and configuration it comes from, its training counters, the model configuration and the
1,000-seed evaluation reported in the main README. Check a download with:

权重放在 GitHub Release 中，不在 git 仓库里。`manifest.json` 列出每个文件的 SHA256、来源训练线与配置、训练计数器、
模型配置，以及主 README 中报告的 1,000 种子评测结果。下载后可以这样校验：

```
cd checkpoints && sha256sum -c ../models/SHA256SUMS
```

Each file is a `torch.save` dictionary / 每个文件是一个 `torch.save` 字典：

| key | content / 内容 |
|---|---|
| `format` | checkpoint format version / 存档格式版本 |
| `ema` | the EMA weights used for play (fp32 state dict) / 对局使用的 EMA 权重（fp32） |
| `extra` | model config, tokenizer config, vocab / catalog hashes / 模型与 tokenizer 配置、vocab / catalog 哈希 |
| `counters` | update version, samples seen / 更新次数、已消耗样本数 |
| `config` | the PPO configuration of the run / 该训练线的 PPO 配置 |

The optimizer state and the online (non-EMA) weights were removed with `py/scripts/slim_checkpoint.py`;
the EMA tensors are unchanged. Load them with `spire.evalpanel.load_policy_net(path, weights="ema", ...)` or
through `spire.habit_eval` / `spire.evalpanel`.

优化器状态和在线（非 EMA）权重已用 `py/scripts/slim_checkpoint.py` 去掉，EMA 张量本身未做任何改动。
可以用 `spire.evalpanel.load_policy_net(path, weights="ema", ...)` 加载，或直接交给 `spire.habit_eval` / `spire.evalpanel`。
