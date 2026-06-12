# MVP Curriculum EC — Setup Guide

## 问题诊断

你的 CSV 输出显示所有 `fitness=0`、`success_rate=0`，这说明 **EC 评估器没有正确运行**。

### 根本原因

`PushTBasicEvolutionEvaluator` 使用 `FrozenEvaluator` agent 来评估基因组，但这个 agent 需要：
1. `BehaviorType = InferenceOnly`（不等待 Python 训练器决策）
2. 分配训练好的 `.onnx` 模型（用于推理）

如果这两个条件不满足，agent 会等待 Python 训练器的决策，但 EC 运行时没有 Python 连接，导致所有 episode 超时，返回空结果。

---

## 修复步骤

### 步骤 1：训练一个基础模型

```bash
mlagents-learn config/ppo/PushBlock.yaml --run-id=pushblock_warmup
```

训练到 agent 能稳定解决默认环境（约 500k-1M steps）。

### 步骤 2：找到导出的 .onnx 模型

训练完成后，模型在：
```
results/pushblock_warmup/PushBlock/PushBlock.onnx
```

### 步骤 3：配置 FrozenEvaluator

1. **在 Unity 中打开场景** `PushBlockEC_V2`
2. **在 Hierarchy 中找到** `EvaluationArena > FrozenEvaluator`
3. **在 Inspector 中设置 BehaviorParameters：**
   - `Behavior Type` → **Inference Only**
   - `Model` → 拖入 `PushBlock.onnx`

### 步骤 4：运行诊断

1. 在 Hierarchy 中选中 `CurriculumRunner`
2. 如果有 `FrozenEvaluatorDiagnostics` 组件，点击 **"Run Diagnostics"**
3. 检查控制台输出，确保所有检查通过

### 步骤 5：运行 EC

1. 在 Hierarchy 中选中 `CurriculumRunner`
2. 右键点击 `PushTCurriculumEC` 组件 → **"Run Curriculum EC"**
3. 等待 EC 完成（观察控制台日志）

### 步骤 6：验证结果

检查输出文件：
```
Assets/ML-Agents/Examples/PushBlock/Scripts/newPushTEC/curriculum_outputs/
├── genome_pool_stage_001_YYYYMMDD_HHMMSS.json
├── summary_stage_001_YYYYMMDD_HHMMSS.csv
├── all_evaluations_stage_001_YYYYMMDD_HHMMSS.csv
└── episodes_stage_001_YYYYMMDD_HHMMSS.csv
```

**正确的 CSV 应该有非零值：**
```csv
generation,genome_index,fitness,success_rate,mean_progress,...
0,3,142.531,0.500000,0.623000,...  ← 有实际评估数据
```

---

## 控制台日志诊断

### EC 正常运行时应该看到：

```
[PushT EC-Frozen] Model loaded: PushBlock
[PushT EC-Episode] Starting episode 0 with seed 12345
[PushT EC-Episode] Episode 0 finished: elapsed=8.52s, polls=512, IsEpisodeComplete=True, timedOut=False, success=True
Generation 0 best fitness: 142.531, success rate: 50%
[Curriculum EC] Goldilocks: 8 passed filters, 6 selected after diversity filter
[Curriculum EC] Pool updated: generation=1, count=6
```

### EC 有问题时会看到：

```
[PushT EC-Frozen] Agent BehaviorType is Default, should be InferenceOnly. EC will hang waiting for Python trainer decisions.
[PushT EC-Frozen] Agent has no .onnx model assigned.
[PushT EC] Episode 0 timed out after 15s. Agent.IsEpisodeComplete=False.
```

---

## 完整 MVP 流程

```
1. 训练基础模型 (pushblock_warmup)
   ↓
2. 导出 .onnx 模型
   ↓
3. 配置 FrozenEvaluator (InferenceOnly + .onnx)
   ↓
4. 运行 EC → 生成 GenomePool (stage_001)
   ↓
5. 训练 agent 使用 GenomePool (pushblock_curriculum_s1)
   ↓
6. 重复步骤 2-5，逐步提升难度
```

---

## 常见问题

### Q: EC 运行太慢怎么办？
A: 在 `CurriculumRunner` 的 `PushTEvolutionConfig` 中调小参数：
- `populationSize` = 8（默认 24）
- `generationCount` = 3（默认 20）
- `episodesPerGenome` = 5（默认 10）

### Q: Goldilocks 过滤器拒绝所有基因组怎么办？
A: 调宽阈值：
- `goldilocksMinSuccessRate` = 0.10（默认 0.30）
- `goldilocksMaxSuccessRate` = 0.90（默认 0.75）
- `minMeanProgress` = 0.20（默认 0.40）

### Q: 如何知道 EC 是否在正确运行？
A: 查看 `PushTBasicEvolutionEvaluator` 的 `enableVerboseLogging` 是否勾选，控制台会显示详细的 episode 信息。

---

## 文件清单

| 文件 | 用途 |
|------|------|
| `GenomePool.cs` | 基因组池（ScriptableObject + JSON） |
| `PushTCurriculumEnvironment.cs` | 训练环境采样器 |
| `PushTCurriculumEC.cs` | EC 运行器（带 Goldilocks 过滤） |
| `PushTBasicEvolutionEvaluator.cs` | 具体评估器（驱动 FrozenEvaluator） |
| `PushTBlockGenome.cs` | 基因组数据结构 |
| `PushTEvolutionRunner.cs` | 基础 EC 循环 |
| `FrozenEvaluatorDiagnostics.cs` | 诊断工具 |
