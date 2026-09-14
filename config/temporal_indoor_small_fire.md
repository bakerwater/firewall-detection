# 室内小火苗时序配置说明

本文档解释 [`temporal_indoor_small_fire.yaml`](./temporal_indoor_small_fire.yaml) 中的全部参数。该配置只调整 YOLO 输出后的候选过滤、目标跟踪和时序门控，不修改 YOLO 或 MViT 模型权重。

处理顺序如下：

```text
YOLO 原始检测框
  -> candidate_confidence 候选过滤
  -> ignore_regions 区域过滤
  -> tracker 目标跟踪
  -> TPT/AVT 时序门控
  -> MViT 二级验证（启用时）
  -> 最终报警
```

## 1. 时序门控模式

```yaml
temporal_mode: tpt
```

选择当前使用的时序门控算法：

- `tpt`：Temporal Persistence Technique，判断同一目标是否在一段时间内持续出现。当前室内小火苗配置使用该模式。
- `avt`：Area Variation Technique，判断同一轨迹的检测框面积是否发生足够变化。

命令行参数优先于 YAML。可以临时切换而不修改配置文件：

```bash
python run_temporal.py --source input.mp4 \
  --config config/temporal_indoor_small_fire.yaml \
  --temporal avt
```

## 2. 目标跟踪参数

```yaml
tracker:
  iou_threshold: 0.15
  center_distance_threshold: 1.5
  max_misses: 12
  max_missed_seconds: 1.5
  max_history: 64
  max_timestamp_gap: 5.0
  prediction_horizon_seconds: 0.75
```

### `iou_threshold: 0.15`

当前检测框与已有轨迹框的交并比（IoU）达到 `0.15` 时，可以关联为同一个目标。

- 调高：关联更严格，降低错误合并风险，但小火苗框抖动时容易形成轨迹碎片。
- 调低：轨迹更容易保持连续，但相邻目标可能被错误关联。

小火苗的检测框形状变化明显，因此这里使用较宽松的 `0.15`。

### `center_distance_threshold: 1.5`

当 IoU 不满足条件时，跟踪器使用中心点距离进行备用关联。实际距离会除以两个检测框的平均对角线长度进行归一化。

`1.5` 表示中心点移动不超过约 `1.5` 个目标框对角线时，仍可关联为同一目标。数值越大越不容易碎片化，但错误关联风险也越高。

### `max_misses: 12`

允许一条轨迹连续 `12` 次推理没有匹配到检测框；第 `13` 次连续漏检时删除轨迹。

- 10 FPS 推理时，约能容忍 `1.2` 秒的连续漏检。
- 5 FPS 推理时，按次数约为 `2.4` 秒，但还会受到 `max_missed_seconds` 的限制。

### `max_missed_seconds: 1.5`

如果距离该轨迹最后一次成功检测已经超过 `1.5` 秒，则删除轨迹。它与 `max_misses` 同时生效，任一条件先达到都会使轨迹过期。

### `max_history: 64`

每条轨迹最多保留最近 `64` 条历史记录，包括置信度、检测框、中心点、面积以及命中/漏检状态。

10 FPS 下最多约覆盖 `6.4` 秒。增大该值主要增加内存占用，通常不会直接提高检出率。

### `max_timestamp_gap: 5.0`

同一摄像头相邻两次处理的时间戳间隔超过 `5` 秒时，系统认为视频发生断流、跳转或重连，并清空该摄像头的旧轨迹，防止断流前后的检测被错误拼接。

### `prediction_horizon_seconds: 0.75`

跟踪器根据轨迹最近两次位置和大小变化预测下一位置，最多向前预测 `0.75` 秒。用于改善检测框快速抖动或短暂漏检后的重新关联。

## 3. 候选置信度

```yaml
candidate_confidence:
  fire: 0.30
  smoke: 0.20
default_candidate_confidence: 0.25
```

YOLO 推理阶段使用较低的原始阈值 `conf=0.01` 保留候选框，随后由这里按类别过滤。低于候选阈值的检测框不会进入跟踪器。

### `fire: 0.30`

只保留置信度不低于 `0.30` 的 fire 检测框。该值针对室内小火苗从默认配置的 `0.60` 降低，以保留弱小火焰候选。

- 调低：提高弱小目标召回，但增加误候选、跟踪负载和二级验证负载。
- 调高：减少误候选，但可能漏掉早期小火苗。

现场建议依次测试 `0.30`、`0.35`、`0.40`，不建议直接降到 `0.10`。

### `smoke: 0.20`

只保留置信度不低于 `0.20` 的 smoke 检测框。该值维持已有验证参数，避免优化小火苗时同时放大烟雾误候选。

### `default_candidate_confidence: 0.25`

当模型输出了未在 `candidate_confidence` 中单独配置的类别时，使用 `0.25` 作为该类别的候选阈值。

可在命令行临时覆盖 fire 或 smoke 阈值：

```bash
python run_temporal.py --source input.mp4 \
  --config config/temporal_indoor_small_fire.yaml \
  --candidate-confidence-fire 0.35 \
  --candidate-confidence-smoke 0.20
```

## 4. TPT 火焰参数

```yaml
classes:
  fire:
    window_size: 5
    min_hits: 3
    min_track_duration: 0.4
    min_confidence: 0.30
    high_confidence: 0.80
    max_misses: 2
```

只有 `temporal_mode: tpt` 时，`classes` 中的参数才生效。

### `window_size: 5`

检查该轨迹最近 `5` 次推理状态，包括成功检测和漏检。

- 10 FPS 下约覆盖最近 `0.5` 秒。
- 5 FPS 下约覆盖最近 `1.0` 秒。

### `min_hits: 3`

普通 fire 候选在最近 `5` 次推理中至少要有 `3` 次成功检测，才可以通过 TPT。

```text
命中 命中 漏检 命中 漏检 -> 3/5，可以继续判断
命中 漏检 漏检 命中 漏检 -> 2/5，不通过
```

### `min_track_duration: 0.4`

普通 fire 轨迹至少持续 `0.4` 秒才能通过。该条件用于过滤单帧灯光、反光和瞬时噪声。

当最新检测达到 `high_confidence` 时，会走高置信度快速通道，不受这个持续时间限制。

### `min_confidence: 0.30`

普通 TPT 窗口内至少要有一个检测框的置信度达到 `0.30`。

该值与候选阈值作用不同：候选阈值决定检测框能否参与跟踪，`min_confidence` 决定轨迹能否通过 TPT。例如命令行将候选阈值临时降到 `0.20` 后，低置信度框可以帮助维持轨迹，但窗口内仍需至少一个框达到 `0.30`。

### `high_confidence: 0.80`

最新 fire 检测框达到 `0.80` 时立即通过 TPT，不再等待 `3/5` 命中和 `0.4` 秒持续时间。

这是小火苗快速报警通道，也可能接纳单帧高置信度误检，因此建议配合 MViT 二级验证，不要把 TPT 事件直接作为最终报警。

### `max_misses: 2`

最近 `5` 次状态中最多允许 `2` 次漏检。它与 `min_hits: 3` 共同保证最近窗口至少有三次有效检测。

## 5. TPT 烟雾参数

```yaml
  smoke:
    window_size: 5
    min_hits: 3
    min_track_duration: 2.0
    min_confidence: 0.25
    high_confidence: 0.85
    max_misses: 2
```

- `window_size: 5`：检查最近五次推理状态。
- `min_hits: 3`：最近五次至少三次检测到同一烟雾轨迹。
- `min_track_duration: 2.0`：普通烟雾轨迹至少持续两秒；烟雾变化较慢，因此比火焰更严格。
- `min_confidence: 0.25`：窗口内至少有一个 smoke 框达到 `0.25`。候选阈值为 `0.20`，所以 `0.20` 至 `0.25` 的框可以维持轨迹，但不能单独让轨迹通过。
- `high_confidence: 0.85`：最新 smoke 框达到 `0.85` 时立即通过快速通道。
- `max_misses: 2`：最近五次最多允许两次漏检。

## 6. AVT 备用参数

```yaml
avt_classes:
  fire:
    window_size: 20
    area_threshold: 0.05
    min_samples: 2
    min_track_duration: 0.0
  smoke:
    window_size: 20
    area_threshold: 0.05
    min_samples: 2
    min_track_duration: 0.0
```

当前配置使用 TPT，因此本节参数暂不生效。使用 `--temporal avt` 切换后，`classes` 停止生效，`avt_classes` 开始生效。

### `window_size: 20`

使用该轨迹最近 `20` 个有效检测框的面积。这里只统计成功检测的面积，不把漏检作为面积零加入计算。

### `area_threshold: 0.05`

AVT 使用检测框面积变异系数判断目标是否具有火焰或烟雾的动态变化：

```text
面积变异系数 = 检测框面积的总体标准差 / 检测框平均面积
```

面积变异系数达到 `0.05` 时通过 AVT。

- 调低：对微小面积变化更敏感，但误报风险提高。
- 调高：要求变化更明显，可能漏掉稳定或很小的火苗。

### `min_samples: 2`

至少需要两个有效面积样本才能计算面积变化，避免单个检测框直接通过 AVT。

### `min_track_duration: 0.0`

AVT 不额外要求轨迹持续时间，只要样本数和面积变化满足要求即可通过。

fire 和 smoke 的这四项参数含义相同。

## 7. 忽略区域

```yaml
ignore_regions: []
```

空列表表示当前不忽略任何图像区域。如果固定灯具、指示灯或反光区域长期产生误报，可以按摄像头和类别增加忽略区：

```yaml
ignore_regions:
  - camera_id: camera_001
    class_name: fire
    bbox: [0.0, 0.0, 0.15, 0.20]
    normalized: true
```

- `camera_id`：只对指定摄像头生效。
- `class_name`：只过滤指定类别。
- `bbox`：忽略区域的 `[x1, y1, x2, y2]`。
- `normalized: true`：坐标使用 `0` 至 `1` 的归一化比例，而不是像素值。

当检测框中心点落入忽略区域时，该检测会在进入跟踪器前被过滤。

## 8. 当前 fire 判定逻辑汇总

```text
fire < 0.30
  -> 丢弃，不进入跟踪器

fire >= 0.80
  -> 建立/更新轨迹
  -> TPT 高置信度快速通过
  -> 提交 MViT 验证

0.30 <= fire < 0.80
  -> 建立/更新轨迹
  -> 最近 5 次至少命中 3 次
  -> 窗口内最多漏检 2 次
  -> 轨迹持续至少 0.4 秒
  -> 提交 MViT 验证
```

同一条轨迹只产生一次 TPT/AVT 触发事件；最终是否报警由启用后的 MViT 概率和滑动窗口投票决定。

