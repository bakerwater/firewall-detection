# 基于小模型 + TPT + 大模型的火灾/烟雾检测系统设计方案

## 1. 文档目的

本文档用于设计一个面向视频监控场景的两级火灾/烟雾检测系统。系统以一个已经训练完成的小型 YOLO 模型作为前端快速检测器，在小模型发现疑似火焰或烟雾后，先使用 **TPT（Temporal Persistence Technique，时间持续性分析）** 进行时序过滤，只有通过 TPT 门控的候选事件才会调用大模型进行多帧确认，最终决定是否报警。

整体目标不是让小模型直接判断“是否发生火灾”，而是将不同模块的职责分开：

- **小模型**：快速发现候选目标，强调召回率；
- **TPT**：过滤只出现一帧或短暂出现的误检，控制大模型调用次数；
- **大模型**：分析连续视频片段，确认是否是真实火焰/烟雾；
- **状态机**：管理疑似、验证、报警和恢复过程，避免状态抖动；
- **报警模块**：负责去重、锁存、恢复、重试和日志记录。

本文档参考了 `pedbrgs/Fire-Detection` 项目的混合检测思想。原项目将系统划分为“空间检测”和“时间分析”两个阶段，并提供 YOLOv5 + TPT 的组合；但本文档将 TPT 进一步改造成小模型与大模型之间的触发门控，并补充完整的实时报警状态机。

参考项目：

- [Fire-Detection 项目主页](https://github.com/pedbrgs/Fire-Detection/)
- [原项目 README](https://github.com/pedbrgs/Fire-Detection/blob/main/README.md)
- [原项目 detect.py](https://github.com/pedbrgs/Fire-Detection/blob/main/detect.py)
- [原项目 temporal/tracker.py](https://github.com/pedbrgs/Fire-Detection/blob/main/temporal/tracker.py)

---

## 2. 系统总体架构

```text
                    ┌──────────────────────┐
                    │      视频流输入       │
                    │ 文件 / 摄像头 / RTSP  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     视频采集模块      │
                    │ 时间戳、帧号、断流检测 │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      小模型 YOLO      │
                    │ 低成本、低帧率持续运行 │
                    └──────────┬───────────┘
                               │
                 无候选 ───────┤─────── 有候选
                               │
                               ▼
                    ┌──────────────────────┐
                    │       TPT 时序门控     │
                    │ 持续性、轨迹、类别校验 │
                    └───────┬───────┬──────┘
                            │       │
                  未通过 ───┘       └── 通过
                            │             │
                            ▼             ▼
                      继续监控     ┌───────────────┐
                                   │ 视频片段缓存   │
                                   │ 触发前+触发后   │
                                   └───────┬───────┘
                                           │
                                           ▼
                                   ┌───────────────┐
                                   │    大模型      │
                                   │ 多帧/视频推理  │
                                   └───────┬───────┘
                                           │
                                           ▼
                                   ┌───────────────┐
                                   │  多窗口结果融合 │
                                   └───────┬───────┘
                                           │
                              未确认 ──────┴────── 确认
                                           │             │
                                           ▼             ▼
                                     恢复正常        报警状态
```

---

## 3. 各模块职责

### 3.1 视频采集模块

视频采集模块负责从文件、摄像头或 RTSP 流中获取帧，并为每一帧附加必要信息：

```python
frame_item = {
    "camera_id": "camera_001",
    "frame_id": 12345,
    "timestamp": 1712345678.123,
    "image": frame,
}
```

必须区分以下两种情况：

```text
没有检测到目标 ≠ 没有收到视频帧
```

如果摄像头断流、帧率异常或解码失败，应该进入设备故障状态，而不能直接判断为“当前没有火灾”。

视频采集模块还需要维护一个环形缓冲区，保存最近若干秒的视频帧。

建议：

```text
正常状态缓冲：最近 3～5 秒
疑似状态缓冲：继续以高帧率写入
```

这样在小模型触发时，可以同时获取触发前的视频内容。

### 3.2 小模型检测模块

小模型是系统的前端候选检测器，主要目标是低延迟和高召回率。

小模型可以输出：

```python
small_detection = {
    "class_name": "fire",       # fire 或 smoke
    "confidence": 0.62,
    "bbox": [x1, y1, x2, y2],
    "camera_id": "camera_001",
    "timestamp": 1712345678.123,
}
```

小模型不应直接触发报警，也不建议让单帧结果直接触发大模型。小模型的结果首先要进入 TPT 门控。

建议采用自适应推理频率：

```text
正常监控：1～2 FPS
检测到运动或高风险区域：2～5 FPS
进入 TPT 验证：10～15 FPS
```

如果系统资源有限，可以让视频解码保持较高帧率，但小模型只按低频运行。

### 3.3 目标跟踪模块

TPT 不应该只统计“这一帧有没有检测框”，而应该判断同一个候选目标是否持续存在。因此需要为检测框建立轨迹。

跟踪可以使用：

- 中心点距离匹配；
- IoU 匹配；
- SORT；
- ByteTrack；
- Kalman Filter；
- 其他轻量级多目标跟踪器。

每个目标轨迹至少记录：

```python
track = {
    "track_id": 12,
    "class_name": "fire",
    "first_seen": 1712345670.0,
    "last_seen": 1712345673.0,
    "hit_count": 3,
    "miss_count": 1,
    "confidence_history": [0.42, 0.58, 0.62],
    "bbox_history": [],
    "area_history": [],
    "center_history": [],
}
```

同一个摄像头中的不同目标不能共用一个持续性计数器。

### 3.4 TPT 时序门控模块

TPT 的作用不是最终判断火灾，而是回答：

> 这个小模型检测到的候选目标，是否持续出现到值得调用大模型？

TPT 应当同时考虑：

1. 检测持续次数；
2. 目标持续时间；
3. 目标轨迹是否连续；
4. 目标类别是否稳定；
5. 置信度是否达到最低要求；
6. 是否位于忽略区域；
7. 是否存在视频断帧或异常。

推荐的基本判断逻辑：

```text
最近 L 次小模型检测中，至少 M 次检测到同一目标
并且目标轨迹持续时间超过最小持续时间
并且目标不在忽略区域
→ 允许调用大模型
```

例如：

```text
window_size = 4
min_hits = 2
min_track_duration = 1.0 秒
```

表示最近 4 次小模型采样中，至少 2 次检测到同一个候选目标，并且该目标至少持续 1 秒。

### 3.5 大模型验证模块

大模型负责对连续视频片段进行进一步判断。大模型输入可以包含：

- 触发前视频帧；
- 触发后视频帧；
- 疑似目标裁剪区域；
- 原始全画面；
- 小模型检测框和置信度；
- 目标轨迹信息。

推荐输入形式：

```text
视频片段长度：2～4 秒
输入帧数：16～32 帧
滑动步长：0.5～1 秒
```

如果大模型支持视频输入，直接输入固定长度的帧序列。如果大模型只能处理图片，则可以对多帧分别推理，再进行时间投票，但这种方式的时序建模能力相对有限。

### 3.6 结果融合和报警模块

大模型一次输出不能直接作为最终报警条件。系统应该对多个滑动窗口进行统计。

例如：

```text
Clip 1：0.61
Clip 2：0.78
Clip 3：0.83
```

可以使用：

```text
最近 3 个窗口中至少 2 个为阳性
```

或者：

```text
大模型平均置信度超过阈值
并且至少有一个窗口为高置信度阳性
```

最终报警模块负责：

- 报警去重；
- 报警锁存；
- 保存报警视频；
- 发送消息；
- 失败重试；
- 报警确认；
- 报警恢复；
- 记录完整事件日志。

---

## 4. TPT 详细流程

### 4.1 NORMAL：正常监控状态

系统在 NORMAL 状态下只运行小模型，不调用大模型。

处理流程：

```text
1. 读取一帧视频；
2. 写入环形缓冲区；
3. 根据调度器决定是否执行小模型；
4. 如果没有检测到目标，继续监控；
5. 如果检测到候选目标，创建或更新目标轨迹；
6. 对目标执行 TPT 判断。
```

### 4.2 CANDIDATE：候选目标状态

当小模型首次检测到目标时，系统进入 CANDIDATE 状态，但暂时不调用大模型。

系统需要：

```text
1. 记录首次检测时间；
2. 保存候选框和置信度；
3. 与历史轨迹进行匹配；
4. 等待后续若干次小模型结果；
5. 统计命中次数和丢失次数；
6. 判断是否达到 TPT 门控条件。
```

如果候选目标只出现一次，然后消失：

```text
清除候选轨迹，回到 NORMAL。
```

如果候选目标持续出现并通过 TPT：

```text
进入 VERIFYING，调用大模型。
```

### 4.3 VERIFYING：大模型验证状态

进入 VERIFYING 后：

```text
1. 提升采样率；
2. 从环形缓冲区获取触发前帧；
3. 继续采集触发后的高频帧；
4. 生成一个或多个视频片段；
5. 异步提交给大模型；
6. 统计大模型输出；
7. 判断确认、失败或超时。
```

建议第一次验证时间设置为 3～5 秒。

如果大模型结果不确定，但小模型仍在稳定检测，可以有限延长验证时间：

```text
首次验证：5 秒
最多延长：5 秒
最大验证时间：10 秒
```

不能无限等待，否则单个摄像头可能长期占用验证资源。

### 4.4 ALARM：报警状态

满足确认条件后，系统进入 ALARM：

```text
1. 生成唯一事件 ID；
2. 发送报警；
3. 保存触发前后视频；
4. 记录摄像头、类别、置信度和位置；
5. 继续运行模型监控；
6. 防止同一事件重复报警。
```

报警不能因为单帧阴性立即解除。

### 4.5 RECOVERING：恢复观察状态

报警解除需要满足持续阴性条件：

```text
小模型连续阴性
并且大模型最近多个窗口阴性
并且持续时间超过清除阈值
```

建议：

```text
连续 5～10 秒阴性后进入 RECOVERING
RECOVERING 再持续 2～5 秒后回到 NORMAL
```

如果恢复期间再次检测到疑似目标：

```text
立即回到 ALARM 或 VERIFYING。
```

---

## 5. TPT 门控规则设计

### 5.1 基础规则

推荐先实现一个清晰、可解释的规则版本：

```python
should_verify = (
    hit_count >= min_hits
    and track_duration >= min_track_duration
    and confidence_max >= min_confidence
    and miss_count <= max_misses
    and not in_ignore_region
)
```

推荐的初始参数可以是：

```text
small_conf_candidate：0.25～0.40
small_conf_high：0.75～0.85
window_size：4～6 次采样
min_hits：2～3 次
min_track_duration：1～2 秒
max_misses：1～2 次
```

这些值只是启动实验的建议，不能替代验证集调参。

### 5.2 高置信度快速通道

为了避免小模型漏掉突然出现的真实火焰，可以保留一个高置信度快速通道：

```text
如果小模型置信度非常高：
    跳过严格 TPT，直接调用大模型
```

这样系统同时具备：

```text
普通候选：严格时序门控
高置信度候选：快速验证
```

### 5.3 火焰和烟雾分别处理

火焰和烟雾的时间特征不同，不建议完全使用同一套参数。

火焰：

```text
可以较快触发验证；
重点关注闪烁、亮度和边界变化。
```

烟雾：

```text
允许更长观察时间；
重点关注扩散、形状和持续性。
```

示例：

```text
fire：最近 3 次中至少 2 次，持续 1 秒
smoke：最近 5 次中至少 3 次，持续 2 秒
```

---

## 6. 推荐的数据流和伪代码

```python
state = "NORMAL"

while True:
    frame_item = capture.read()

    if frame_item is None:
        state = "FAULT"
        handle_camera_fault()
        continue

    ring_buffer.append(frame_item)

    if state == "NORMAL":
        if small_scheduler.should_run(frame_item.timestamp):
            detections = small_model.predict(frame_item.image)
            tracks = tracker.update(detections, frame_item.timestamp)

            candidates = filter_candidate_tracks(tracks)

            for track in candidates:
                if tpt_gate_pass(track):
                    state = "VERIFYING"
                    event_id = create_event(track)

                    clip_builder.start(
                        event_id=event_id,
                        prebuffer=ring_buffer.get(seconds=2),
                        target_track=track,
                    )

                    verifier_queue.submit(event_id)
                    break

    elif state == "VERIFYING":
        clip_builder.append(frame_item)

        if clip_builder.has_ready_clip():
            clip = clip_builder.get_clip()
            big_result = large_model.predict(clip)
            event_manager.update_big_result(event_id, big_result)

        if event_manager.is_confirmed(event_id):
            state = "ALARM"
            alarm_manager.send_once(event_id)
            event_manager.save_event_video(event_id)

        elif event_manager.is_timeout(event_id):
            if event_manager.small_target_still_exists(event_id):
                event_manager.extend_once(event_id)
            else:
                state = "NORMAL"
                event_manager.close(event_id, reason="verification_failed")

    elif state == "ALARM":
        clip_builder.append(frame_item)

        if event_manager.clear_condition_met(event_id):
            state = "RECOVERING"
            event_manager.start_recovery(event_id)

    elif state == "RECOVERING":
        if event_manager.new_positive_detected(event_id):
            state = "ALARM"
        elif event_manager.recovery_finished(event_id):
            state = "NORMAL"
            event_manager.close(event_id, reason="cleared")

    elif state == "FAULT":
        if camera_recovered():
            state = "NORMAL"
```

---

## 7. 与原 Fire-Detection 项目的关系

原项目的设计是：

```text
YOLOv5 检测候选区域
        ↓
AVT 或 TPT 时间分析
        ↓
过滤检测框
```

其中，README 将 TPT 推荐用于室内场景，并提供 `window-size` 和 `persistence-thresh` 等参数；代码中也确实在 YOLO 检测结果后执行 persistence 逻辑。原项目同时支持单独运行 YOLOv5，或运行 YOLOv5 + 时间分析组合。([README](https://github.com/pedbrgs/Fire-Detection/blob/main/README.md))

本文方案对原逻辑进行了扩展：

```text
YOLOv5 小模型
        ↓
TPT 时序门控
        ↓
大模型多帧验证
        ↓
多窗口融合
        ↓
报警状态机
```

因此，原项目中的 TPT 可以作为本系统的基础模块，但不能原样视为完整的“大模型触发系统”。原项目的 `detect.py` 主要对当前检测结果进行时序抑制，并没有在 TPT 通过后自动调用另一个大模型；大模型调用、视频片段缓存和报警状态机需要另行实现。([detect.py](https://github.com/pedbrgs/Fire-Detection/blob/main/detect.py))

---

## 8. 对原 TPT 实现的适配注意事项

### 8.1 不要只使用一个全局缓冲区

多摄像头系统中，TPT 状态必须按摄像头区分：

```python
camera_states[camera_id]
```

如果还需要区分目标，则进一步按轨迹区分：

```python
camera_states[camera_id][track_id]
```

不能让摄像头 A 的检测结果影响摄像头 B 的持续性计数。

### 8.2 不要只统计“任意检测框”

TPT 应该统计：

```text
同一摄像头
同一类别
同一目标轨迹
```

否则以下情况可能被错误累计：

```text
第 1 帧左侧检测到灯光
第 2 帧右侧检测到反光
第 3 帧中间检测到烟雾
```

虽然不是同一个目标，但全局计数可能认为目标持续存在。

### 8.3 使用真实时间而不是只使用帧数量

如果摄像头帧率变化，固定帧数不一定等于固定时间。建议同时记录时间戳：

```python
if current_timestamp - track.first_seen >= min_duration:
    duration_pass = True
```

### 8.4 给检测丢失设置容忍时间

火焰和烟雾边界不稳定，不能一帧没有检测到就立刻删除轨迹。

建议：

```text
允许连续丢失 0.5～1 秒
```

超过容忍时间后再删除候选轨迹。

### 8.5 TPT 通过后要有防重复机制

进入 VERIFYING 后，如果小模型继续检测到同一个目标，不应重复创建大模型任务。

建议：

```text
同一 camera_id + track_id 只允许一个验证任务
```

---

## 9. 系统优势

### 9.1 降低大模型调用次数

小模型只负责发现候选目标，TPT 会过滤掉大量单帧误检和短时误检，因此大模型不需要处理所有帧或所有候选。

这可以降低：

- GPU 使用率；
- 大模型推理次数；
- 系统延迟；
- 多摄像头部署成本；
- 大模型任务队列压力。

### 9.2 兼顾召回率和精度

小模型可以使用相对宽松的阈值保证召回率，TPT 负责筛掉明显的瞬时误报，大模型再负责最终确认。

与单一模型相比，系统可以分工：

```text
小模型：不轻易漏掉候选
TPT：过滤短暂异常
大模型：提升最终精度
```

### 9.3 充分利用视频时序信息

火焰和烟雾不是静态目标。通过持续性、目标轨迹、检测框变化和连续视频片段，可以利用单帧图片无法提供的信息。

### 9.4 可解释性较好

系统可以解释报警原因：

```text
小模型连续检测 4 次
目标轨迹持续 2.1 秒
大模型 3 个窗口中 2 个阳性
最终进入报警状态
```

这比直接输出一个无法解释的报警概率更适合工程部署。

### 9.5 计算资源可以动态分配

正常情况下只运行小模型；疑似事件出现时，才临时提升采样率并调用大模型，适合边缘设备和多摄像头系统。

### 9.6 便于逐步迭代

系统可以按以下顺序逐步实现：

```text
第一阶段：小模型 + TPT
第二阶段：加入大模型验证
第三阶段：加入目标跟踪
第四阶段：加入多摄像头融合
第五阶段：加入学习型分数融合
```

---

## 10. 系统不足和潜在风险

### 10.1 小模型漏检无法被后续模块弥补

如果小模型完全没有检测到目标，TPT 和大模型都不会被触发。

这叫做级联系统的“前级漏检传播”问题：

```text
小模型漏检
    ↓
TPT 没有候选
    ↓
大模型没有机会判断
```

解决方法：

- 提高小模型在关键场景的召回率；
- 运动区域提高小模型频率；
- 对高风险区域定期调用大模型；
- 使用光流、帧差或亮度异常作为额外触发器；
- 对小模型设置高置信度快速通道和低置信度时序通道。

### 10.2 TPT 可能导致检测延迟

TPT 需要等待多个时间点来确认目标，因此报警可能比单帧检测慢几百毫秒到几秒。

解决方法：

- 使用高置信度快速通道；
- 火焰和烟雾分别设置持续时间；
- 限制 TPT 最大等待时间；
- 对高风险区域使用更高采样率。

### 10.3 TPT 可能过滤掉短暂但真实的火焰

有些火焰可能只出现很短时间。如果要求连续多帧出现，可能造成漏报。

解决方法：

```text
高置信度检测 → 跳过严格 TPT，直接送大模型
```

同时保留触发前视频缓冲，交给大模型判断是否是真实事件。

### 10.4 固定阈值不适合所有摄像头

不同摄像头存在以下差异：

- 分辨率不同；
- 视角不同；
- 夜间噪声不同；
- 光照不同；
- 背景复杂度不同；
- 目标大小不同。

因此应该支持按摄像头配置：

```yaml
camera_001:
  candidate_threshold: 0.30
  min_hits: 2
  window_size: 4
  clear_seconds: 8

camera_002:
  candidate_threshold: 0.40
  min_hits: 3
  window_size: 5
  clear_seconds: 10
```

### 10.5 大模型依然可能误报

TPT 只能减少短暂误报，不能保证大模型一定正确。大模型仍可能把以下内容识别成火灾：

- 红色灯光；
- 夕阳；
- 车辆尾灯；
- 蒸汽和雾气；
- 屏幕播放的火焰视频；
- 反光和镜面高亮；
- 施工粉尘。

需要在大模型训练数据中加入困难负样本，并使用真实监控场景进行验证。

### 10.6 多帧输入要求数据质量稳定

如果视频存在丢帧、时间戳错误、帧顺序混乱或不同摄像头帧混用，大模型的时序判断会受到影响。

必须记录：

```text
camera_id
frame_id
timestamp
```

生成视频片段时需要按照时间戳排序并检查帧间隔。

### 10.7 多摄像头会增加系统复杂度

多摄像头系统需要解决：

- 每个摄像头独立状态；
- GPU 推理任务调度；
- 同一事件的报警合并；
- 摄像头断流处理；
- 不同摄像头阈值配置；
- 事件视频的存储管理。

不能简单地把所有摄像头的检测结果放入同一个 TPT 缓冲区。

### 10.8 报警系统需要考虑工程可靠性

模型检测正确不代表报警一定成功。实际部署还要考虑：

- 网络断开；
- 消息发送失败；
- 报警重复发送；
- 服务器重启；
- 视频保存失败；
- GPU 显存不足；
- 摄像头时间不同步。

报警发送应该支持唯一事件 ID、重试和幂等处理。

---

## 11. 推荐的初始参数

以下参数可以作为第一版实验的起点：

```yaml
# 正常状态
small_model_fps: 1

# TPT
window_size: 4
min_hits: 2
min_track_duration: 1.0
max_misses: 1

# 大模型验证
verification_fps: 10
clip_seconds: 3
clip_frames: 16
clip_stride_seconds: 0.5
verification_timeout_seconds: 5
max_verification_seconds: 10

# 确认报警
big_positive_windows: 2
big_window_count: 3

# 恢复
clear_negative_seconds: 8
recovery_seconds: 3
alarm_cooldown_seconds: 30
```

注意：这些参数不是通用标准，只是用于启动实验。最终应通过包含正常场景、困难负样本、真实火灾和烟雾的视频验证集进行调整。

---

## 12. 评估指标

不能只看单帧 mAP 或准确率，应该按照事件级别评估整个系统。

### 12.1 小模型指标

- Precision；
- Recall；
- mAP；
- 火焰和烟雾分别的 AP；
- 不同光照条件下的召回率。

### 12.2 TPT 门控指标

- 每小时触发大模型的次数；
- TPT 过滤掉的误报比例；
- TPT 引入的漏报数量；
- TPT 平均延迟；
- 不同窗口大小下的效果。

### 12.3 大模型指标

- 视频片段级 Precision；
- 视频片段级 Recall；
- 确认延迟；
- 误报警率；
- 漏报警率。

### 12.4 系统级指标

- 每个摄像头每天误报次数；
- 每小时大模型调用次数；
- 平均报警延迟；
- 报警事件召回率；
- 同一事件重复报警率；
- GPU 和 CPU 使用率；
- 断流恢复时间。

最重要的三个工程指标是：

```text
每小时误报次数
每个摄像头每天大模型调用次数
真实事件的平均报警延迟
```

---

## 13. 最终推荐逻辑

最终建议采用以下规则：

```text
1. 正常状态下，小模型低帧率运行。

2. 小模型检测到候选目标后，不立即调用大模型。

3. 为候选目标建立轨迹，并使用 TPT 统计：
   - 是否连续出现；
   - 是否属于同一目标；
   - 是否达到最小持续时间；
   - 是否处于允许检测区域。

4. 只有 TPT 通过后，才启动高频采样和大模型验证。

5. 大模型使用触发前和触发后的视频片段进行多帧推理。

6. 大模型需要多个滑动窗口共同确认，不使用单次结果报警。

7. 报警后采用锁存和迟滞机制，不能因为一帧阴性立即恢复。

8. 同一摄像头、同一目标、同一事件只能创建一个验证任务。

9. 摄像头断流、模型异常和帧不足必须单独处理。

10. 按摄像头和类别分别调节阈值，不能所有场景使用同一组参数。
```

最终系统可以概括为：

```text
小模型负责“发现”
TPT负责“判断是否值得进一步确认”
大模型负责“确认”
状态机负责“稳定地报警和恢复”
```

这种设计能够减少小模型误报导致的大模型频繁调用，同时保留小模型的高召回能力，是比“小模型一检测到就调用大模型”更适合实时部署的方案。
