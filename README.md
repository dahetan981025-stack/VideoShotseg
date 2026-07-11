# VideoShotseg — 视频镜头分割工具

基于聚类分析的视频镜头切分工具。将视频帧 embedding 通过 centroid 相似度曲线 + 自适应阈值检测镜头边界，支持 FFmpeg 视频处理。

## 功能

- 🎬 **FFmpeg 集成** — 自动提取视频帧、生成 embedding
- 🔪 **镜头边界检测** — 基于 centroid 聚类 + 自适应阈值
- 🧩 **场景合并** — 将相似镜头合并为语义场景
- 📊 **CLI 接口** — 命令行一键操作
- 🔌 **Python API** — 可直接集成到其他项目

## 安装

```bash
pip install -r requirements.txt
```

## 使用

### CLI 方式

```bash
# 直接处理视频文件
python -m shotseg video.mp4 --show

# 使用已有的 embedding 文件
python -m shotseg embeddings.npz --show
python -m shotseg embeddings.npz -o result.json
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--window` | 40 | centroid 滑动窗口大小（帧数） |
| `--k` | 1.8 | 自适应阈值倍数 |
| `--min-gap` | 3.0s | 强制切分的最小间隔 |
| `--scene-sim` | 0.85 | 场景合并相似度阈值 |
| `--scene-dur` | 0.8s | 最小场景时长 |

### Python API

```python
from shotseg import ShotSeg

seg = ShotSeg()
result = seg.segment(embeddings, timestamps)

print(result.summary())
for scene in result.scenes:
    print(f'Scene {scene.scene_id}: {scene.start_time:.1f}s ~ {scene.end_time:.1f}s')
```

## 输入格式

`.embeddings.npz` 文件需包含：
- `embeddings`: shape `(n_frames, dim)` 的 float64 数组
- `timestamps`: shape `(n_frames,)` 的时间戳数组

## 项目结构

```
shotseg/
├── __init__.py      # 包入口
├── __main__.py      # CLI 接口
├── types.py         # 数据结构
├── detection.py     # 镜头边界检测算法
├── clustering.py    # 聚类分析
├── merge.py         # 场景合并
└── pipeline.py      # 主流程
```

## License

MIT
