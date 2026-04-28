# README

## Machine-learning electron spectra workspace

New standalone folder:

- `machine_learning/download_electron_spectra.py`
  - Downloads MAVEN SWE `svypad` electron spectra by calling `download_maven_data.py`.
- `machine_learning/analyze_electron_spectra_ml.py`
  - Loads SWE electron spectra, normalizes them, clusters characteristic spectral shapes, and reports the nearest real timestamp for each cluster.
- `machine_learning/README.md`
  - Contains command examples and output descriptions.

Example:

```bash
python machine_learning/download_electron_spectra.py --year 2024
python machine_learning/analyze_electron_spectra_ml.py --start 2024-11-07T00:00:00 --end 2024-11-08T00:00:00 --auto-clusters
```

## 1. 维护指令

每次更新代码的同时，必须同步更新此 `README.md` 文件。

更新时至少检查以下内容：

- 是否新增、删除或重命名了脚本/网页文件
- 是否修改了输入参数、默认配置或输出目录
- 是否改变了下载逻辑、拓扑判别规则或可视化逻辑
- 是否需要在“更新日志”中新增一条记录

---

## 2. 使用说明

### 2.1 项目目标

本项目用于处理 MAVEN 的电子能谱与磁场数据，完成以下工作：

- 下载指定日期的 MAVEN 数据
- 提取指定时刻的电子能谱和磁场信息
- 在一段时间范围内根据电子能谱特征判断磁场拓扑结构
- 输出轨迹图、统计结果和 JSON 汇总文件
- 在本地网页中加载结果并进行三维可视化

当前默认使用的数据产品为：

- `SWEA svypad`
  - 用于电子能谱和 pitch angle 分析
- `STATIC c6-32e64m`
  - 用于离子能量谱和质量谱上下文面板
- `MAG ss1s`
  - 用于 `SS / MSO` 坐标系下的磁场和位置
- `MAG pc1s`
  - 用于 `PC / 火星固连` 坐标系下的位置

### 2.2 坐标系说明

当前项目中和可视化直接相关的坐标系有两套：

- `SS / MSO`
  - `Sun-State`
  - 用于当前默认磁场与位置分析
  - 更适合从太阳风和外部空间环境角度观察轨迹
- `PC`
  - `Planetocentric`
  - 火星固连坐标系
  - 更适合从火星本体、经纬度和固定表面区域角度观察轨迹

当前分析脚本会在结果 JSON 中同时写出两套位置：

- `positions_by_frame_km["ss"]`
- `positions_by_frame_km["pc"]`
- `positions_by_frame_rm["ss"]`
- `positions_by_frame_rm["pc"]`
- `altitude_km`
- `altitude_rm`

为了兼容旧逻辑，脚本仍保留：

- `position_km`
- `position_rm`

这两个字段默认对应 `SS / MSO`。

### 2.3 目录结构

主要目录如下：

- [data](/g:/本研/科学/MARS/ML/maven_code_linux/data)
  - 存放下载得到的 MAVEN 原始数据文件
- [outputs](/g:/本研/科学/MARS/ML/maven_code_linux/outputs)
  - 存放处理后的图像、JSON 汇总和分析结果

主要文件如下：

- [download_maven_data.py](/g:/本研/科学/MARS/ML/maven_code_linux/download_maven_data.py)
- [process_maven_spectra.py](/g:/本研/科学/MARS/ML/maven_code_linux/process_maven_spectra.py)
- [run_maven_pipeline.py](/g:/本研/科学/MARS/ML/maven_code_linux/run_maven_pipeline.py)
- [analyze_magnetic_topology.py](/g:/本研/科学/MARS/ML/maven_code_linux/analyze_magnetic_topology.py)
- [magnetic_topology_viewer.html](/g:/本研/科学/MARS/ML/maven_code_linux/magnetic_topology_viewer.html)
- [photoelectron_and_magnetism_topology.txt](/g:/本研/科学/MARS/ML/maven_code_linux/photoelectron_and_magnetism_topology.txt)
- [maven_variable_summary.csv](/g:/本研/科学/MARS/ML/maven_code_linux/maven_variable_summary.csv)

### 2.4 环境准备

运行 Python 脚本前，建议准备好以下依赖：

- `python`
- `requests`
- `numpy`
- `matplotlib`
- `cdflib`

如果需要下载新数据，需要本机可以访问 LASP 的数据接口。

### 2.5 各个 Python 脚本的作用

#### [download_maven_data.py](/g:/本研/科学/MARS/ML/maven_code_linux/download_maven_data.py)

作用：

- 根据目标日期访问 MAVEN LASP 文件接口
- 自动查找当天最合适的 `SWE svypad`、`STATIC c6-32e64m`、`MAG ss1s` 和 `MAG pc1s` 文件
- 将文件下载到本地 `data/maven/...` 目录

命令示例：

```bash
python download_maven_data.py --time 2024-11-07T02:15:00
```

输出：

- 下载后的数据文件保存在 `data/maven/`

#### [process_maven_spectra.py](/g:/本研/科学/MARS/ML/maven_code_linux/process_maven_spectra.py)

作用：

- 读取单个 SWE PAD 文件
- 提取目标时刻前向和平行/反平行方向的电子能谱
- 读取同一时刻附近的 MAG 磁场矢量
- 输出单时刻谱图和 `spectrum_summary.json`

命令示例：

```bash
python process_maven_spectra.py --time 2024-11-07T02:15:00
```

输出位置：

- `outputs/maven_spectra/<timestamp>/directional_electron_spectra.png`
- `outputs/maven_spectra/<timestamp>/spectrum_summary.json`

#### [run_maven_pipeline.py](/g:/本研/科学/MARS/ML/maven_code_linux/run_maven_pipeline.py)

作用：

- 作为单时刻批处理入口
- 先下载目标时刻对应的数据
- 再调用 `process_maven_spectra.py` 逐个处理

你主要修改的位置：

- 文件内部的 `CONFIG["target_times"]`

运行方式：

```bash
python run_maven_pipeline.py
```

输出位置：

- `outputs/maven_spectra/`
- `outputs/maven_spectra/pipeline_summary.json`

#### [analyze_magnetic_topology.py](/g:/本研/科学/MARS/ML/maven_code_linux/analyze_magnetic_topology.py)

作用：

- 在一个时间范围内分析电子能谱
- 根据预设的光电子特征判断磁场拓扑结构
- 读取 `MAG ss1s` 中的磁场
- 同时读取 `MAG ss1s` 和 `MAG pc1s` 中的卫星位置
- 读取 `STATIC c6-32e64m` 生成离子能量谱和质量谱上下文数据
- 从 `SWE svypad` 生成 SWE PAD 和电子能谱上下文数据
- 输出轨迹图和 `topology_summary.json`

你最常修改的位置有两处：

1. 顶部 `CONFIG`
   - `start_time`
   - `end_time`
   - `step_seconds`
   - `auto_download_missing_data`
   - `data_root`
   - `output_root`

2. 顶部 `TOPOLOGY_RULES`
   - `pitch_angle`
   - `knee`
   - `auger_peak`
   - `dropoff`
   - `classification`

判别逻辑概念如下：

- `knee`
  - 检查 `50-70 eV` 附近是否出现明显拐点
- `auger_peak`
  - 检查 `180-320 eV` 附近是否存在相对左右肩更高的局部峰
- `dropoff`
  - 检查高能段是否快速衰减
- `classification`
  - 如果前向与后向都达到足够特征分数，则判为 `closed`
  - 如果前向与后向都只有较少特征，则判为 `open`
  - 其余情况判为 `ambiguous`

脚本还有两个重要功能：

- 当请求时间范围超出本地已下载数据范围时
  - 如果 `auto_download_missing_data=True`
  - 会自动下载缺失的 `svypad`、`c6-32e64m`、`ss1s` 和 `pc1s`
- 输出的 JSON 同时包含 `SS / MSO` 和 `PC` 两套位置
  - 方便网页可视化切换坐标系
- 输出的 JSON 还包含 `context_overview`
  - 供网页在点击轨迹点时显示前后 10 分钟的上下文面板

运行方式 1：直接使用文件顶部配置

```bash
python analyze_magnetic_topology.py
```

运行方式 2：命令行覆盖配置

```bash
python analyze_magnetic_topology.py --start 2024-11-07T00:00:00 --end 2024-11-08T00:00:00 --step-seconds 120
```

常用参数：

- `--start`
- `--end`
- `--step-seconds`
- `--data-root`
- `--output-root`
- `--no-auto-download`

输出位置：

- `outputs/magnetic_topology/<start>_<end>/magnetic_topology_trajectory.png`
- `outputs/magnetic_topology/<start>_<end>/topology_summary.json`

#### [plot_maven_orbit_map.py](/g:/本研/科学/MARS/ML/maven_code_linux/plot_maven_orbit_map.py)

作用：
- 绘制 MAVEN ground track 与火星壳磁场背景图
- ground track 使用 `MAG pc1s` 中的 planetocentric 位置
- 壳磁场背景使用 `Morschhauser et al. (2014)` 球谐模型
- 默认绘制 `185 km` 高度处的 `|B|`
- 默认经纬度网格步长为 `2°`

普通运行方式：
```bash
python plot_maven_orbit_map.py --time 2024-11-07T02:15:00
```

常用参数：
```bash
python plot_maven_orbit_map.py --time 2024-11-07T02:15:00 --window-minutes 20 --crustal-altitude-km 185 --grid-step-deg 2 --model-max-degree 60
```

为了避免每次重复计算壳磁场背景，脚本会优先读取预计算缓存：
- `data/models/mars_crustal/precomputed/morschhauser2014_alt185p000km_step2p000deg_deg60_lon000_180.npz`
- `data/models/mars_crustal/precomputed/morschhauser2014_alt185p000km_step2p000deg_deg60_lon180_360.npz`

如果缓存不存在，脚本会自动计算并保存。也可以手动预处理两个经度窗口：
```bash
python plot_maven_orbit_map.py --precompute-crustal-cache --crustal-altitude-km 185 --grid-step-deg 2 --model-max-degree 60
```

输出 summary 中的 `crustal_cache_hit` 可用于确认是否复用了缓存。

#### [plot_maven_data_panels.py](/g:/本研/科学/MARS/ML/maven_code_linux/plot_maven_data_panels.py)

作用：
- 从 `topology_summary.json` 渲染静态数据面板 PNG
- 面板采用单列纵向布局，便于上下对比同一时间窗口内的不同物理量
- 前 8 个时间相关面板共享同一 x 轴范围
- 每个时间相关面板用黑色竖虚线标记传入的 target time

运行方式：
```bash
python plot_maven_data_panels.py --summary-json outputs/magnetic_topology/20241107T020000_20241107T030000/topology_summary.json --time 2024-11-07T02:15:00 --window-minutes 20
```

输出：
- `outputs/maven_data_panels.png`

#### [run_maven_event_figures.py](/g:/本研/科学/MARS/ML/maven_code_linux/run_maven_event_figures.py)

作用：
- 单个事件时间的总控入口
- 自动检查所需 MAVEN 数据
- 生成 directional electron spectra
- 生成 orbit/crustal-field map
- 生成 topology context
- 生成单列 data panels
- 汇总输出 `event_pipeline_summary.json`

直接使用默认配置运行：
```bash
python run_maven_event_figures.py
```

指定事件时间运行：
```bash
python run_maven_event_figures.py --time 2015-01-05T00:30:00
```

推荐的快速本地运行方式：
```bash
python run_maven_event_figures.py --time 2015-01-05T00:30:00 --window-minutes 10 --step-seconds 60 --no-auto-download
```

说明：
- 如果本地数据已经齐全，建议加 `--no-auto-download`，避免网络检查或下载导致运行变慢。
- 壳磁场背景图会复用 `plot_maven_orbit_map.py` 的预计算缓存。
- `analyze_interval()` 仍会生成完整 `context_overview`，包括沿 MAVEN 轨道的 `model_b_mso`，因此总控脚本比单独画 orbit map 更耗时。

输出位置：
- `outputs/maven_event_figures/<timestamp>/directional_electron_spectra.png`
- `outputs/maven_event_figures/<timestamp>/orbit_crustal_map.png`
- `outputs/maven_event_figures/<timestamp>/maven_data_panels.png`
- `outputs/maven_event_figures/<timestamp>/event_pipeline_summary.json`
- `outputs/maven_event_figures/<timestamp>/topology_context/<start>_<end>/topology_summary.json`

#### [magnetic_topology_viewer.html](/g:/本研/科学/MARS/ML/maven_code_linux/magnetic_topology_viewer.html)

作用：

- 在本地离线打开
- 加载 `topology_summary.json`
- 在三维火星球面上显示轨迹和拓扑颜色
- 支持在 `SS / MSO` 和 `PC / 火星固连` 坐标系之间切换
- 点击轨迹点查看对应时刻的时间、位置、磁场和特征分数
- 点击轨迹点后显示该点前后 10 分钟的上下文面板：
  - `STATIC Energy`
  - `STATIC Mass`
  - `|B|`
  - `B_MSO`
  - `Model B_MSO` 占位说明
  - `SWE PAD (111-140 eV)`
  - `SWE Energy`

特点：

- 不依赖外部网页库
- 可以直接本地双击打开
- 适合快速可视化结果

使用步骤：

1. 先运行 `analyze_magnetic_topology.py` 生成结果
2. 打开 `magnetic_topology_viewer.html`
3. 选择 `topology_summary.json`
4. 点击“加载”
5. 在页面左侧选择 `SS / MSO` 或 `PC / 火星固连`

交互方式：

- 左键拖拽旋转
- 滚轮缩放
- 点击点查看详情
- “重置视角”按钮恢复默认视角

### 2.6 文本和表格文件的作用

#### [photoelectron_and_magnetism_topology.txt](/g:/本研/科学/MARS/ML/maven_code_linux/photoelectron_and_magnetism_topology.txt)

作用：

- 记录光电子谱特征与磁场拓扑之间关系的说明
- 是 `analyze_magnetic_topology.py` 中判别规则的物理依据文本

#### [maven_variable_summary.csv](/g:/本研/科学/MARS/ML/maven_code_linux/maven_variable_summary.csv)

作用：

- 记录 MAVEN 变量信息汇总
- 适合查阅变量名、产品结构和字段内容

### 2.7 推荐使用流程

如果你要分析一个新的时间范围，推荐按下面顺序操作：

1. 先在 [analyze_magnetic_topology.py](/g:/本研/科学/MARS/ML/maven_code_linux/analyze_magnetic_topology.py) 中修改 `CONFIG`
2. 运行：

```bash
python analyze_magnetic_topology.py
```

3. 到 `outputs/magnetic_topology/...` 查看：
   - `magnetic_topology_trajectory.png`
   - `topology_summary.json`
4. 打开 [magnetic_topology_viewer.html](/g:/本研/科学/MARS/ML/maven_code_linux/magnetic_topology_viewer.html)
5. 加载对应的 `topology_summary.json`
6. 在网页中切换 `SS / MSO` 或 `PC / 火星固连` 查看轨迹

如果你只想看某一个时刻的电子谱，推荐：

1. 修改 [run_maven_pipeline.py](/g:/本研/科学/MARS/ML/maven_code_linux/run_maven_pipeline.py) 中的 `CONFIG["target_times"]`
2. 运行：

```bash
python run_maven_pipeline.py
```

### 2.8 当前已知限制

- 当前网页支持切换 `SS / MSO` 与 `PC`
- 还不支持真正的日心惯性坐标系
- 拓扑判别目前是规则法
  - 不是机器学习模型
  - 阈值需要结合样本继续调整
- 网页可视化当前是轻量离线版
  - 适合本地查看
  - 不包含高精度火星表面纹理或真实地形

---

## 3. 更新日志

### 2026-03-27

- 新增 `README.md`
- 补充项目维护规则，要求每次代码更新同步更新 README
- 整理当前项目的使用方法、输入输出和脚本职责说明

### 2026-03-17 之前的现有功能概览

- 已支持 MAVEN `SWE svypad` 与 `MAG ss1s` 数据下载
- 已支持单时刻电子谱处理与输出
- 已支持时间区间内磁场拓扑判别
- 已支持自动下载缺失日期的数据
- 已支持本地离线三维网页可视化

### 2026-03-27 坐标系扩展

- `download_maven_data.py` 现已同时支持下载 `MAG pc1s`
- `analyze_magnetic_topology.py` 现已在输出 JSON 中同时写入 `SS / MSO` 和 `PC` 两套位置
- `magnetic_topology_viewer.html` 新增坐标系切换，可在 `SS / MSO` 与 `PC / 火星固连` 之间切换轨迹显示

### 2026-03-27 网页交互修正

- 调整 `magnetic_topology_viewer.html` 的默认缩放和最小缩放距离，使火星可放大到更适合观察细节的程度
- 修复轨迹绘制顺序问题，去掉因线段连接顺序错误导致的棕色阴影/错误连线效果

### 2026-03-27 上下文面板扩展

- `download_maven_data.py` 新增 `STATIC c6-32e64m` 下载
- `analyze_magnetic_topology.py` 新增 `context_overview` 输出，用于保存整段时间内的 STATIC、MAG、SWE 上下文数据
- `magnetic_topology_viewer.html` 新增点击轨迹点后的 10 分钟上下文面板显示
- `Model B_MSO` 目前仍为占位说明，待确定具体壳场模型后再接入

### 2026-03-27 Morschhauser 模型接入

- 新增 `mars_crustal_model.py`，用于下载和读取 `Morschhauser2014` 系数文件
- 新增基于 Mars body-fixed 到 MSO 转换的模型磁场计算流程
- `analyze_magnetic_topology.py` 现可在 `context_overview.model_b_mso` 中输出 `Morschhauser et al. (2014)` 模型结果
- `magnetic_topology_viewer.html` 现可在 `(e) Model B_MSO` 面板显示模型 `Bx / By / Bz`

### 2026-03-27 STATIC 与模型解析修复

- 修复 `STATIC c6-32e64m` 维度解释错误，现在按真实结构 `(time, mass, energy)` 生成 `STATIC Energy` 和 `STATIC Mass` 面板
- 修复 `Morschhauser2014` 系数文件解析逻辑，现在支持文件中的 `m < 0` 表示 `h(n, |m|)` 的格式
- 修复 JSON 清洗逻辑，避免 `NaN` 导致网页加载失败，并保留布尔值语义

### 2026-03-27 Viewer timeline and color legends
- `magnetic_topology_viewer.html` now shows numeric color legends next to the `STATIC Energy`, `STATIC Mass`, `SWE PAD`, and `SWE Energy` heatmaps.
- Added a time slider that stays synchronized with the selected orbit point.
- Added direct time jump support by ISO timestamp input, moving to the nearest available sample.

### 2026-03-27 Viewer three-column layout
- Adjusted `magnetic_topology_viewer.html` to a three-column layout: controls on the left, Mars and orbit view in the center, and data panels on the right.
- Moved the selected-sample detail card into the right-hand data column so the orbit view stays visually centered.

### 2026-03-27 Viewer compact mode and details page
- Moved `Run Summary` and `Selected Sample` out of `magnetic_topology_viewer.html` to keep the main viewer more compact.
- Added `magnetic_topology_details.html` as a dedicated details page.
- The main viewer now includes a link to the details page and passes the loaded JSON plus the current selected sample through browser session storage.

### 2026-03-27 Viewer left-side control layout
- Moved `Time Navigation` and `Data Panels` into the left column of `magnetic_topology_viewer.html`.
- The main page is now a two-column layout: compact control/data column on the left and the Mars orbit view on the right.

### 2026-03-27 Viewer right-side plots only
- Adjusted `magnetic_topology_viewer.html` again to a three-column layout.
- The left column now contains controls and `Time Navigation`, the center column contains the Mars orbit view, and the right column contains only the data visualization panels.

### 2026-03-27 Viewer scale consistency fix
- Fixed the center orbit view so the rendered Mars radius is derived from the same projection scale as the orbit positions.
- This keeps the orbit altitude and Mars size visually proportional during rotation and zoom.

### 2026-03-27 Larger center viewer
- Enlarged the center orbit-view area in `magnetic_topology_viewer.html` by narrowing the left and right columns and increasing the center column minimum width.
- Increased the center viewer card height so the Mars and trajectory view occupies more of the page.

### 2026-03-27 Wider zoom range
- Expanded the zoom range in `magnetic_topology_viewer.html` so the center Mars-orbit view can zoom in closer and zoom out farther.
- Changed mouse-wheel zoom from a fixed increment to multiplicative scaling for smoother control across a larger range.

### 2026-03-27 Single-screen dashboard layout
- Compressed the viewer layout in `magnetic_topology_viewer.html` to reduce the need for scrolling on typical desktop screens.
- Reduced card spacing and plot heights, and arranged the right-side data panels into a compact two-column grid.

### 2026-03-29 Fixed plot ranges in viewer
- Updated `magnetic_topology_viewer.html` so the main data panels use fixed plotting ranges for easier comparison across times.
- Fixed ranges now include:
  - `STATIC Energy`: log-energy axis from `0.1` to `10000.0 eV`, color scale `10^3` to `10^9`
  - `STATIC Mass`: color scale `10^3` to `10^9`
  - `|B|`: `0` to `50 nT`
  - `B_MSO`: `-50` to `50 nT`
  - `Model B_MSO`: `-30` to `30 nT`
  - `SWE Energy`: log-energy axis from `0.1` to `10000.0 eV`, color scale `10^3` to `10^9`

### 2026-03-29 Swappable center window
- Updated `magnetic_topology_viewer.html` so clicking a data panel swaps the center window from the Mars orbit view to a larger live view of that selected plot.
- Heatmap legends remain visible in the larger center view, and the plot stays synchronized when the selected sample changes.
- Added a `Back To Orbit` button to switch the center window back to the Mars trajectory view.

### 2026-03-29 Adaptive redraw in focused view
- Improved the focused center plot view so it is redrawn at the larger canvas size instead of scaling up the smaller side-panel image.
- Axis padding, tick labels, and legend layout now adapt to the larger center window for clearer viewing.

### 2026-03-29 Real time labels on plot x-axes
- Updated the main data plots in `magnetic_topology_viewer.html` so the x-axes show actual UTC times instead of only relative window markers.
- The start, middle, and end times of the current plotting window are now labeled on both line plots and heatmaps.

### 2026-03-29 Code comments for learning
- Added more detailed module-level comments and key function comments to the main Python files.
- Added structural comments to the HTML viewer files to explain layout, state handling, and the main interaction flow.

### 2026-03-30 Directional electron spectra in viewer
- Added per-sample forward/backward electron spectra to `topology_summary.json` through `analyze_magnetic_topology.py`.
- Added a `Directional Electron Spectra` panel to `magnetic_topology_viewer.html`.
- The selected sample now shows a forward/backward electron spectrum sketch similar to the plotting style used in `process_maven_spectra.py`.

### 2026-03-30 Topology-rule bands on spectra
- Added shaded helper bands to the directional electron spectrum plot in `magnetic_topology_viewer.html`.
- The spectrum now highlights the configured topology-rule energy bands:
  - `50-70 eV` knee
  - `180-320 eV` Auger
  - `550-900 eV` dropoff

### 2026-04-05 Compact stacked data panels
- Adjusted the right-side data presentation in `magnetic_topology_viewer.html` to a more paper-like stacked-panel layout.
- Panel labels are now shorter and placed in a dedicated left label strip, while the plots remain aligned in a compact vertical column.

### 2026-04-05 Responsive layout and draggable panes
- Updated `magnetic_topology_viewer.html` so the three-column layout adapts better to browser window size and zoom level.
- Added draggable splitters between the three columns, allowing the left, center, and right panes to be resized interactively.
- The adjusted pane widths are stored in browser local storage and restored on the next page load.

### 2026-04-05 Splitter bug fix
- Fixed a pane-resize bug in `magnetic_topology_viewer.html` where clicking a splitter could incorrectly expand the left column across the page.
- The splitter logic now uses the current rendered pane widths in pixels instead of parsing CSS `clamp(...)` expressions.

### 2026-04-05 Persisted layout recovery
- Hardened the pane-width restore logic in `magnetic_topology_viewer.html` so invalid widths saved in browser local storage no longer collapse the page into a single visible column.
- Pane widths are now persisted as validated pixel values, automatically clamped on load and on browser resize, and cleared when the stored layout is not usable.

### 2026-04-05 Denser right-side data column
- Tightened the right-side data column in `magnetic_topology_viewer.html` so the stacked plots use more of the available vertical space and waste less padding between panels.
- Changed the panel stack to proportional full-height rows and slightly enlarged the plot-axis and legend text for better readability.

### 2026-04-05 Standalone data-panel page
- Added `magnetic_topology_data_panels.html` as a separate page dedicated to the data plots previously shown in the viewer's right column.
- The new page can load `topology_summary.json` directly or reuse cached state from the main viewer, and it keeps the same slider-based and typed-time navigation workflow.
- Added an `Open Data Page` link to `magnetic_topology_viewer.html` so the standalone data page can be opened from the main orbit viewer.

### 2026-04-05 Orbit altitude in analysis results
- Added per-sample orbit altitude fields to `topology_summary.json`: `altitude_km` and `altitude_rm`.
- Updated the viewer pages so the selected-time readout and details page now show the current sample altitude directly.

### 2026-04-05 Altitude plot in data panels
- Added an `Altitude` line plot to the third-column data panel stack in `magnetic_topology_viewer.html`.
- Added the same altitude panel to `magnetic_topology_data_panels.html`, linked to the same time slider and typed-time navigation as the other plots.

### 2026-04-06 Normalized SWE PAD rendering
- Updated the `SWE PAD (111-140 eV)` panel in both `magnetic_topology_viewer.html` and `magnetic_topology_data_panels.html`.
- The panel now shows a normalized pitch-angle distribution instead of absolute electron flux.
- For each time slice, the 111–140 eV PAD is divided by its mean over pitch-angle bins before plotting, and the color scale is now linear with a fixed normalized range of `0.6` to `1.4`.

### 2026-04-06 SWE PAD rainbow colorbar
- Updated the normalized `SWE PAD` panel so its colorbar now uses a rainbow-style mapping closer to the paper-style reference bar: blue/purple at the low end, then cyan/green, then yellow/red at the high end.
- This custom color mapping is only applied to the normalized `SWE PAD` panel and does not change the color mapping used by the other energy and mass plots.

### 2026-04-06 Stable panel resizing during splitter drag
- Improved `magnetic_topology_viewer.html` so the orbit view and third-column data plots redraw continuously while dragging the splitters, instead of only after the drag ends.
- Added resize-observer based redraws so browser resizing and zoom-driven layout changes keep the plot proportions more stable and adaptive.

### 2026-04-06 Adaptive focused-plot layout
- Improved the enlarged center-panel view in `magnetic_topology_viewer.html` so the focused plot area uses a more adaptive grid layout instead of relying on a fixed height subtraction.
- Updated the focused plot renderer to scale margins, axis spacing, and font sizes with the actual canvas dimensions, reducing width/height distortion after splitter drags.

### 2026-04-07 Textured Mars globe in orbit view
- Updated the center orbit view in `magnetic_topology_viewer.html` so the Mars sphere now uses a locally generated surface texture instead of a plain shaded circle.
- The texture remains fully offline and is generated inside the page, making the orbit visualization feel more realistic without depending on external web assets.

### 2026-04-28 Orbit map cache and event figures
- Updated `plot_maven_orbit_map.py` so the crustal magnetic-field background defaults to `185 km` altitude and `2°` latitude/longitude spacing.
- Added reusable `.npz` cache files for the two longitude windows used by the orbit map: `0-180°` and `180-360°`.
- Added `--precompute-crustal-cache` to precompute the two Morschhauser 2014 crustal-field backgrounds before plotting.
- Vectorized the crustal-field grid calculation so first-time cache generation is much faster while matching the original scalar evaluator to floating-point precision.
- Updated `run_maven_event_figures.py` to use the `185 km / 2°` crustal-field map defaults.

### 2026-04-28 Single-column static data panels
- Updated `plot_maven_data_panels.py` to render the static data panels in one vertical column for easier cross-panel comparison.
- Added a black dashed vertical marker at the requested target time on every time-dependent panel.
- Set a shared x-axis window across the time panels so the target-time marker aligns vertically.
- Updated UTC datetime handling in the static panel renderer to avoid Python deprecation warnings.
## Data Shape Flow

This section explains how array shapes change from raw MAVEN files to the final visualization JSON and then to the HTML plots.

Notation:

- `T`: total number of raw time samples
- `P`: number of pitch-angle bins
- `E`: number of energy bins
- `M`: number of mass bins
- `N`: number of topology samples kept for classification
- `Tw`: number of samples inside the currently selected time window

### 1. SWE PAD raw data to normalized arrays

Entry point: `[load_pad_data()](/g:/本研/科学/MARS/ML/maven_code_linux/process_maven_spectra.py)`

The raw SWEA CDF can store the flux array in different axis orders. After normalization, the internal representation is always:

- `times`: `(T,)`
- `energy`: `(E,)`
- `pitch`: `(P,)` or `(T, P, E)`
- `flux`: `(T, P, E)`

So inside this project, SWE PAD is standardized as:

```text
flux: (time, pitch, energy) = (T, P, E)
```

### 2. One-time directional spectrum extraction

Entry point: `[extract_directional_flux()](/g:/本研/科学/MARS/ML/maven_code_linux/analyze_magnetic_topology.py)`

For a single chosen time:

- `pad_data["flux"]`: `(T, P, E)`
- `flux_at_time = flux[time_index]`: `(P, E)`

Then the code averages over pitch-angle bins for roughly parallel and anti-parallel directions:

- `forward_flux`: `(E,)`
- `backward_flux`: `(E,)`

So the shape flow is:

```text
(T, P, E) -> (P, E) -> (E,)
```

### 3. One topology sample

Entry point: `[sample_from_time()](/g:/本研/科学/MARS/ML/maven_code_linux/analyze_magnetic_topology.py)`

After feature detection and MAG matching, one sample contains mostly scalars and short vectors:

- `magnetic_field_nT`: `(3,)`
- `position_km`: `(3,)`
- `position_rm`: `(3,)`

For the whole interval:

- `samples`: list of length `N`

When plotting the orbit, those positions are reassembled into:

```text
positions: (N, 3)
```

### 4. STATIC raw data to heatmap matrices

Entry point: `[load_static_context()](/g:/本研/科学/MARS/ML/maven_code_linux/analyze_magnetic_topology.py)`

The STATIC product is interpreted as:

- `flux`: `(T, M, E)`
- `energy`: `(M, E, sweep_table)`
- `mass_arr`: `(M, E, sweep_table)`

First, the axis coordinates are collapsed to 1D:

- `energy_axis_values`: `(E,)`
- `mass_axis_values`: `(M,)`

Inside the selected time window:

- `selected_flux`: `(Tw, M, E)`

Then two spectrogram-style products are built:

- `energy_spectrogram = median(selected_flux, axis=1)`: `(Tw, E)`
- `mass_spectrogram = median(selected_flux, axis=2)`: `(Tw, M)`

So:

```text
STATIC: (T, M, E) -> (Tw, M, E) -> (Tw, E) and (Tw, M)
```

### 5. SWE context heatmaps

Entry point: `[build_swe_context()](/g:/本研/科学/MARS/ML/maven_code_linux/analyze_magnetic_topology.py)`

Starting from:

- `flux`: `(T, P, E)`

After time-window selection:

- `flux[indices]`: `(Tw, P, E)`

Then two context products are built:

1. Omni electron spectrum
- `omni_spectrum = mean(flux, axis=1)`: `(Tw, E)`

2. PAD in the 111-140 eV band
- `band_flux`: `(Tw, P, Eb)`
- `pad_band`: `(Tw, P)`

Final JSON fields are:

- `times_unix`: `(Tw,)`
- `energy_eV`: `(E,)`
- `pitch_deg`: `(P,)`
- `omni_eflux`: `(Tw, E)`
- `pad_111_140_eflux`: `(Tw, P)`

So:

```text
SWE: (T, P, E) -> (Tw, P, E) -> (Tw, E) and (Tw, P)
```

### 6. MAG context lines

Entry point: `[build_mag_context()](/g:/本研/科学/MARS/ML/maven_code_linux/analyze_magnetic_topology.py)`

After parsing the MAG STS file:

- `data`: `(Tmag, C)`
- `times`: `(Tmag,)`

After time-window filtering:

- `selected`: `(Tw, C)`

Then the plotted magnetic quantities are extracted as 1D series:

- `bx`: `(Tw,)`
- `by`: `(Tw,)`
- `bz`: `(Tw,)`
- `bmag`: `(Tw,)`

### 7. Crustal model context

Entry point: `[build_model_context()](/g:/本研/科学/MARS/ML/maven_code_linux/analyze_magnetic_topology.py)`

The spacecraft positions from MAG PC are:

- `positions`: `(Tw, 3)`

If there are too many points, they are downsampled:

- `sampled_positions`: `(Tw2, 3)`
- `sampled_times`: `(Tw2,)`

Model outputs are then stored as:

- `bx_values`: `(Tw2,)`
- `by_values`: `(Tw2,)`
- `bz_values`: `(Tw2,)`

### 8. Final `topology_summary.json` shapes

Main structures in the exported JSON are:

- `samples`
  - length `N`
  - each sample contains short vectors like `(3,)`
- `context_overview.static`
  - `times_unix`: `(Tw,)`
  - `energy_eV`: `(E,)`
  - `mass_amu`: `(M,)`
  - `energy_eflux`: `(Tw, E)`
  - `mass_eflux`: `(Tw, M)`
- `context_overview.swe`
  - `times_unix`: `(Tw,)`
  - `energy_eV`: `(E,)`
  - `pitch_deg`: `(P,)`
  - `omni_eflux`: `(Tw, E)`
  - `pad_111_140_eflux`: `(Tw, P)`
- `context_overview.mag`
  - `times_unix`: `(Tw,)`
  - `bx_nT/by_nT/bz_nT/bmag_nT`: `(Tw,)`
- `context_overview.model_b_mso`
  - `times_unix`: `(Tw2,)`
  - `bx_nT/by_nT/bz_nT`: `(Tw2,)`

### 9. JSON to HTML visualization

In `[magnetic_topology_viewer.html](/g:/本研/科学/MARS/ML/maven_code_linux/magnetic_topology_viewer.html)`, the page slices the JSON again to only show the currently selected window around the current sample.

Examples:

- `STATIC Energy`
  - JSON: `(Tw, E)`
  - displayed window: `(Tw_window, E)`
- `SWE PAD`
  - JSON: `(Tw, P)`
  - displayed window: `(Tw_window, P)`
- `|B|`
  - JSON: `(Tw,)`
  - displayed window: `(Tw_window,)`

At the HTML drawing layer:

- orbit plot: `(N, 3)`
- heatmap matrix: `(Tw_window, Y)`
- line plot: `times = (Tw_window,)`, `values = (Tw_window,)`

### 10. One-line summary

The main shape flow through the whole project is:

1. Raw data are normalized
   - `SWE -> (T, P, E)`
   - `STATIC -> (T, M, E)`
   - `MAG -> (T, C)`
2. Analysis reduces dimensions based on physical meaning
   - `SWE (T,P,E) -> (P,E) -> (E,)`
   - `STATIC (T,M,E) -> (Tw,E)/(Tw,M)`
   - `SWE (Tw,P,E) -> (Tw,E)/(Tw,P)`
   - `MAG (Tw,C) -> (Tw,)`
3. JSON stores these as lists for the viewer
4. The HTML viewer slices those lists again into the current display window
