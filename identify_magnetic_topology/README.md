# identify_magnetic_topology 使用说明

这个目录中的程序用于基于 MAVEN SWE/MAG/LPW 数据识别磁场拓扑结构。当前流程主要分为三层：

1. 判断磁场方向：磁场指向火星表面还是远离火星表面。
2. 分别计算 shape parameter 与 PAD score。
3. 调用 `region_id`，按相同的 shape 样本时间判断卫星所在区域。
4. 先应用 `region_id=0/1 -> draped DP`，其他区域再按 Xu et al. 2019 表格融合 shape/PAD/flux ratio。

所有命令默认从仓库根目录运行：

```powershell
cd G:\本研\科学\MARS\ML\maven_code_linux
```

时间格式使用 UTC，例如：

```text
2016-10-06T18:02:00
```

## 1. magnetic_topology_table_method.py

主程序。它会同时调用 `shape_parameter_method.py` 和 `PAD_score_method.py` 的计算逻辑，并额外计算 35-60 eV 的 A/T ratio：

```text
A/T ratio = away_flux / toward_flux
```

随后调用 `region_id` 并根据以下顺序输出磁场拓扑类型：

```text
region_id == 0 or region_id == 1
-> topology = DP
-> topology_label = draped DP
-> topology_subcase = 7b

其他 region_id
-> 保留 Xu et al. 2019 shape/PAD/ratio 表格结果
```

`region_id` 使用每个 shape 样本的原始时间戳，不使用固定 cadence
重新取样。CSV 同时保存覆盖前的 `table_topology`，用于追溯原始表格结果。

融合判别中的 loss cone 不直接使用 `PAD_score_method.py` 输出的分类标签，而是读取 toward/away PAD score 数值：

```text
PAD score < -3 -> LC
PAD score >= -3 -> No LC
```

默认阈值 `-3` 可通过 `--loss-cone-pad-score-threshold` 调整。

Superthermal electron void 使用 40 eV 附近电子微分能量通量的绝对阈值判断。程序选择最接近 40 eV 的 SWE 能道，并对该能道所有有效 pitch-angle flux 求平均：

```text
mean_flux_near_40eV < 1e5 eV cm^-2 s^-1 sr^-1 eV^-1
-> superthermal electron void
```

目标能量和绝对阈值可分别通过 `--electron-void-energy` 和 `--electron-void-flux-threshold` 调整。

默认输出目录：

```text
outputs\identify_magnetic_topology\magnetic_topology_based_on_Xu2019\<START>_<END>\
```

基本运行：

```powershell
python identify_magnetic_topology\magnetic_topology_table_method.py `
  --start 2016-10-06T18:02:00 `
  --end 2016-10-06T18:12:00 `
  --photoelectron-shape-threshold 1.0
```

主要输出：

```text
magnetic_topology_classification.csv
magnetic_topology_timeseries.png
summary.json
shape_parameter_method\shape_parameters.csv
PAD_score_method\pad_score_classification.csv
region_id\region_id_timeseries.csv
region_id\region_id_timeseries.png
region_id\region_id_summary.json
```

关键参数：

```text
--photoelectron-shape-threshold
```

shape parameter 小于等于该阈值时判为 `Phe`，大于该阈值时判为 `SWe`。这个阈值需要根据 template 和样本分布校准。

其他常用参数：

```powershell
python identify_magnetic_topology\magnetic_topology_table_method.py `
  --start 2016-10-06T18:02:00 `
  --end 2016-10-06T18:12:00 `
  --photoelectron-shape-threshold 1.0 `
  --shape-energy-range 20 80 `
  --pad-energy-range 100 300 `
  --loss-cone-pad-score-threshold -3 `
  --electron-void-energy 40 `
  --electron-void-flux-threshold 1e5 `
  --ratio-energy-range 35 60 `
  --region-id-boundary-margin-km 100 `
  --max-region-id-delta-seconds 2 `
  --spectral-smoothing-window 5
```

如果不希望在 shape parameter 分析前做谱的时间平滑：

```powershell
python identify_magnetic_topology\magnetic_topology_table_method.py `
  --start 2016-10-06T18:02:00 `
  --end 2016-10-06T18:12:00 `
  --photoelectron-shape-threshold 1.0 `
  --no-spectral-smoothing
```

拓扑输出字段包括：

```text
time_utc
topology
topology_label
topology_subcase
topology_source
table_topology
table_topology_subcase
region_id
region_name
region_id_delta_seconds
region_id_reason
away_shape_class
toward_shape_class
away_pad_lc
toward_pad_lc
at_ratio_35_60eV
void
electron_void_actual_energy_eV
electron_void_flux
electron_void_flux_threshold
topology_reason
```

其中：

```text
A = away
T = toward
LC = loss cone
Phe = photoelectron-like spectrum
SWe = solar-wind/backscattered-like spectrum
```

如果 `region_id` 为 0 或 1，即使 shape、PAD 或 ratio 信息不足，也会按
case 7b 输出 `draped DP`。其他 `region_id` 仍遵循原表格；信息不足时输出
`unknown`，不会强行分类。

## 2. shape_parameter_method.py

计算 toward/away 两个方向的 shape parameter，用于区分光电子型谱和太阳风/反照电子型谱。

处理逻辑：

1. 读取指定时间段内的 SWE PAD 数据。
2. 将电子谱分成 parallel 与 antiparallel。
3. 读取 LPW spacecraft potential，并对能量轴做修正：

```text
corrected_energy_eV = measured_energy_eV - spacecraft_potential_V
```

如果某个时间点没有可用 LPW 数据，程序仍继续计算，但不对能量轴做 spacecraft-potential 平移。输出中会标记：

```text
spacecraft_potential_V = NaN
lpw_available = False
energy_correction_applied = False
energy_correction_potential_V = 0
status = ..._missing_lpw_no_energy_correction
```

4. 对方向能谱做时间平滑，默认窗口为 5 个 SWE 样本。
5. 计算 df-E 谱。
6. 与 template 谱比较，在指定能量范围内对差值求和，得到 shape parameter。
7. 结合 MAG 判断 parallel/antiparallel 哪个对应 toward/away。

默认输出目录：

```text
outputs\identify_magnetic_topology\shape_parameter_method\<START>_<END>\
```

基本运行：

```powershell
python identify_magnetic_topology\shape_parameter_method.py `
  --start 2016-10-06T18:02:00 `
  --end 2016-10-06T18:12:00
```

主要输出：

```text
shape_parameters.csv
shape_parameters.png
summary.json
```

常用参数：

```text
--template
--shape-energy-range
--spectral-smoothing-window
--no-spectral-smoothing
--cadence-seconds
--max-lpw-delta-seconds
--max-mag-delta-seconds
```

示例：

```powershell
python identify_magnetic_topology\shape_parameter_method.py `
  --start 2016-10-06T18:02:00 `
  --end 2016-10-06T18:12:00 `
  --shape-energy-range 20 80 `
  --spectral-smoothing-window 5
```

## 3. PAD_score_method.py

计算 toward/away 两个方向的 PAD score，并判断是否存在 loss cone。

PAD score 定义：

```text
PAD score = (fFA - fperp) / sqrt(sigma_FA^2 + sigma_perp^2)
```

当前实现：

```text
fFA low side  = 0-30 deg 内有效 bin 的平均 flux
fFA high side = 150-180 deg 内有效 bin 的平均 flux
fperp         = 85-95 deg 内有效 bin 的平均 flux
```

如果 85-95 deg 内没有有效 bin，则取 75-85 deg 内最接近 85 deg 的有效 bin，以及 95-105 deg 内最接近 95 deg 的有效 bin，线性插值到 90 deg。任意一侧没有有效 bin 时，`fperp` 和对应 PAD score 记为缺失。

sigma 默认来自 measured electron fluxes 的 Poisson statistics：

```text
sigma_flux = abs(diff_en_fluxes) / sqrt(counts)
```

如果某个文件没有可用的 3D `counts`，则 fallback 到 CDF 产品中的 uncertainty/variance，并在输出列 `sigma_source` 中标记。

默认输出目录：

```text
outputs\identify_magnetic_topology\PAD_score_method\<START>_<END>\
```

基本运行：

```powershell
python identify_magnetic_topology\PAD_score_method.py `
  --start 2016-10-06T18:02:00 `
  --end 2016-10-06T18:12:00
```

主要输出：

```text
pad_score_classification.csv
pad_score_time_series.png
pad_score_classification.png
summary.json
```

常用参数：

```text
--energy-range
--energy-method
--group-size
--threshold-sigma
--parallel-low
--perpendicular
--antiparallel-high
```

示例：

```powershell
python identify_magnetic_topology\PAD_score_method.py `
  --start 2016-10-06T18:02:00 `
  --end 2016-10-06T18:12:00 `
  --energy-range 100 300 `
  --energy-method mean `
  --group-size 4 `
  --threshold-sigma 2.0
```

输出中重点检查：

```text
toward_pad_score
away_pad_score
toward_class
away_class
pad_shape
perpendicular_method
sigma_source
reason
```

`toward_class` 和 `away_class` 可为：

```text
loss_cone
beam
isotropic
electron_depletion
invalid
```

在融合拓扑判别中：

```text
loss_cone -> LC
isotropic/beam -> No LC
electron_depletion -> Void
```

## 4. magnetic_field_direction.py

判断 MAG 磁场方向相对于火星径向方向是指向表面还是远离表面。

规则：

```text
angle(B, radial position) > 90 deg -> toward_surface
angle(B, radial position) < 90 deg -> away_from_surface
```

基本运行：

```powershell
python identify_magnetic_topology\magnetic_field_direction.py `
  --start 2016-10-06T18:02:00 `
  --end 2016-10-06T18:12:00
```

默认输出：

```text
outputs\identify_magnetic_topology\magnetic_field_direction\<START>_<END>.csv
```

指定输出文件：

```powershell
python identify_magnetic_topology\magnetic_field_direction.py `
  --start 2016-10-06T18:02:00 `
  --end 2016-10-06T18:12:00 `
  --output outputs\identify_magnetic_topology\magnetic_field_direction\test.csv
```

输出字段包括：

```text
time_utc
bx_nT
by_nT
bz_nT
x_mso_km
y_mso_km
z_mso_km
dot_b_r
field_angle_deg
field_direction
```

## 5. print_coadded_pad_results.py

调试程序。用于打印 `PAD_score_method.py` 中 4 个 PAD 样本合并后的中间结果，方便检查每个 pitch angle bin 的 flux、sigma、variance。

默认输出目录：

```text
outputs\identify_magnetic_topology\coadded_pad_results\<START>_<END>\
```

基本运行：

```powershell
python identify_magnetic_topology\print_coadded_pad_results.py `
  --start 2016-10-06T18:02:00 `
  --end 2016-10-06T18:12:00
```

主要输出：

```text
coadded_pad_group_summary.csv
coadded_pad_by_pitch.csv
summary.json
```

常用参数与 PAD score 程序一致：

```text
--energy-range
--energy-method
--group-size
--keep-partial
```

示例：

```powershell
python identify_magnetic_topology\print_coadded_pad_results.py `
  --start 2016-10-06T18:02:00 `
  --end 2016-10-06T18:12:00 `
  --energy-range 100 300 `
  --energy-method mean `
  --group-size 4
```

## 常见检查

查看最终拓扑结果：

```powershell
Import-Csv outputs\identify_magnetic_topology\magnetic_topology_based_on_Xu2019\20161006T180200_20161006T181200\magnetic_topology_classification.csv |
  Select-Object time_utc,topology,away_shape_class,toward_shape_class,away_pad_lc,toward_pad_lc,at_ratio_35_60eV,topology_reason
```

查看 PAD score 是否因为覆盖不足而 invalid：

```powershell
Import-Csv outputs\identify_magnetic_topology\PAD_score_method\20161006T180200_20161006T181200\pad_score_classification.csv |
  Group-Object reason
```

查看 sigma 来源：

```powershell
Import-Csv outputs\identify_magnetic_topology\PAD_score_method\20161006T180200_20161006T181200\pad_score_classification.csv |
  Select-Object -First 5 time,sigma_source
```

## 数据依赖

这些程序通常需要以下 MAVEN 数据已经下载到 `data\maven`：

```text
swe/l2/svypad
mag/ss1s
lpw/mrgscpot
```

其中：

```text
SWE svypad  -> 电子 PAD 与能谱
MAG ss1s    -> 磁场方向与 spacecraft position
LPW mrgscpot -> spacecraft potential
```

如果某类数据缺失，程序会跳过对应时间点或输出 `unknown`/`invalid`，不会用缺失数据强制分类。
