# MAVEN 轨道空间覆盖率

本目录中的 `calculate_maven_orbital_coverage.py` 用 MAVEN MAG
`sun-state/MSO 1-second`（`ss1s`）日文件计算指定时间段内的轨道空间覆盖率。
程序先检查 `data/maven` 下的本地数据；某一天缺失时，默认调用项目根目录已有的
`download_maven_data.py` 从 LASP 下载补全。

## 覆盖率定义

时间区间采用半开区间 `[start, end)`。一个空间网格只要被至少一个有效的
1 秒轨道位置样本命中，就视为已覆盖：

```text
coverage_rate = covered_cell_count / total_cell_count
```

逐网格 CSV 还会给出 `sample_count`、该网格样本数占所有范围内样本的百分比，
以及 `covered`（0 或 1）。

直角坐标覆盖率的分母是用户设置的长方体中的所有网格。如果长方体包含火星内部，
这些内部网格也会进入分母。球坐标中的海拔高度定义为距火星平均表面的高度，
单位为 `R_MARS`，其中代码采用 `R_MARS = 3389.5 km`。

## 坐标定义

- 直角坐标：MAG `ss1s` 文件直接给出的 MSO `X/Y/Z`，统一除以火星半径。
- 球坐标：由同一 MSO 位置转换得到。
  - `altitude_rm = sqrt(X^2 + Y^2 + Z^2) - 1`
  - 经度从 `+X` 朝 `+Y` 增大。
  - 纬度朝 `+Z` 为正。

这里的经纬度是 **MSO 球坐标经纬度**，不是行星固连的地理经纬度。

## 基本运行

在项目根目录执行：

```powershell
python MAVEN_orbital_coverage_rate\calculate_maven_orbital_coverage.py `
  --start 2016-10-06T18:00:00Z `
  --end 2016-10-07T00:00:00Z
```

若数据齐全，程序只读本地文件；缺失时自动下载。若只允许使用本地数据，可加：

```powershell
--no-auto-download
```

## 调整直角坐标网格

下面的例子调查：

- `X ∈ [-3, 3] R_MARS`，80 格
- `Y ∈ [-2, 2] R_MARS`，60 格
- `Z ∈ [-2, 2] R_MARS`，60 格

```powershell
python MAVEN_orbital_coverage_rate\calculate_maven_orbital_coverage.py `
  --start 2016-10-06T18:00:00Z `
  --end 2016-10-07T00:00:00Z `
  --cartesian-range -3 3 -2 2 -2 2 `
  --cartesian-bins 80 60 60
```

`--cartesian-bins` 表示各轴的**网格单元数**。

## 调整球坐标网格

```powershell
python MAVEN_orbital_coverage_rate\calculate_maven_orbital_coverage.py `
  --start 2016-10-06T18:00:00Z `
  --end 2016-10-07T00:00:00Z `
  --altitude-range 0 2 `
  --altitude-bins 40 `
  --latitude-range -90 90 `
  --longitude-range -180 180 `
  --delta-degree 2
```

`--delta-degree` 同时控制经度和纬度的网格角度。若角度范围不能被它整除，
最末一格会略窄。也可以分别直接指定格数；指定后，该轴不再使用
`--delta-degree`：

```powershell
--latitude-bins 90 --longitude-bins 180
```

经度也支持 `--longitude-range 0 360`，或者跨度不超过 360° 的局部范围。

## 输出

默认输出目录为：

```text
outputs/MAVEN_orbital_coverage_rate/<start>_<end>/
```

可用 `--output-dir` 修改。目录中包含：

- `cartesian_mso_coverage.csv`：每个直角坐标网格一行。
- `spherical_mso_coverage.csv`：每个高度—纬度—经度网格一行。
- `coverage_summary.csv`：两种坐标系的总网格数、覆盖网格数、覆盖率、有效样本数、
  范围外样本数和实际网格参数。

CSV 使用 UTF-8 with BOM 编码，可直接由中文版 Excel 打开。

## 全部参数

```powershell
python MAVEN_orbital_coverage_rate\calculate_maven_orbital_coverage.py --help
```

## 测试

测试使用合成坐标，不会访问网络或下载数据：

```powershell
python -m unittest MAVEN_orbital_coverage_rate.test_orbital_coverage
```
