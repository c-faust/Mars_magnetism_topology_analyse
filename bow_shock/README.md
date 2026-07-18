# Mars bow-shock models

本目录提供火星弓激波的统计模型、位置判定接口和给定 UTC 的 MAVEN 位置图。

## 已实现模型

| 名称 | 类型 | 参数或方程 | 用途 |
| --- | --- | --- | --- |
| `vignes2000` | 轴对称圆锥曲面 | `x0=0.64 R_M`, `epsilon=1.03`, `L=2.04 R_M` | 常见 MGS 平均弓激波模型 |
| `trotignon2006` | 轴对称圆锥曲面 | `x0=0.60 R_M`, `epsilon=1.026`, `L=2.081 R_M` | MGS 与 Phobos-2 统计模型 |
| `gruesbeck2018_mso` | 非对称三维二次曲面 | 论文 Table 1 的 all-points MSO 系数 | 默认模型；表示 MAVEN 观测到的南北不对称 |

圆锥模型使用

```text
r = L / (1 + epsilon*cos(theta))
x - x0 = r*cos(theta)
rho = sqrt(y^2 + z^2) = r*sin(theta)
```

三维模型使用

```text
A*x^2 + B*y^2 + C*z^2 + D*x*y + E*y*z + F*x*z
    + G*x + H*y + I*z = 1
```

所有模型坐标和参数均采用 MSO 与火星半径 `R_M`。代码使用
`R_M = 3389.5 km`。

这些模型是统计平均边界。传入 UTC 后，接口会读取该时刻最近的 MAVEN
MAG `ss1s` 位置；模型不会因为 UTC 本身自动变化。若要表示实时太阳风动态压力、
磁声马赫数或 EUV 条件，需要以后接入相应上游数据和参数化模型。

## 数据接口

```python
from bow_shock import get_bow_shock_context, get_bow_shock_surface

context = get_bow_shock_context(
    "2024-11-07T02:15:00Z",
)
print(context.inside_bow_shock)
print(context.boundary_position_mso_km)
print(context.radial_offset_km)

surface = get_bow_shock_surface(
    "2024-11-07T02:15:00Z",
    model_name="gruesbeck2018_mso",
)
```

`get_bow_shock_context` 返回：

- 请求时间、最近 MAG 样本时间和时间差；
- MAVEN 的 MSO 位置、SZA 和高度；
- `inside_bow_shock` 与 `location`；
- 沿 MAVEN 火心径向射线与模型的交点；
- MAVEN 相对该交点的径向偏移，负值表示位于模型内侧；
- 模型名称、类型、坐标系和 MAG 来源文件。

已有 MSO 位置时可跳过 MAG 文件读取：

```python
context = get_bow_shock_context(
    "2024-11-07T02:15:00Z",
    spacecraft_position_mso_km=[5000.0, 1000.0, -500.0],
)
```

## 绘图

```powershell
python bow_shock\plot_bow_shock.py `
  --time 2024-11-07T02:15:00Z
```

默认输出：

```text
outputs/bow_shock/<UTC>_<MODEL>.png
outputs/bow_shock/<UTC>_<MODEL>.json
```

PNG 包含 `X-Y`、`X-Z` 截面和三维曲面；红色星号为 MAVEN，黄色圆点为沿
MAVEN 火心径向射线得到的模型边界位置。JSON 保存同一时刻的完整位置接口结果。

## 文献

- Vignes et al. (2000), *The Solar Wind interaction with Mars: Locations and
  shapes of the Bow Shock and the Magnetic Pile-up Boundary*,
  https://doi.org/10.1029/1999GL010703
- Trotignon et al. (2006), *Martian shock and magnetic pile-up boundary
  positions and shapes determined from the Phobos 2 and Mars Global Surveyor
  data sets*, https://doi.org/10.1016/j.pss.2006.01.003
- Gruesbeck et al. (2018), *The Three-Dimensional Bow Shock of Mars as
  Observed by MAVEN*, https://doi.org/10.1029/2018JA025366
