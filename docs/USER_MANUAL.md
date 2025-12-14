# 使用手册

## 运行方式

1. 安装依赖

```bash
pip install -r requirements.txt
```

如遇缺包，也可以启动后点击底部“安装依赖(清华源)”一键安装。

2. 配置 Tushare Token

方式一：设置环境变量

PowerShell 当前窗口：

```bash
$env:TUSHARE_TOKEN="你的token"
```

永久设置：

```bash
setx TUSHARE_TOKEN "你的token"
```

方式二：创建本地文件

在项目根目录创建 `secrets/tushare_token.txt`，文件内容仅一行 token。

2. 启动桌面程序

```bash
python main.py
```

如需从旧入口启动也可使用：

```bash
python app.py
```

## 输出目录

项目所有输出写入项目根目录下 `outputs/`：

- `outputs/figures` 图像
- `outputs/tables` 表格与报告
- `outputs/forum_db` 抓取原始与清洗数据
- `outputs/models` 深度学习模型与配置
- `outputs/logs` 日志文件
- `outputs/cache` 缓存文件

## 全局界面说明

- 底部进度条：后台任务运行时会自动滚动，UI 不会卡死
- 底部日志面板：显示运行日志与错误信息
- 使用手册按钮：打开本文件

## Tab1 股票数据与回归

### 数据源

- `Tushare (CN)` 默认，适用于 A 股与指数日线/周线
- `yfinance` 备选，适用于美股/ETF
- `CSV 上传` 离线模式，必须同时提供标的与基准 CSV

### Tushare ts_code 示例

- 股票：`600519.SH`、`000001.SZ`
- 指数：`000300.SH`（沪深300）、`399001.SZ`（深证成指）

### 参数说明

- 开始/结束日期：`YYYY-MM-DD`
- 频率：日/周
- 价格字段：Close 或 Adj Close（若缺失会自动兼容）
- 对数收益率：log return 或 simple return
- 是否年化：收益率是否按频率年化
- 无风险利率：年化小数，默认 0；可选超额收益 CAPM

### 导出

- 一键导出：写入 `outputs/tables`
- 保存图像：写入 `outputs/figures`

## Tab2 爬虫 + 文本分析

### 数据源

- `自动抓取` 优先尝试 JSON 接口，失败再 HTML 解析，内置超时、重试、频率限制与缓存
- `CSV 上传` 解析用户已抓取的 CSV
- `HTML 上传` 解析用户已抓取的 HTML 中的链接列表

### 参数说明

- 关键词：股票代码或名称关键词
- 抓取页数/条数上限：控制抓取规模
- 开始/结束时间：可留空
- 请求间隔：默认 1 秒，建议不要小于 1
- Cookie/Headers：支持 JSON 或 `k:v`/`k=v` 多行格式
  - 高级配置（写在同一个文本框里）：`workers=12`、`fetch_detail=1`、`detail_min_len=40`、`max_detail=2000`

### 导出

- 原始与清洗数据：`outputs/forum_db`
- 分析结果 Excel：`outputs/tables`

### 自检

```bash
python self_check_text.py
```

## Tab3 深度学习示例

### 数据源

- `Tushare (CN)` 默认：根据 tickers 拉取 OHLCV 并构造样本
- `yfinance` 备选
- `CSV 上传` 用户提供 OHLCV 数据训练

### 参数说明

- tickers：逗号分隔
- 任务：回归预测下一期收益率 / 分类预测涨跌
- 特征：OHLCV + 技术指标（MA/RSI/MACD/Volatility）
- 切分：按时间顺序 train/val/test，不随机打乱
- 快速演示模式：自动限制样本与训练轮数

### 保存

点击“保存模型与报告”会生成：

- `outputs/models/dl_model_*.pt`
- `outputs/models/dl_config_*.json`
- `outputs/tables/dl_report_*.csv`、`outputs/tables/dl_report_*.json`
- `outputs/tables/dl_pred_*.json`
- `outputs/figures/dl_*.png`

## Tushare Token 安全

- Token 仅用于 API 初始化，项目不会将 Token 打印到日志、控制台或导出文件中
- 如需更换 Token，可修改 `src/config.py` 中的 `TUSHARE_TOKEN`
