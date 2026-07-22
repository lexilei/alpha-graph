# 套利记录器首轮判读(~6.5h 数据,2026-07-22 05:30 UTC)

三个记录器 2026-07-21 晚启动,均在跑。数据:poly/binance/kalshi 流 ~961MB
(`data/raw/polymarket/`),odds 快照 21 拍(`data/raw/odds/`)。以下按
可执行性排序;所有结论都是 6.5 小时隔夜样本,只够杀死明显幻觉、
不够确认任何边。

## 1. Polymarket 5m taker(狙击过期报价)— 基本死,符合"arb is dead"

`polymarket_latency_analysis.py`:162 个 btc-updown-5m token,29 万次报价
更新 vs 81 万笔 Binance 成交,同一本地时钟。

- Binance 1s 内 ≥4bp 跳变 15 次;跳后 5s 内 poly mid 中位移动 12c,86.7%
  移动 ≥3c — 信息量确实大。
- 但跳后首次报价修订 p50 = 51ms(p90 472ms)。做市商在几十 ms 内撤改。
  我们的可达延迟(家宽 ~200ms / Mac ~500ms)只够碰 p90 尾部,再扣 1%
  taker 费 + 点差,剩余期望接近零。
- lead-lag 相关网格全部 ≈0:5m token 生命周期太短,100ms 网格 ffill 后
  噪声淹没信号 — 方法学问题,不构成独立证据。

判读:慢玩家的 taker 套利在这个市场已被收割干净。不追。

## 2. Polymarket 5m maker(外部 FV 做市)— 最大的机会,但辖区不可执行

`maker_width_analysis.py`(bf68512):用 Binance 作 fair value,按自身
延迟档报"安全半宽" δ(τ) = φ(0)/(σ√τ)·q99(L):

- Mac 500ms 档也只需 9.5c 半宽即可让 q99 的价格移动吃不掉报价;
  vps 50ms 档 3.2c。
- 隔夜流量外推 24h ≈ $10.3M taker 名义/天(白天未测,×2-4);
  10-25% 成交份额 → 毛收入 $4.3-11k/天,pickoff 项在安全宽度下 ≈ 0。
- 模型仍是上界:未含对手做市商压价响应、份额假设未验证、白天 vol
  regime 未测。

判读:结构上是真机会 — 但 Polymarket 对美国 + ON/BC/AB/QC 封锁,
当前辖区不可执行。作为参照系保留:同一 FV-maker 逻辑迁移到 Kalshi
(美国合法,`kalshi_client.py` demo 下单已通)是可执行变体,
safe-width 曲线直接复用。

## 3. Kalshi KXBTC 小时桶 pin 溢价 — 唯一"报价 vs 模型"背离,n=6 不可判

`kalshi_pin_analysis.py`:10s orderbook 流(60s market list 报价在末段
严重过期,已弃用),结算前 10 分钟,现货所在桶:

- 303 拍 / 6 次结算。buy_edge = 布朗隐含命中概率 − ask − taker 费:
  p50 +0.15,p90 +0.41;67.7% 的拍面 edge > 5c。
- 关键形态:bid 和 ask **两侧**都系统性低于隐含概率(sell_edge p50
  −0.22)— 不是宽点差,是市场整体给钉住桶的定价低于终值正态模型。
  这正是跳跃风险溢价的形状:市场付钱买"桶会被击穿"的保护。
- 校准:p_impl∈[0.9,1] 55 拍命中 100%;[0.75,0.9) 88%;6 次结算 5 次
  钉住桶全胜,04:00 那次被甩出(flagged 拍命中仅 42%)— 肥尾的亏损
  就是这么来的,n=6 分不清"免费钱"和"正确定价的跳跃风险"。
- 容量小:best ask 中位挂量 241 张 ≈ $150-250/次。
- 结算代理是 Binance 末笔成交,Kalshi 用自家指数 — 边界局会翻。

判读:三路里唯一值得继续花数据的 taker 方向。需要:(a) 周级样本积累
命中率;(b) 结算指数对齐;(c) 白天时段。隔夜薄流动性可能是 edge 的
全部来源 — 若白天消失,则只是"没人在班上"的残渣。

## 4. MLB 博彩共识 vs Polymarket — 赛前无错价;poller 两个 bug 已修

`odds_edge_analysis.py`。先记录流程教训:odds_poller 上线以来从未抓到
过一场真实比赛 —

- bug 1:gamma `closed=false` 永久返回 5-6 月未结算僵尸市场,
  `ascending=true` 让它们占满窗口(end_date_min 修复,0137bee);
- bug 2:gamma endDate = 比赛日 +7 天结算缓冲,3 天 horizon 过滤把
  所有未来比赛全砍(slug 日期过滤修复);
- 分析侧:MLB 系列赛同队连打,只按队名匹配把已结束/隔日比赛错配 —
  修复前"edge"高达 +49c(多伦多 0.001 ask,实为已结算比赛)。
  按 (队伍对, ET 比赛日) 匹配 + 仅赛前之后:

- 首个有效快照(05:31 UTC,22 场):edge p50 −1.5c,p90 −0.5c,
  最大 +0.7c(Detroit,扣 1% 费后)— Polymarket 赛前 moneyline 贴在
  devig 共识 ±2c 内。跨场所锁定(ask + 1/odds < 1)最好 0.990,
  在费用噪声内。
- HN 式 esports 套利未在 MLB 复现 — 但这只是一个隔夜快照,比赛
  17:00Z 后的临场快照今晚起才开始积累。免费额度 473/500,15min
  一拍可跑 ~5 天。

判读:先验降低,判决延后到有临场数据。

## 下一步(按值博率)

1. 继续攒 Kalshi pin 数据到周级 n;对齐结算指数。
2. Kalshi 版 FV-maker:把 safe-width 曲线套到 KXBTC 小时桶双边报价,
   demo 账户先跑(client 已通)。
3. MLB 临场快照落地后重跑 edge 分析。
4. 白天时段重跑 1/2/3 — 隔夜结论全部不外推。
