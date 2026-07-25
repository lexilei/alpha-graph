# 发车检查包 — Kalshi maker 生产冒烟测试

九席评审团(3 QR / 2 QT / 3 SDE / 1 desk head)全数返回:**一致有条件 GO**。
所有条件已实现或写入本清单。零红灯。

## 定性(desk head C1,全员背书)

这是一次**生产管道冒烟 + 破产筛查**,不是 edge 测试。$50 在统计上不可能
证明 edge(需 ~105 个独立成交;有效样本≈独立 BTC 事件数)。成功的定义:
挂单/撤单/自毁/止损/成交遥测在生产端全部真实工作,无失控。
**P&L 数字不更新任何 alpha 信念。成交太少 = 证据不足,不是无 edge。**

## 发车前已完成(代码侧,全部提交)

- 环境隔离:文件驱动(`data/private/maker_env.txt`),watchdog 重启天然继承;
  基线/止损旗标/遥测流按环境命名空间隔离(demo $962 基线无法再污染生产)
- 生产参数:止损 $15(带宽 30%)、MAX_STRIKES 5(预算可两侧铺满)、
  净库存上限 25 张(最坏结算损失 < 止损)
- 成交遥测:/portfolio/fills 逐笔 + 决策上下文(spot/σ/fv/我方报价/净库存),
  从第一笔成交开始记录;markout 离线可算(+30s/+2min/+0.25τ,按
  series×τ×时段×moneyness 分段)
- 止损求值先于 spot(spot 故障不再弄瞎止损);spot 陈旧 >30s 视同毛刺;
  sink 热路径 ENOSPC 防护
- smoke 脚本只读阶段已实测生产 8/8 通过

## GO 序列(用户回 "go" 后,按序执行)

1. `.venv/bin/python scripts/prod_smoke.py arm` — 生产实弹冒烟(~2 美分):
   1c 远虚值单 → t+45s 确认挂着 → t+140s 确认**自毁**(生产 dead-man 实证,
   全部无人值守安全性的基石)→ 第二张单显式撤单成功。任何 FAIL → 停,不发车。
2. 停 demo maker(watchdog token 冲突,两者互斥);确认 `ps` 无 maker 进程。
3. `maker_env.txt` 写入 `prod`;确认无生产基线/旗标残留(命名空间下天然干净)。
4. 启动 maker(watchdog 拉起或手动);**首个 maker_cycle 必须显示
   baseline ≈ $49.5**。若显示 ~$960 → 立即停,环境隔离失败(панель 4 硬门)。
5. 通报用户:发车确认 + 首周期读数。

## 首小时清单(панель 4)

- T+0-2min:baseline ≈ 49.5;止损 armed;env=prod 在启动行
- T+2-10min:周期延迟 p50 ~10s(>20s 持续 → 查);order_err 应零
  insufficient_balance(少量 post-only cross 正常);equity 与 API 余额对账到分
- T+10-60min:首笔成交 → fill 记录落盘,价格在报价带内;净库存不贴 25 上限
  跨结算;`eq_med − baseline` 走势
- **人工规则:eq_med − baseline ≤ −$10(生产止损的 2/3)→ 手动停,不等自动**
- 检查节奏:发车时、+1h、睡前各一眼;首晚**不持仓过 00:00 UTC 结算**(尽量
  在结算前人工确认净库存≈0)

## kill / hold / scale(панель 2,预承诺)

s = 报价半宽。毒性 = signed FV-markout @ min(2min, 0.25τ)。
- **立即 KILL**:单笔亏损 > $8;post-only 单出现 is_taker=true(故障);
  ≥30 笔跨 ≥3 个 BTC 事件后 markout 差于 −0.5s;每笔净 P&L < −0.5s
- **HOLD 于 $50**(最可能落点):markout 在 −0.5s ~ −0.2s 之间 → 继续攒,不升级
- **SCALE → $500 需全部**:markout 优于 −0.2s;结算参照净 P&L>0 且按独立
  BTC 事件 bootstrap CI 排零;≥100 笔 / ≥5 事件 / ≥3 天;**maker 费率实证**
  (fills 的 fee 字段——我们从未验证过 maker 是否付费!);周内零环境/状态事故
- $500 另需:推送告警 + LaunchDaemon 已装 + 重启演练通过(SRE 关卡)

## 明天不测什么(边界)

- 不测 alpha(见定性);不测 ±150 库存态(资金上限压不到,$500 才会暴露);
- devscan/pin 两条 taker 线已由评审判死(待周末 BRTI 参照重打分终审确认);
- perp 判决时钟 day-0 = 07-26(泄漏修复后首个新信号调仓)。

## 风险边界(风控官签认)

硬上限 = 账户余额 $49.53(交易所预付费机制,无超额路径);止损 $15 是二道防线;
key 不可提现;订单 120s 自毁(生产端将在 GO 序列第 1 步实证)。
