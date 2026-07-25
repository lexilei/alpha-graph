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

## GO 序列(夜审 r2 运维镜头验证版:**先改文件,后杀进程**——
## 否则 watchdog 的 300s 扫描会在间隙用旧文件复活 demo,切换静默失败)

0. 状态核对:env=demo、恰好 1 个 maker 进程、watchdog 活着。
1. 生产实弹冒烟——**已于 07-25 06:4x 提前通过**(dead-man t+140s 自毁 +
   撤单实证,failures=0,~2 美分)。smoke 仅限发车前运行(prod maker 活着
   时会误判+下真单,已加防呆)。
2. **先改文件**(原子写):`printf prod > maker_env.txt.tmp && mv -f …`;
   cat 确认恰为 `prod`。此刻起任何 maker 启动都是 prod;在跑的 demo 不受影响。
3. 归档死文件:`mv data/logs/launchd/maker_baseline.txt{,.old}`(旧未隔离基线)。
4. **后杀进程**:`kill -TERM <maker_pid>`;demo 挂单 120s 内自毁。
5. **让 watchdog 单点复活**(≤300s,不手动补启;maker 现已加 flock 单例,
   但首选 watchdog 路径)。
6. 验证三连:`pgrep` 恰好 1 个 pid;启动行 `env=prod stop=$15`;
   `makerprod_*.jsonl.gz` 新文件出现。**首个 maker_cycle baseline ≈ $49.5**
   (~$960 = 隔离失败,立即停)。暖机 ~100s 不下单,是最后确认窗口。
7. 通报用户:发车确认 + 首周期读数,进入首小时清单。

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
