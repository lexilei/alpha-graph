# 夜间对抗审查日志

## 2026-07-24 第四轮(用户加发,审"修复的修复",范围 ab32e85^..HEAD)

四镜头全部命中,8 commit 修复,全实测验证:
- **[HIGH,运维+钱数双确认]** maker 暖机回归:基线未建立的 ~90s 里止损失灵却照常报价,且 round(None) 每周期崩掉遥测(当日 15:10 重启 9 连 maker_err 实锤)。修:暖机期不报价(maker_warmup 记录)、滚日分支同加 ≥10 样本护栏、基线文件原子写、解析防呆。重启后实测:3 周期 0 错 0 暖机,基线即时领养。
- **[资源,实测 6.2GB]** iter_jsonl 整 member 物化在最大 poly 小时峰值 6,172MB(机器已在 swap)。重写为流式(增量解压+增量解码+逐行产出):同文件 **290MB**,296 万条记录一条不差;四个损坏样本文件与批处理路径完全等价。顺手把最后三个裸 gzip 消费者(replay/pin/odds_edge)也统一。
- **[MED,回归]** TokenMap:slug 修复让 discovery-first 的 token 永久空 market(26 个已中招);get() 现在回填,下次压缩自愈。retention 拒绝 <7 天窗口(N≤0 本会删光全部已认证 raw)。
- **[LOW]** queue SIGTERM 认领空窗(spec 先于 Popen 注册)+ TERM 不理会时升级 KILL;watchdog pgrep token 加 re.escape。
- **量化战果**:twap 不修的话 ETH 打印会把 BRTI 均值拉偏 −2,917bp(37.7% 秒数偏 >10bp);修后基差 +6.9bp 与判决口径一致。perp 今晚过渡确认无损(100/100 仓位现有报价,首次调仓即全量补 last_mark)。
- **攻击未破**:watchdog 正则对全部 10 种活进程形态 1:1;四象限保证金公式(含买 no@0.98→$4.90);poly flush 实测 628-674 标记/小时;compact 原子写/阻塞锁/边界价格。


每天 04:07 自动运行:有新 commit 则派 4 个 Opus agent(逻辑/数据/运维/钱数四镜头)审当日改动,复核后修 CRITICAL/HIGH。无新 commit 则跳过。

---

## 2026-07-24(首次运行)

**范围** e262360^..d3a0726 之前的 12 个 commit(前夜修复批 + queue_runner + compact_poly,共 12 文件 +630 行)。另捎带:夜间三个分析脚本(latency/width/twap)首夜全崩,查明是最后三个未接 gz_recover 的裸 gzip 读取器,已修(3 commit),latency 复跑通过。

**发现(经本人复核,全部实锤)**:1 latent-CRITICAL、5 HIGH、8 MED、若干 LOW。修复 7 commit,全部推送。

核心条目:
- **[数据/CRITICAL]** compact_poly:白天手动压缩当日 → 半天数据被标 verified → backfill 永久跳过 → 14 天后 retention 删掉从未压缩的下半天 raw,不可逆。修:verified 必须是"完整日(< today)+ 非零 + 三表行数全对账"(books 表原先完全不在验证内)。retention 删除前另加 parquet 存在性复查、--retention N 参数原先被静默丢弃也已修。
- **[数据/HIGH]** TokenMap 并发丢更新:两进程同时压缩会给不同 asset 发同一个 token id,后写者覆盖前者(实测复现)。修:compact_poly 全进程阻塞 flock 串行化。
- **[数据/HIGH]** poly 主流(3.5GB/天)恰恰是唯一没拿到 5s 热循环 flush 的流——静市被 kill 可整丢 30 秒缓冲(实测:未 flush 的 200 行恢复 0 行)。已补齐。
- **[运维/HIGH]** queue_runner 三连:归档 rename 一炸整个 runner 永久死且无人拉起;无单例锁,双开会破坏串行承诺(实测双开互踩);SIGTERM 孤儿化正在跑的作业。修:flock 单例、循环级异常保护、作业独立进程组随 runner 一起终止、纳入 watchdog 管理;另修 gate 队头阻塞(未到期任务不再堵住后面可跑的)。
- **[钱数/HIGH]** perp NAV:前夜的修复是无效修复——账本记的 nav2 仍过滤 bookless 仓位,且每次调仓覆写仓位时抹掉 last_mark(实测当前 state 100 仓位 0 个带 last_mark)。两处已修;修复赶在下次调仓(00:10 UTC)之前,零污染。
- **[钱数/HIGH]** maker 日止损基线不持久:重启即重置,日损上限实际是"重启次数×$25";启动首周期瞬时读数还会把结算尖峰锁成全天基线(双向失效)。修:基线落盘按日持久 + ~100s 中位数暖机 + 保守取下中位。
- **[钱数/HIGH]** 保证金预算用的 balance_dollars 是毛现金(实测挂 24 单余额不变),留存挂单占用未扣。修:预算先减去保留挂单的最大损失占用。
- **[逻辑]** 无 CRITICAL/HIGH;gz_recover salvage 被独立验证(80KB 坏尾 member:新码全恢复 vs 旧码 0 字节)。

**攻击未破(抽样)**:gz_recover 全轴 held(顺序/去重/截断前缀严格超集);Kalshi client_order_id 幂等结构正确(服务端去重待周六实测);watchdog flock/TERM 阶梯/宽限数学 held;微单位精度无损;费率符号无误。

**推迟/记录**:maker+perp 进程重启留待白天(夜间不动交易进程,代码已就位);止损中位数滞后是接受的权衡;HISTORY.log 无轮转;perp last_mark 无时效上限(记账口径待定);PID 复用 TOCTOU(单用户机,忽略)。

**执行动作**:10 commit 推送;polymarket_recorder/devscan/watchdog/queue_runner 已用新代码重启(11/11 单实例);poly_meta 重置 + 全量重压已入队(新验证规则重新认证三天 + 收录 discovery slug)。

**收尾补记(当晚后续)**:
- twap 复跑暴露 ETH 接入的连锁伤:kraken 记录改成 dict 后 twap 仍按裸列表迭代(崩溃),且 coinbase 分支没过滤 prod——ETH $1.8k 成交正混入 BTC BRTI 代理毒化基差。已修(e81c27c),复跑 51 个结算通过;T-15s TWAP 投影误差 p50 0.06bp vs 裸现货 0.44bp,结算锁定的机械信息可用于末分钟定价。
- watchdog 实战暴露 pgrep 是 POSIX ERE、不认 `\S`:带路径前缀的 argv 永远匹配不上 → 每 5 分钟误判 queue_runner 死亡并起新实例(全被 flock 弹走——防御生效,检测在漏)。已修([^ ] 类)并重启验证。
- queue 作业 cwd 陷阱:watchdog 从 scripts/ 启动 runner,相对路径 spec 三连 rc=127;runner 现在固定作业 cwd 并立规范"spec 自带 cd"。
- 重压完成:三天全部按严格规则重新 verified(books 表按 level 行数对账),5.96GB→881MB,discovery slug 已入 token 表。
- 分析三连全部复跑通过(latency/width/twap exit 0)。
