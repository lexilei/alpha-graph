# Ideas Ledger — 想法账本(测试纪律的镜像)

FACTORS.md 管"测一个想法",这里管"选择测什么"。想法先登记、先排序,再进代码。

## 规则

1. **任何想法先进这里,再动数据/代码。** 登记不算 look;N 只在该想法领到
   C-ID 并写入 `reports/factor_preregistration.md` 之后开始计。
2. **从队列头部取**(分数最高的 backlog 条目),不从最近的灵感取。跳过队头
   要写一行理由(deferral 也是决定,镜像 C19 的处理)。
3. 进队列必须填全六个字段:**机制**(为什么 mispriced、对手盘是谁)、
   **数据源**(是否在手)、**文献先验**、**拥挤度**、**我的边**、**测试成本**。
   填不出机制的想法留在 Inbox,不参与排序。
4. 文献先验里的数字登记时可标 **(待核)**;promotion 前必须核对原文或
   OSAP(openassetpricing.com),核不上的数字降 P 重排。
5. 分数是先验:family 证据出来后更新分数并注明日期,不删旧分。
6. **Family 预算与退出规则(= 账本 binding rule 6,2026-07-15 起)**:
   family 在第一个成员领 C-ID 注册时,同一条 ledger 行里钉四样 —
   look 预算(全体成员 primary + pinned diagnostics 合计)、日历截止
   (默认开族起 6 周)、stop rule(默认:预算用尽或到期时最好 primary
   sn incr \|t\| < 1.5 → family 永久关闭)、reopening bar(默认 2.0,
   仅对未试过的新构造开放 — C24 先例)。过 1.5 的成员走 member 级
   candidate 流程,confirmation 不占 family 预算。预算修订仅限该
   family 下一次计算之前,且进账本留痕。U1 换 universe 后,因栖息地
   关闭的 family 重新钉预算即可重开(2.0 bar 不跨 universe)。
   账本里的版本是 binding,这里是镜像。

## 排序

**分 = P(真) × 容量 ÷ 成本 × 正交**

- **P(真)** 1–5 — 2026-07-15 起按衰减文献核定的规则打(出处与原始文件:
  `data/reference/{hxz,osap,decay_literature}/`;OSAP 发表后 t 一律从
  `PredictorLSretWide.csv` 本地重算复核;MP 精确系数来自 82 特征草稿版
  PDF,发表版定性一致、精确数字待机构访问):
  - 地板:同行评审异象 P≥3 — 纯数据挖掘偏差只占 in-sample 收益 ~12.3%
    (SE 1.7pp,Chen-Zimmermann RAPS 2020);往下压靠罚则,不靠发表怀疑论。
  - 发表后 OOS 证据仍在 → P=4;仍在 + 机制扎实 + 栖息地匹配 → 才给 5
    (平均发表后衰减 ~50-58%,McLean-Pontiff JF 2016)。
  - 只有 in-sample → 封顶 3(MP:纯统计偏差 ≈ 26%)。
  - 发表后证据已死(t≈0 或反号)→ P≤2;HXZ NYSE-VW(最接近本面板的
    已发表设计)直接 FAIL → P=1(U1 条件)。
  - **栖息地罚 −1**:效应集中于小盘/低流动/高特质波动(MP Table 8:
    idio +4.05 p<.001、size −1.49 p=.013、$vol −1.67 p<.01 — 本面板恰在
    拟合残留 ≈0 的角落;Jacobs-Müller 2020:US 是唯一有可靠发表后衰减
    的市场)。U1 恢复这 1 分。
  - **本面板校准**:HXZ-VW passer 已在本面板测过 5 例(C25/C26/C28/
    C29/C17),仅 C17 边缘存活 → "HXZ-VW pass 且未测"默认 P=2。
  - 1 = 纯 anecdote、无同行评审证据。
- **容量** 1–5:5 = 宽截面、日频可交易;3 = 窄截面或月频;1 = 每年个位数
  事件(参考 C16 的 MIN_XS 教训:事件因子标准 judge 打不了分)。
- **成本** 1–5:1 = 数据在手、一周内出 primary look;3 = 需要新抓取/解析
  管道;5 = 需要长期前瞻采集或付费数据。
- **正交** 0.8–1.3:对现有 candidate(C11、C17)和 B1–B9 controls 的先验
  正交性;明显同源打 0.8,全新信息源打 1.2–1.3。

## 队列(按分降序;status: backlog → promoted(C-ID) → closed)

| ID | name | family | status | P | 容量 | 成本 | 正交 | 分 | 登记内容 |
|----|------|--------|--------|---|------|------|------|----|----------|
| I2 | `cp_iv_spread` | options-xs | backlog | 4 | 3 | 2 | 1.2 | 7.2 | **机制**:put-call parity 偏离 = 期权市场的方向性私有信息,股价滞后跟随;对手盘 = 只看股票的投资者。**数据(QA 更正 2026-07-15)**:单名 EOD NBBO 实际 **5/61 在盘** — 下载 2026-06-10 起中断,当日已恢复续传(`supervised_singles.sh`);纯报价、无现成 IV/OI(有 volume),IV 需自行反解,vol_smile 管道可移植。**文献**:Cremers-Weinbaum 2010 JFQA — call−put IV spread(同 strike/expiry)预测未来一周~一月收益(量级待核原文);An-Ang-Bali-Cakici 2014 JF — ΔIV 方向性预测。**OSAP 已核(2026-07-15,`PredictorLSretWide.csv` 重算复核)**:CW 本身不在 OSAP;最近类比 CPVolSpread(Bali-Hovakimian 2009,call−put IV)**发表后 t=3.25、2011+ 3.13 — 存活**,P=4 维持,**现为队头**。**拥挤**:知名但有数据门槛。**成本**:管道移植 ~2。61 名截面窄 — 先作 family 先验验证,过了再决定扩数据。**Deferral 2026-07-17(rule 2)**:队头让位 — singles 下载 7/61,QA gate 无法运行;IV 管道已建成并过审计(Black-76 修正,`reports/iv_pipeline_audit_2026-07-17.md`),下载齐即回队头,不占新 family 预算。 |
| I19 | `industry_momentum` | industry-momentum | backlog | 2 | 4 | 1 | 0.8 | 6.4 | **来源:HXZ 复现扫描(2026-07-15)。机制**:行业级信息反应不足 → 行业收益延续(Moskowitz-Grinblatt 1999);对手盘 = 只看个股的选股者。**HXZ NYSE-VW:0.68%/月 \|t\|=2.86,6m/12m 3.01/3.57 — 全 horizon PASS,大盘下最稳的动量变体之一**。**数据**:价格面板 + sector map 在手,成本 1。**设计冲突(开测前必须解决)**:这是纯跨行业信号,而 v0 primary judge 是 sector-neutral — **sn 会按构造消灭它**(旁证:C9 的 spillover 动量被 sn 砍半,说明行业动量在面板里真实存在)。开测前须预注册非 sn 的评价设计(raw incremental + 显式接受行业敞口),或者承认平台不做行业赌注、按 rule 2 写理由跳过。正交 0.8(与 B1/B5 相关 + 设计冲突)。P=2 按本面板校准。**Deferral 2026-07-17(rule 2)**:非 sn 评价 = 平台是否做行业赌注,是用户层决定,未裁定前不开测;数据在手且无设计冲突的 I21 先行。 |
| I14 | `gkm_high_volume` | liquidity-volume | closed(C31 rejected) | 2 | 4 | 1 | 0.8 | 6.4 | **机制**:可见性假说 — 极端放量事件把股票拉进投资者视野 → 买压(个人投资者只买引起注意的票)→ 数周正漂移(Gervais-Kaniel-Mingelgrin 2001 JF,待核 OSAP 存活)。**构造(差异化关键)**:事件式 — 当日/当周成交量落在其自身 trailing 窗口的极端分位(GKM 原文 top decile),**不是**水平 z-score;B4 volume_zscore 就是异常量水平,不差异化就是重测自家 control,正交 0.8 是这么来的。**数据**:在手,成本 1。**反证**:发表后衰减;可见性机制对本就人人盯着的 S&P 500 大盘先验弱(U1 后上调)。占 I9 行钉的 3-primary 家族预算之一。**双源无判决(2026-07-15)**:GKM 不在 HXZ 452,也不在 OSAP 212 — 无任何独立复现追踪;按规则 in-sample-only 封顶 3、栖息地 −1 → P=2 维持。**Promoted 2026-07-18 → C31**(member 2;事件式构造钉死:±1 极端分位 vs 自身前 49 日、grid t+1 = 次日建仓,sign 钉正)。**结果(当日,单 pinned look)**:sn incr t=**−0.10**(raw −0.78 反号)→ **rejected** — 现代大盘无可见性溢价,与压低的 P=2 先验一致;流程胜利:低先验只花了 1 look 就出清。 |
| I21 | `volume_trend` | liquidity-volume | promoted(C30) | 2 | 4 | 1 | 0.8 | 6.4 | **来源:OSAP 扫描(2026-07-15)。机制**:成交量趋势(短窗均量 vs 长窗均量)先于价格调整(Haugen-Baker 1996;机制论述薄,更接近实证规律 — 这本身是个先验扣分项)。**OSAP 证据(重算复核)**:发表后 t=5.08、2011+ 仍 3.84(mean 0.49%/mo)— **trading 类唯一的现代存活者**,与 Amihud/turnover 的发表后全灭形成对照。**大警告**:EW 全 universe 证据,微盘可能承担全部;HXZ 452 无此变量,大盘判决缺失 → 按校准 P=2 不给 3。**数据**:在手,成本 1。**正交 0.8**:与 B4 volume_zscore 同源风险高(B4 是量的异常水平),构造必须用趋势/斜率差异化。**Promoted 2026-07-17 → C30**(开族注册,预算钉在账本:≤3 primaries、deadline 2026-08-28、stop 1.5、reopen 2.0;OSAP 60 月斜率构造,sign 钉负)。**结果(当日,单 pinned look)**:sn incr t=**−3.12**(IC −0.0161,150m,ortho 0.931,与 B4 corr 仅 −0.06)→ candidate,账本史上最强 primary,首越 E[max]-null 天花板。**确认协议(2026-07-20 注册即跑):FRAGILE,live path 关闭** — 分半上半场 **+0.04**(零)、下半场 −4.60,全部效应来自 2020-2026 单一 regime,钉死的 bar(两半同负且 ≥1.0)机械落判;救活需以 post-2020 regime 假设显式重注册 + forward-only bar。 |
| I1 | `iv_skew_xs` | options-xs | backlog | 3 | 3 | 2 | 1.2 | 5.4 | **P 4→3(2026-07-15,OSAP 重算复核)**:skew1(XZZ 2010)in-sample t=2.19,**发表后 t=1.20、2011+ 同 1.20** — 衰减但同号,按规则"发表后衰减仍正"= P=3,让出队头给 I2。**机制**:知情交易者先在期权市场表达负面信息(OTM put 需求 → smirk 变陡),股票市场消化滞后数周;对手盘 = 只看股票的投资者。**数据**:见 I2 的 QA 更正(5/61,续传中);IV 自行反解,成本 2。**拥挤**:期权数据门槛保护,中等。与 I2 同族,预算已钉(≤9 looks)。 |
| I3 | `iv_rv_spread_xs` | options-xs | backlog | 3 | 3 | 2 | 1.2 | 5.4 | **机制**:单名 VRP 截面差异 = 对冲需求/彩票偏好定价差;比 I1/I2 更偏风险溢价而非信息。**数据**:在手(RV 管道 vol_smile 已有,`signals/rv`)。**文献**:Bali-Hovakimian 2009 — implied−realized spread 预测截面收益;**旁证(2026-07-15)**:同论文的 call−put 构造(OSAP CPVolSpread)发表后 3.25 存活,IV−RV 构造本身 OSAP 无独立追踪 → in-sample-only 封顶 P=3 维持。**拥挤**:中等。**注意**:与 vol_smile 的时序 VRP 负结果不冲突(那是市场级、卖方策略;这是截面相对定价)。 |
| I6 | `short_interest_xs` | ownership-flow | promoted(C34) | 3 | 3 | 2 | 1.2 | 5.4 | **机制**:卖空约束下负面信息进价慢;高 SI = 知情空头聚集 → 低收益;对手盘 = 借券成本盲的多头。**文献**:Boehmer-Jones-Zhang 等,发表后在难借券段存活(待核 OSAP)。**数据**:FINRA 双周 SI 免费;Sharadar 含 SI 字段(待核)。**拥挤**:高,但残余集中在小盘难借段。**注意**:S&P 500 大盘易借,先验偏低 — 此条在 universe 扩展(见基建 U1)之后先验显著上调,当前分按现 universe 打。**OSAP 已核(2026-07-15,重算复核)**:ShortInterest(Dechow 等 2001)**发表后 t=3.98、2011+ 3.10 — 发表后增强**(全 universe EW)→ 基线 P=4 − 栖息地罚 1 = **P=3 维持,但从"偏低的 3"变成"有引用的 3"**;警告:IO_ShortInterest 变体 2011+ mean 7.18%/mo 是微盘极端组合 artifact,勿采用该构造。**Promoted 2026-07-20 → C34,当日评估:sn incr t=−1.96(58 月短面板,IC −0.0183 ≈ 注册 MDE)→ candidate**;单时代面板,OOS 只能靠 forward 累积。 |
| I22 | `div_seasonality` | dividend-seasonality | promoted(C36) | 3 | 3 | 2 | 1.1 | 5.0 | **来源:OSAP 扫描(2026-07-15)。机制**:分红月价格压力 — 预计本月除息/派息的股票获得可预期需求(Hartzmark-Salomon 2013 DivSeason;Litzenberger-Ramaswamy 1979 DivYieldST)。**OSAP 证据(重算复核)**:DivYieldST 发表后 t=9.61、2011+ 2.24(mean 0.25%/mo);DivSeason 发表后 2.11、2011+ 2.87(mean 仅 0.10%/mo)。**栖息地罕见地对口**:分红股 = 大盘,本面板不吃 −1 罚 → P=3。**主威胁是 gate-2**:月度轮动 + 小 mean,C15 的"gross ≈ cost"高概率重演 — 注册时必须先算 break-even 换手,过不了就别开族。**数据**:分红历史(调整/未调整价差 + XBRL 股息),成本 2;序列 2010 起 → 2011-14 面板对长回看略短。两构造一族,开族预算合并钉。 |
| I5 | `breadth_13f` | ownership-flow | closed(C35 rejected) | 3 | 4 | 3 | 1.1 | 4.4 | **机制**:Miller 分歧+卖空约束 — 机构持有广度下降 = 悲观者被挤出价格 → 高估 → 低收益。**数据**:EDGAR 13F 免费,45 天滞后是天然 PIT 可用性(比文本家族干净)。**文献**:Chen-Hong-Stein 2002 JFE。**OSAP 已核(2026-07-15,重算复核)**:DelBreadth **发表后 t=1.41、2011+ 1.73 — 弱但同号**,按"衰减仍正"P=3 维持。**拥挤**:13F 被大量使用,该构造中等。**边**:EDGAR 管道现成。**成本**:3 — 13F 解析出名地脏(CUSIP↔ticker 映射、份额单位错报),QA 量大。与已关的文本家族不同源:这是 ownership 数据不是文本。 |
| I25 | `earnings_crush_xs` | options-pnl | backlog | 3 | 3 | 3 | 1.3 | 3.9 | **来源:vol_smile 单票化并入(2026-07-20)。期权空间 P&L vertical(判决 = 期权持仓收益,非股票截面;评价设计开族时预注册)。机制**:财报前 IV run-up 系统性超过实现跳动 — 卖跨式/宽跨吃 crush;对手盘 = 财报前买保护/买彩票的方向性买家。**横截面化(与单纯"每场都卖"的区别)**:按 run-up 相对自身历史与同业的 richness 排序,只卖贵的一端 — 61 名 × 每名 4 次/年 ≈ 244 事件/年。**文献**:财报期权收益的学术记录(卖方平均赚、带尾部;具体 t 待核)→ P=3 地板。**数据**:singles NBBO + 财报日历(2.02 管道在手);成本 3(引擎移植 + 事件对齐)。**尾部风险是本条的全部**:单事件 -10σ 可能,评价设计必须含尾部指标(非只均值),开族时钉。 |
| I18 | `rn_skew_bkm` | options-xs | bench | 3 | 3 | 3 | 1.2 | 3.6 | **预算外替补**(同 I17,测前须 rule-6 修订)。**机制**:风险中性偏度 = 崩盘保险需求 / 彩票需求的截面定价差异(Conrad-Dittmar-Ghysels 2013、Bali-Murray;**方向在文献内有分歧,登记时必须预注册符号**,待核)。**数据**:BKM 无模型矩要求整条 smile — vol_smile 的 SVI 拟合管道可复用,成本 3。**正交 1.2**:三阶矩信息,与 I1 的 skew 斜率相关但不同构造(斜率 vs 积分矩),若 I1 先行且活,I18 增量另算。**旁证(2026-07-15)**:BKM 构造不在 OSAP;实现矩类比 ReturnSkew 发表后 t=1.28 弱 — in-sample-only 封顶 P=3 维持。 |
| I17 | `os_volume_ratio` | options-xs | bench | 2 | 3 | 2 | 1.2 | 3.6 | **P 3→2(2026-07-15,OSAP 重算复核)**:OptionVolume1(Johnson-So 2012)**发表后 t=0.40、2011+ 0.90 — 发表后死**;OptionVolume2 反号(−1.40)。按规则"发表后已死"P≤2。**预算外替补**:options-xs 已按 I1–I3 钉 ≤9 looks,I17 测前须 rule-6 修订 — 现在更没有理由动它。**机制**:知情者偏好期权(杠杆 + 做空约束)→ O/S 量比负向预测(原 in-sample t=3.45)。**数据字段已核(2026-07-15)**:raw_singles 有 volume、无 OI — Johnson-So 的 O/S 只需期权成交量,构造可行;等 61 名下载齐。 |
| I20 | `mom_seasonality_hs` | seasonality | backlog | 1 | 4 | 1 | 0.9 | 3.6 | **P 2→1(2026-07-15,登记当日即被 OSAP 否决 — 记录在案作为流程的胜利)**:HXZ in-sample NYSE-VW Ra1 3.43 强 pass(RFS 2020 Table 3),但 OSAP MomSeason(Heston-Sadka years-2-5,同族最近追踪)**发表后 t=−0.20、2011+ −0.08 — 发表后死**;按规则 post-pub OOS 压倒 in-sample。构造注:OSAP 追踪的是 years-2-5 平均,与 HXZ 的 year-1 年度滞后不完全同构 — 若日后想救,须先核 OSAP 的 MomSeasonShort(year-1 对应物)再谈。**机制**:年度日历周期(机构再平衡/财报日历)。数据在手成本 1,便宜但排后。 |
| I23 | `earnings_streak` | earnings-events | closed(C32 rejected) | 2 | 4 | 2 | 0.9 | 3.6 | **来源:OSAP 扫描(2026-07-15)。机制**:连续同号盈利意外的"连胜"被低估 — 趋势外推不足(Loh-Warachka 2012)。**OSAP 证据(重算复核)**:发表后 t=4.28、2011+ 4.71(mean 0.63%/mo,EW 全 universe)— 现代存活最强的 earnings 类。姊妹构造 RevenueSurprise(Jegadeesh-Livnat 2006:2.76/3.03)记 Inbox 备选。**数据**:C17 的 XBRL EPS + 8-K 2.02 管道直接复用,成本 2。**正交 0.9**:PEAD 线,必须对 C17 SUE 增量成立。**family 归属已裁定(2026-07-20,账本)**:earnings-events — 已关的基本面族关的是水平/比率特征,这是事件流上的序列构造;开族 pin:≤2 primaries(I23 + RevenueSurprise 条件性 member 2)、deadline 2026-08-31、stop 1.5、reopen 2.0;C17 老 path 不占预算。**Promoted → C32**(SRW-SUE 号替代 IBES surprise,构造适配已钉;sign 钉正)。**结果(当日,两轮)**:首评 +1.57 擦线 candidate → 审计抓到过期行不被后续公告取代(39.8% 值被提前吞,恰好美化每日 IC),修正重跑 **+1.40 → rejected**;家族预算尽、best < 1.5 → **earnings-events 永久关闭**。**构造恒等教训**:C32 = C17 值的 streak 子样本选择,非独立信号。**清醒剂**:OSAP 全 universe 的 PEAD 本尊(EarningsSurprise)2011+ 只剩 t=1.03,与 C17 边缘 +2.01 一致 — 现代大盘 PEAD 本来就薄,streak 是否更厚就是这条的全部赌注,P=2。 |
| I27 | `vrp_xs_pnl` | options-pnl | backlog | 3 | 3 | 3 | 1.2 | 3.6 | **来源:vol_smile 并入(2026-07-20)。机制**:单名 IV−RV 溢价的截面差异,delta-hedged 卖贵买贱 — **与 I3 同一信号、不同判决空间**(I3 判股票收益,这条判期权持仓 P&L;registration 时必须交叉引用,避免双记独立证据)。vol_smile 的市场级 VRP≈0 负结果不预判单名截面(那是水平,这是相对定价)。**文献**:Bali-Hovakimian 线 + 卖方溢价文献(待核)。**成本 3**:RV 管道可移植,引擎待移植。**评价设计开族预注册**(target/成本/尾部)。 |
| I9 | `amihud_illiq` | liquidity-volume | backlog | 1 | 4 | 1 | 0.8 | 3.2 | **P 2→1(2026-07-15,双源判决)**:HXZ NYSE-VW 三 horizon 全 FAIL(0.25%/\|t\|1.20 · 0.34/1.64 · 0.39/1.91,RFS 2020 Table 3),All-EW 1.01/3.49 = 纯微盘;OSAP **发表后 t=0.36、2011+ 反号 −0.83**(重算复核)— 发表后连全 universe EW 都死了。"U1 后上调"降格为:U1 且含小微盘才可能值得,且要先过"发表后已死"这一关。机制/构造/预算同前(\|ret\|/$vol;与 B6 差异仅在分子;I9/I13/I14/I21 家族 ≤3 primaries)。Pastor-Stambaugh beta 不登记(HXZ:EW 下也 FAIL)。 |
| I13 | `turnover_dnr` | liquidity-volume | backlog | 1 | 4 | 1 | 0.8 | 3.2 | **P 2→1(2026-07-15,双源)**:HXZ NYSE-VW 全 horizon FAIL(−0.15/0.61 · −0.16/0.62 · −0.11/0.46),All-EW 才活(−0.86/3.53);OSAP ShareVol **发表后 t=0.29、2011+ 0.15 — 发表后死**(重算复核)。构造论点不变(volume/shares,B7 PIT shares;share vs dollar 口径)。 |
| I10 | `news_coverage_neglect` | news | backlog | 3 | 4 | 4 | 0.9 | 2.7 | **news family 的频率轴,成员中先验最高** — 照搬你 8-K 的教训:频率活(C11)、内容死(C12–C14)。**机制**:媒体忽视溢价 — 无覆盖股票跑赢有覆盖(Fang-Peress 2009 JF,待核 OSAP):投资者认知不完备,无覆盖 = 持有者要求信息不完备补偿;对手盘 = 只买上新闻的票的注意力驱动投资者。**数据(瓶颈,family 共用)**:带时间戳可 PIT 的历史新闻语料 — 免费端 GDELT(2015+,实体→ticker 匹配噪声大)、付费端 Tiingo/Polygon(历史深度待核);**开族前置 = GDELT 实体匹配可行性 QA**(抽样精确率报告;数据工程,不算 look)。**与已测重叠**:强制披露子集(8-K)已测完,增量必须来自 filings 外的媒体报道 → 正交 0.9。**拥挤**:高;忽视效应小盘更强(U1 后上调)。 |
| I8 | `employer_reviews` / activity-nowcast 泛化 | alt-activity | backlog | 3 | 3 | 4 | 1.1 | 2.5 | **I7 的可救部分泛化**:"某处公开活动数据"→"公开活动代理 nowcast 公司基本面"。付费端有确证文献:卫星停车场车流预测零售 earnings surprise(Katona-Painter-Patatoukas-Zeng,数据贵、机构已买断)。免费端可测:Glassdoor 雇主评分变化预测收益(Green-Huang-Wen-Zhou 2019 JFE,t≈3 待核;抓取难+ToS 风险),Google Trends 搜索量(Da-Engelberg-Gao 2011 JF — 注意力效应,历史可回取、PIT 干净),Wikipedia 浏览量、app 排名。**共同瓶颈**:免费源的历史 PIT 可得性参差 — 登记时按"Trends/Wiki 先行(历史可回取)"打成本 4。**拥挤**:Trends 高,Glassdoor 中。 |
| I16 | `news_drift_chan` | news | backlog | 3 | 3 | 4 | 0.9 | 2.0 | **机制**:对公司新闻反应不足 → 有新闻月的收益漂移;对无信息价格波动过度反应 → 无新闻月的收益反转(Chan 2003 JFE,待核)。构造是**条件化**的:用当月 coverage 有/无把动量一分为二,预测两种相反的后续 — 与 B1/B5 动量 controls 的增量正好来自这个条件。**数据**:同 I10 的语料与前置 QA。**容量 3**:需要事件级新闻识别,比 I10 的计数重。**拥挤**:中高。 |
| I28 | `fw_weekend_singles` | options-pnl | parked(imported) | 2 | 2 | 2 | 1.2 | 2.4 | **vol_smile 原生变体整体进口(2026-07-20 治理合并)**:F-W 周末构造,in-sample t **1.48/1.71**,在 vol_smile 侧 parked 并预注册了唯一路径 — **单票 OOS promote-or-kill**。pins 原样保留:不重新调参、不加变体,单票数据齐后按原注册跑一次判生死;此前的 vol_smile 侧 look 历史作为前账本背景记录,不计入本账本 N。P=2(纯自家 in-sample、亚阈值)。 |
| I12 | `congress_trades` | informed-following | backlog | 2 | 2 | 2 | 1.0 | 2.0 | **I11 instinct 的可救部分**:不看重要人物说什么(推特),看他们**被强制披露的交易**。**机制**:议员对政策/拨款有私有信息,STOCK Act 强制 45 天内披露 → 跟单;45 天披露滞后是天然 PIT 可用性(同 13F,比推特时间戳干净得多)。对手盘 = 不读披露的市场。**数据**:`scripts/fetch_congress_trades.py` 已在仓库(有 test,从未评估)— 成本 2。**文献(先验一般)**:Ziobrowski 2004 — 参议员组合 1993–98 超额 ~85bp/月(待核);但 Eggers-Hainmueller 后续样本 ≈ 0;STOCK Act 后的跟单研究多数发现披露滞后之后无 alpha(待核)。媒体热度高(Pelosi tracker 等)= 拥挤度近年陡增。**容量**:2 — 事件稀疏、宽度有限,可能踩 C16 的 MIN_XS 坑,登记时先想好 judge 用事件研究还是 XS-IC。**正交**:与 Form 4 insider family 机制同源(跟知情人)但主体不同,1.0。 |
| I26 | `dispersion_lite` | options-pnl | backlog | 3 | 2 | 4 | 1.3 | 1.95 | **来源:vol_smile 并入(2026-07-20)。机制**:隐含相关溢价 — 指数 IV 隐含的成分相关系统性高于实现(Driessen-Maenhout-Vilkov 2009 JF,发表后 待核)→ 卖指数 vol / 买单名 vol 的相关腿。**数据**:两腿都有(vol_smile SPY clean + singles);**容量 2**:本质是一个聚合交易,不是宽截面;**成本 4**:双腿引擎 + 权重管理,最重的一条。61 名对 SPY 的成分覆盖率是先验风险(截面窄 → 相关复制误差),开族前先算覆盖率,算不过就等 U1 或放弃。 |
| I24 | `wq101_alphas` | price-formulaic | backlog | 1 | 4 | 2 | 0.9 | 1.8 | **用户来源(2026-07-15):WorldQuant 2015 公开的 101 formulaic alphas**(Kakushadze, "101 Formulaic Alphas")。**登记为来源而非单一因子,先验诚实打低**:(a) 无逐条发表 t 值、无发表后追踪(论文只给聚合统计)— 证据等级低于任何 OSAP 条目;(b) 公开十年、全行业实现过 = 拥挤天花板;(c) 多数持仓 0.6–6.4 天,**horizon 错配**:C15 已证明月度平台 20×/yr 换手就吃光 gross,日频轮动在 EOD t+1 平台先天 gate-2 死。**N 税是硬约束**:101 条全测 = +101 selection looks,ceiling 直接爆 — 若测只允许两种预注册形态:101 条合成一个复合信号(单 look),或按预注册标准(低换手 + 月度可持有)先选 ≤3 条。**盘中数据到手后重估**:作为执行层/日内研究素材的价值高于作为月度因子。 |
| I15 | `news_tone_firm` | news | backlog | 2 | 4 | 4 | 0.9 | 1.8 | **机制**:公司新闻负面词密度预测盈利与收益(Tetlock-Saar-Tsechansky-Macskassy 2008 JF,待核)。**先验折扣到 2 的理由**:内容/情绪轴在你的 8-K family 全灭(C12–C14 tone/item 全 rejected),且新闻情绪是最商品化的量产信号(RavenPack/Bloomberg)= 被套利最狠的轴;放进队列只为完整覆盖 family 的三条轴(频率 I10 / 漂移 I16 / 内容 I15),开族时若预算 <3 primaries,先砍这条。**数据**:同 I10。 |
| I11 | `public_figure_posts` | public-figure | backlog | 1 | 1 | 2 | 1.0 | 0.5 | **覆盖特朗普推特/Truth Social + Musk 等市场敏感人物的帖子。机制**:点名公司的帖子(关税、订单、批评)引发即时重定价;可能有过度反应→反转或漂移。**数据(不是瓶颈)**:Trump Twitter Archive 完整带时间戳(至 2021 封号),Truth Social 2022+ 有第三方存档,免费;点名→ticker 匹配量小。**文献(是反证)**:事件研究(Ge-Kurov-Wolfe 等,待核)结论一致 — 效应真实但集中在**分钟级**且多数当日回吐,EOD 无漂移;2017 年起就有秒级跟单 bot(Trump2Cash)。**这正是 C15/C16 的教训重演**:信号存在但 EOD t+1 入场时已结束。**样本**:点名公司的帖子每年几十条、政权依赖(只在任期内有效)→ 容量 1。**结论**:队尾。唯一回路 = 若日后为 vol_smile 建了盘中数据(CLAUDE.md 已排队的 intraday NBBO),分钟级事件研究才可测;届时另行注册。可救部分(跟知情公众人物的**交易**而非言论)→ I12。 |
| I7 | `pentagon_pizza` | alt-activity | backlog | 1 | 1 | 5 | 1.0 | 0.2 | **机制**:地缘军事行动前 Pentagon 加班 → 附近披萨店 Google Maps 实时繁忙度上升 → 先于新闻的 risk-off 信号(目标是市场级 timing:油/防务/VIX,不是横截面)。**数据**:Google Maps popular times 只有实时值,无公开历史档案,ToS 禁爬 — 样本只能前瞻采集数年。**文献**:无同行评审;2025 年起是公开 meme(@PenPizzaReport),事件样本个位数;若有信号,公开后已被分钟级消化。**拥挤**:极高(正因为出名)。**边**:无。**成本**:5。**结论**:队尾;instinct 的可救部分已泛化为 I8 — 机制保留("公开活动数据先于官方信息"),换成有历史、有截面、有文献的数据源。 |

## 基建(不按因子公式排序,横向改变所有 family 先验)

| ID | name | status | 内容 |
|----|------|--------|------|
| U2 | `book_allocator` | backlog | **Book/资本分配层("一个大策略"的融合层,2026-07-20 用户批准登记)**:一个分配器管所有 sleeve — 统一风险预算、保证金/netting(期权 delta 腿与 equity 持仓同账户对消)、统一 P&L 与监控、多 rebalance 时钟。vol_smile 的 A2 vol-managed overlay(唯一幸存者)在此升格为整本书的仓位调节器;intraday 到手后先作全 sleeve 共享执行层(C15 类换手成本的解药)再论独立 sleeve。**触发条件(钉死,防提前施工):≥2 个 sleeve 各有 ≥1 个过确认(confirmation-passed)的策略才开工**;当前 0 accepted、2 candidate(C17/C34),未触发。判决层不融合(每 sleeve 自己的 judge)— 融合只发生在账本层(已做)与 book 层(此项)。 |
| U1 | `universe_sharadar` | backlog | 扩到 Sharadar 全市场含退市 PIT universe(2000+ 名)。修掉每行 C 因子的 survivorship caveat;把栖息地搬到异象实际存活的小盘/难套利段(McLean-Pontiff 残余所在)。第一个标定测试:C17 SUE/PEAD 全市场重跑(PEAD 小盘集中,兼作 C17 的 out-of-design 确认)。成本:数据订阅 + `sharadar.py`/`sharadar_qa.py` 扩展 + 面板重建,约数天到一周。**这是当前最高杠杆的一项投资,优先级高于队列中任何单因子。** **衰减文献定论(2026-07-15,`data/reference/decay_literature/`)**:U1 把测试从"发表异象拟合残留 ≈ 0 的角落"(最大市值/最高流动/低特质波动,McLean-Pontiff Table 8:idio +4.05 p<.001、size −1.49 p=.013、$vol −1.67 p<.01)搬进残余 alpha 实际存活的栖息地,等价于给队列每个难套利异象恢复 +1 P;且 US 是唯一有可靠发表后衰减的市场(Jacobs-Müller 2020)— 本面板是全球最坏栖息地,U1 是对它唯一的结构性修复。 |

## Family 覆盖图(kill log 当数据读;测过哪里、结论是什么)

| family | 状态 | 证据 |
|--------|------|------|
| 10-K/10-Q 文本相似度 | **CLOSED**(v0,永久) | C1–C7, C21/C22/C24:10 变体全灭,paper-faithful +0.06,freshness 修正 +1.09 < 2.0 重开门槛 |
| LLM graph / economic links | closed;唯一回路 = SEC 披露客户 | C8–C10 rejected;C19 parked(用户决定,3-look 一次性协议已钉) |
| 8-K 事件 | 研究上有真效应,live 关闭 | C11 −2.0(candidate,月度);C15 gate-2 成本 ≈ gross;item/tone 轴死(C12–C14) |
| Form 4 insider | 关闭待新构造 | C16 = 成分+慢 regime,无可变现 timing;C20 rejected(episodic 读法未测,需另注册) |
| 12b-25 / SC 13D | 一次性测毕 | C18、C23 rejected |
| XBRL 基本面 | **对新成员关闭**(现 universe) | C17 +2.01 candidate = 唯一存活成员(走自己的 pinned path);C25–C29 五连拒 2026-07-15(−1.22/+0.36/+0.47/−0.92/+0.29,全 < 1.5)— C28/C29 注册时即框定为最后两枪,预算如期执行;大盘栖息地对经典基本面因子不友好,U1 的最强一条证据;U1 后重钉预算重开 |
| options-xs | 未开,队列头部,**预算已钉**(2026-07-15) | 正选 I1–I3(占 ≤9 looks / 开族起 6 周预算;stop:三 primary 全 \|t\| < 1.5 → 关,reopen 2.0);**OSAP 核定(07-15)**:I2 类比 CPVolSpread 发表后 3.25 存活 → **I2 队头**;I1 skew1 发表后 1.20 → P=3;I17 量比发表后死 → P=2(bench);I18 无追踪(bench);xs≈61 → MDE@80% ≈ IC 0.03,每成员注册时重推 power line;IV 面板建设不算 look,member 1 注册前先过单名 IV 覆盖 QA gate。**QA 首轮 2026-07-15**:店内实际 5/61(下载 2026-06-10 中断,已恢复续传);QA 管道建好并抓到 spot 拆股基准 bug — spot.parquet 的 close 是回溯调整值、strike 是当时原值,moneyness 必须用 put-call parity 反推现货(`scripts/qa_singles_iv_coverage.py`);4 个完整名字 100% 配对覆盖 — 格式合格,gate 待下载齐后全 universe 重跑 |
| liquidity-volume | **C30 candidate(FRAGILE,live 关)/ C31 rejected** | 预算钉(账本):≤3 primaries、deadline 2026-08-28、stop \|t\|<1.5 → 永久关、reopen 2.0;**C30=I21 volume_trend:−3.12 → 确认分半 FAIL(上半场 +0.04 / 下半场 −4.60,纯 2020+ regime)→ FRAGILE,live path 关(2026-07-20),救活须 post-2020 regime 假设显式重注册**;**C31=I14 GKM 事件式:sn incr t=−0.10 → rejected(2026-07-18,与 P=2 先验一致)**;余 1 保留 primary 名额(member 3 须 deadline 前另行预注册);I9 ILLIQ / I13 turnover 双源发表后死 → P=1、U1 条件、不占预算;HXZ trading-frictions 类 NYSE-VW 复现率仅 3.8%(102/106 FAIL) |
| industry-momentum | 未开,设计冲突待决 | I19;HXZ NYSE-VW 全 horizon PASS(2.86/3.01/3.57),但 v0 sn judge 按构造消灭跨行业信号 — 开测前须预注册非 sn 评价设计或写理由跳过;注意 OSAP 无行业动量条目("IntMom" 是 Novy-Marx 中期动量,发表后 1.05 弱,勿混淆勿顺手测) |
| seasonality | 未开,**OSAP 判决后降级**(2026-07-15) | I20:HXZ in-sample NYSE-VW 3.43 pass 被 OSAP 发表后追踪(MomSeason −0.20)压倒 → P=1;若想救先核 year-1 对应物(MomSeasonShort) |
| dividend-seasonality | **已开 2026-07-21**(C36=DivYieldST 注册) | 前置成本检查先行并通过:精度 79.1%/召回 97.7%,**DivYieldST break-even 17.1bp 稳过 6bp 线 → C36**;DivSeason 6.8bp 擦线 → member 2 资格冻结,须 C36 结果 + 成本线重新论证的修正案才可花 look;pin:≤2 primaries、deadline 2026-09-01、stop 1.5、reopen 2.0;栖息地对口大盘,不吃 −1 罚 |
| earnings-events(PEAD 线) | **预算已花完 2026-07-20:C32 candidate / C33 rejected** | 裁定:序列构造归此族,基本面族的关闭只覆盖水平/比率特征;pin:≤2 primaries、deadline 2026-08-31、stop 1.5、reopen 2.0;C17 老 path 不占预算;**C32=I23 streak:+1.57 擦线 candidate(构造恒等教训:= C17 值的 streak 选择,非独立信号;每名 IC 浓缩 29% 补不回截面减半)**;**C33=RevenueSurprise:+0.44 rejected(方向对、量级无;OSAP EW 2.67 不过大盘栖息地)**;exhaustion 判读:best 1.57 ≥ 1.5 → 不永久关,但无剩余 look,C32 走 member 级 candidate 流程;OSAP 全 universe PEAD 本尊 2011+ 仅 1.03 — 现代 PEAD 薄,与 C17 边缘 +2.01 一致 |
| price-formulaic | 队尾 | I24 WQ101:无逐条统计、拥挤天花板、horizon 错配;N 税约束 — 只许复合单 look 或预注册选 ≤3;盘中数据到手后作为执行层素材重估 |
| ownership-flow | **预算已尽:C34 candidate / C35 rejected** | pin:≤2 primaries、deadline 2026-08-31、stop 1.5、reopen 2.0;**C34 short_interest_xs:sn incr −1.96(58 月短面板)→ candidate,OOS 靠 forward 累积**;**C35 breadth_13f:+0.63 → rejected(2026-07-21,与注册 power line 预判一致——2011+ 证据 1.73 低于 MDE,明知故买的末位名额)**;13F 管道建成(53 期 / 3,890 万行 / CUSIP 503/503),RIO_* 4 个 gated 存活者要测需 rule-6 预算修订;I6 先验依赖 U1 的部分(难借券段)不变 |
| alt-activity | 未开,先验中低 | I7、I8 |
| news | 未开,已展开成员;数据是瓶颈 | 三条轴:I10 频率(忽视溢价,先验最高 — 照搬 8-K 教训:频率活内容死)/ I16 漂移(Chan 条件动量)/ I15 内容(先验最低,预算紧先砍);开族前置 = GDELT 实体匹配可行性 QA(数据工程,不算 look);强制披露子集(8-K)已测完,增量须来自 filings 外的媒体报道 |
| options-pnl(期权空间,单票) | 未开;**评价设计须开族前预注册** | 2026-07-20 治理合并自 vol_smile(账本行);判决空间 = 期权持仓 P&L,v0 XS-IC judge 不适用 — target/controls/真实 NBBO 价差成本模型/尾部指标开族时钉;引擎按 bs.py verbatim 纪律从 vol_smile 移植;成员 I25 crush(3.9)/ I27 vrp_pnl(3.6,与 I3 同信号异判决,须交叉引用)/ I26 dispersion(1.95)/ I28 F-W(parked,原 pins 进口);全部等 singles 下载(11/61) |
| informed-following | 未开,数据在手 | I12;congress trades 已抓取未评估 |
| public-figure | 队尾,EOD 不可测 | I11;文献说效应是分钟级,回路 = 盘中数据 |

## Inbox(原始想法,一行一条,不排序)

想法随手丢这里(手机上丢 Notion Experiments Inbox 也行,定期搬运)。
补全六字段后领 I-ID 进队列。

- **post-U1 基本面 re-pin 候选**(现 universe family 已关,U1 重开时按此
  优先;HXZ = RFS 2020 Table 3 NYSE-VW \|t\|,OSAP = 发表后 t / 2011+ t,
  重算复核):NetPayoutYield(OSAP 3.78/3.67,mean 1.5%/mo)· XFIN
  (3.56/3.67)· dNoa(HXZ 4.14)· Cop 现金经营盈利(HXZ 3.57)·
  ShareIss5Y(OSAP 3.30/3.32)· Pda(HXZ 3.91)· Cei(HXZ 3.32;OSAP
  2.25)· roaq(OSAP 2.93/2.85)· SP 销售/价格(OSAP 3.20/2.59)· cfp
  (3.01/2.69)· ChTax(2.75/3.09)· Tax(3.19/2.45)· Rdm(HXZ 2.75)·
  Ol(HXZ 2.63)· Noa(HXZ 3.25)。
- **RevenueSurprise**(Jegadeesh-Livnat 2006)— **→ C33(2026-07-20 注册,
  earnings-events member 2,末位预算 primary;注册时复核:发表后 2.76、
  2011+ 2.67 — 旧记 3.03 不可复现,已更正)**。
- **盘中数据待入手(2026-07-15)**:伴侣有 ~5 年盘中数据(粒度未知,
  分享中)。到手核验粒度/覆盖后重估:I11(分钟级事件研究解锁)、I24
  (执行层素材)、C15 daily 变体(盘中入场)、执行成本模型;vol_smile
  的 intraday 队列同受益。**数据未验前不改任何分数。**
- **OSAP 完备性判决(2026-07-20,`reports/survey_github_survivors_
  2026-07-20.md`)**:53 存活者全部入账(7 测/2 控/14 队列/9 锁/21 残余);
  残余实质挖尽——真正未测且在栖息地的只剩 3 个边缘名:**OperProfRD**
  (唯一 VW,profitability 族)、**ShareRepurchase**(回购事件,与
  buyback_blackout 互补)、**DivOmit**(停派息事件,事件稀疏 MIN_XS
  风险)。结构性结论:新因子来自新数据(13F 免费管道 = 解锁 5 条,
  性价比最高;IBES 个人难买走 WRDS/Zacks;U1 让 21 残余 + 15 post-U1
  进栖息地)或新构造,不是再刷基本面表。**JKP 13-cluster 已入
  data/reference/jkp/;CZ Placebos 作 judge 假阳性校准是最便宜的
  基建升级(不算 look)。**
- **crypto-xs vertical(用户提出 2026-07-20;潜在第四 sleeve,排在现有
  承诺之后)**:crypto 截面因子(momentum/size/funding-carry)。先验
  两面已议:栖息地论证成立(机构套利密度低)+ 数据免费 PIT 干净
  (交易所 API)+ 容量无虞;但幸存者偏差比 SP500 恶劣一个量级
  (universe 必须含死币)、wash-trading 假量污染 volume 类信号、
  regime 不稳(整个资产类只有三四个 regime,分半检验先天难看)、
  交易所/托管尾部风险直接进 P&L。文献:Liu-Tsyvinski-Wu JF 2022
  (size/momentum 截面,发表后样本短,待核)。**前置(数据工程,不算
  look)**:含退市的 universe 构建 + wash-trading 筛查 + 日频面板;
  开族按 rule 6 钉预算,judge 移植 XS-IC。**明确排序:在 C17/C34 确认、
  options-xs 三连、U1/分析师数据决定之后**;"好赚钱"不是登记理由,
  栖息地 + 免费数据 + 平台机器可移植才是。 |
  `reports/ideation_agents_2026-07-17.md`;57 条 → 以下 12 条入 Inbox,
  9 条并入既有 family 菜单,其余归档/淘汰;去重逮住 2 条已测尸体
  C25/C23)**:
  - **buyback_blackout**:3/5 agent 独立收敛 — 10b-18 回购盘(大盘最大
    的价格不敏感买家)财报前 ~4-5 周法定沉默,撤走的 bid 无法被套利
    补上;XBRL 回购 $ + 财报日历 + ADV 全在手;无发表横截面。Inbox 头名。
  - **attention_crowdout_pead**:财报日拥挤度条件化 SUE — C17 的直接
    扩展,数据全在手(SUE + 2.02 日期 + 宏观日历);DellaVigna-Pollet、
    Hirshleifer-Lim-Teoh;earnings-events family 归属。
  - **rsu_vest_taxsell**:Form 4 code-F/S 的机械 vest 卖压(文献专门
    丢弃的行),form4_trans.parquet 在手;Form 4 族关闭中 → 2.0 bar。
  - **buyback_insider_divergence**:回购盘 × 内部人净卖出的背离;
    Form 4 + XBRL 在手;新交集;同受 Form 4 2.0 bar。
  - **passive/inelastic 簇**(inelastic-float / passive-gap /
    etf_flow_pressure / thematic-basket 合并):被动持有份额 × 流量的
    过度反应+反转,及基本面意外 × 不弹性持有的欠反应;Gabaix-Koijen、
    Koijen-Yogo、Ben-David 等;13F + ETF holdings 组装为成本。
  - **crowd_exit_liquidity**:top-N 13F 持仓 $ ÷ ADV = 集体退出天数,
    左尾预测;无发表;13F + 价格在手,便宜。
  - **float_shrink_rebalance**:回购缩 float → 指数权重下调 → 被动
    强制卖;shares PIT + rebalance 日历在手。
  - **oct_fye_windowdress**:10 月 FYE 基金的税损卖压(12 月版已被套利,
    10 月切面没有);13F + 收益在手,季节性。
  - **fed_contract_award_flow**:USAspending/FPDS 合同净授予流领先
    政府收入敞口名字的营收;免费、action 日期戳(retro 修改 → PIT 需
    快照纪律);Belo-Gala-Li。
  - **patent_maintenance_lapse**:主动弃缴专利维持费 = 创新线剪枝的
    revealed preference;USPTO 费用事件文件免费、PIT 干净;无发表。
  - **trademark_launch_pipeline**:商标申请领先产品发布;USPTO bulk
    免费、PIT 干净;近乎无发表。
  - **trace_bond_lead / fx_revenue_beta**:跨资产二条 — 发行人债券
    超额收益领先股票坏消息(TRACE,staleness 过滤);FX 篮子 ×
    XBRL 地理分部收入的换算滞后;均免费可 PIT。
  - **options-xs 扩展菜单(family 层记录,I1-I3 判决后再议)**:
    option_term_slope(期限倒挂)、skew_term_twist(短长 skew 扭转,
    无发表)、pcp_implied_stock_gap(parity 缺口 = borrow/知情定位,
    `parity_spot` 已算出原料)、O/S 量比 innovation 形态(I17 的
    level 形态发表后已死,只许 innovation 构造)。
