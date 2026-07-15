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
6. Promotion 时同步给 family 钉 look 预算 + stop rule(进预注册账本,
   镜像 2026-07-14 的 family/role 修订)。

## 排序

**分 = P(真) × 容量 ÷ 成本 × 正交**

- **P(真)** 1–5:5 = 发表后复现存活且机制清楚;3 = 有文献但衰减明显或
  栖息地与我的 universe 不符;1 = 纯 anecdote、无同行评审证据。
- **容量** 1–5:5 = 宽截面、日频可交易;3 = 窄截面或月频;1 = 每年个位数
  事件(参考 C16 的 MIN_XS 教训:事件因子标准 judge 打不了分)。
- **成本** 1–5:1 = 数据在手、一周内出 primary look;3 = 需要新抓取/解析
  管道;5 = 需要长期前瞻采集或付费数据。
- **正交** 0.8–1.3:对现有 candidate(C11、C17)和 B1–B9 controls 的先验
  正交性;明显同源打 0.8,全新信息源打 1.2–1.3。

## 队列(按分降序;status: backlog → promoted(C-ID) → closed)

| ID | name | family | status | P | 容量 | 成本 | 正交 | 分 | 登记内容 |
|----|------|--------|--------|---|------|------|------|----|----------|
| I1 | `iv_skew_xs` | options-xs | backlog | 4 | 3 | 2 | 1.2 | 7.2 | **机制**:知情交易者先在期权市场表达负面信息(OTM put 需求 → smirk 变陡),股票市场消化滞后数周;对手盘 = 只看股票的投资者。**数据**:61 单名 EOD NBBO 在手(`data/raw_singles/`),vol_smile 的 IV/SVI 管道可移植。**文献**:Xing-Zhang-Zhao 2010 JFQA — smirk 陡度预测未来低收益,风险调整 ~10%/yr、持续数月(待核);OSAP 存活性待核。**拥挤**:期权数据门槛保护,中等。**边**:数据护城河 + 现成期权管道。**成本**:管道移植,~2。61 名截面窄 — 先作 family 先验验证,过了再决定扩数据。 |
| I2 | `cp_iv_spread` | options-xs | backlog | 4 | 3 | 2 | 1.2 | 7.2 | **机制**:put-call parity 偏离 = 期权市场的方向性私有信息,股价滞后跟随;对手盘同 I1。**数据**:同 I1,在手。**文献**:Cremers-Weinbaum 2010 JFQA — call−put IV spread(同 strike/expiry)预测未来一周~一月收益,~50bp/周量级(待核);An-Ang-Bali-Cakici 2014 JF — ΔIV_call 正向、ΔIV_put 负向预测(待核)。**拥挤**:知名但有数据门槛。**边/成本**:同 I1。与 I1 同 family,预算合并钉。 |
| I9 | `liquidity_volume_xs` | liquidity-volume | backlog | 2 | 4 | 1 | 0.8 | 6.4 | **机制**:两条不同的机制并存 — (a) 流动性溢价:持有难交易股票的补偿,对手盘 = 需要即时流动性的投资者(Amihud ILLIQ、turnover);(b) 可见性/注意力:异常放量把股票拉进投资者视野 → 短期需求(Gervais-Kaniel-Mingelgrin 2001 JF 高量溢价)。**数据**:全部在手(价量面板),成本 = 1,全队列最便宜。**文献**:Amihud 2002(ILLIQ,经典);Datar-Naik-Radcliffe 1998(低 turnover → 高收益);GKM 2001(待核 OSAP 存活性)。**反证**:Ben-Rephael-Kadan-Wohl — 流动性溢价 2000 年后大幅衰减;且效应集中在小/微盘,S&P 500 是全市场最液段,先验被栖息地压到 2(U1 之后上调)。**拥挤**:极高 — 价量数据人人都有。**正交 0.8(关键风险)**:B4 volume_zscore 和 B6 log_dollar_volume 已经是 controls,这个 family 一半已被自家基线张成;任何构造必须在增量上打赢它们,构造上要刻意差异化(ILLIQ 分子带 \|ret\|,GKM 用极端量事件定义,而非水平/z-score)。**N 税警告**:美元成本 ≈ 0 但每个 look 照常计 N、抬高全账本 ceiling — family 预算钉紧(建议 ≤3 looks:ILLIQ、turnover、GKM 各一)。Pastor-Stambaugh 流动性 beta 出名地脆,不登记。 |
| I3 | `iv_rv_spread_xs` | options-xs | backlog | 3 | 3 | 2 | 1.2 | 5.4 | **机制**:单名 VRP 截面差异 = 对冲需求/彩票偏好定价差;比 I1/I2 更偏风险溢价而非信息。**数据**:在手(RV 管道 vol_smile 已有,`signals/rv`)。**文献**:Bali-Hovakimian 2009 — implied−realized spread 预测截面收益(待核)。**拥挤**:中等。**注意**:与 vol_smile 的时序 VRP 负结果不冲突(那是市场级、卖方策略;这是截面相对定价)。 |
| I6 | `short_interest_xs` | ownership-flow | backlog | 3 | 3 | 2 | 1.2 | 5.4 | **机制**:卖空约束下负面信息进价慢;高 SI = 知情空头聚集 → 低收益;对手盘 = 借券成本盲的多头。**文献**:Boehmer-Jones-Zhang 等,发表后在难借券段存活(待核 OSAP)。**数据**:FINRA 双周 SI 免费;Sharadar 含 SI 字段(待核)。**拥挤**:高,但残余集中在小盘难借段。**注意**:S&P 500 大盘易借,先验偏低 — 此条在 universe 扩展(见基建 U1)之后先验显著上调,当前分按现 universe 打。 |
| I5 | `breadth_13f` | ownership-flow | backlog | 3 | 4 | 3 | 1.1 | 4.4 | **机制**:Miller 分歧+卖空约束 — 机构持有广度下降 = 悲观者被挤出价格 → 高估 → 低收益。**数据**:EDGAR 13F 免费,45 天滞后是天然 PIT 可用性(比文本家族干净)。**文献**:Chen-Hong-Stein 2002 JFE — breadth 变化预测截面收益(待核发表后衰减)。**拥挤**:13F 被大量使用,该构造中等。**边**:EDGAR 管道现成。**成本**:3 — 13F 解析出名地脏(CUSIP↔ticker 映射、份额单位错报),QA 量大。与已关的文本家族不同源:这是 ownership 数据不是文本。 |
| I10 | `firm_news_drift` | news | backlog | 3 | 4 | 4 | 0.9 | 2.7 | **机制**:有限注意 + 信息扩散慢 → 公司重大新闻后漂移(Chan 2003 JFE:有新闻的股票漂移、无新闻的反转);负面新闻语言预测盈利与收益(Tetlock-Saar-Tsechansky-Macskassy 2008 JF,待核)。对手盘 = 注意力受限、消化慢的投资者。**数据(瓶颈)**:带时间戳、可 PIT 的历史新闻语料 — 免费端 GDELT(2015+,实体→ticker 匹配噪声大)、Common Crawl news;付费端 Tiingo/Polygon news(历史深度待核);机构端 RavenPack 买不起。**与已测的重叠(关键)**:重大新闻的"强制披露子集"= 8-K family,已测完 — 频率活(C11)、内容死(C12–C14)、公告窗死(C27);增量必须来自**未进 filings 的媒体报道**(报道时点先于/独立于披露),而那正是数据贵的部分,故正交只给 0.9。**拥挤**:极高(新闻情绪是商品化产品),衰减快、地平线短(天级),EOD t+1 能剩多少存疑。**成本**:4(语料建设 + 实体匹配管道是真项目)。栖息地:注意力效应小盘更强,U1 后上调。 |
| I8 | `employer_reviews` / activity-nowcast 泛化 | alt-activity | backlog | 3 | 3 | 4 | 1.1 | 2.5 | **I7 的可救部分泛化**:"某处公开活动数据"→"公开活动代理 nowcast 公司基本面"。付费端有确证文献:卫星停车场车流预测零售 earnings surprise(Katona-Painter-Patatoukas-Zeng,数据贵、机构已买断)。免费端可测:Glassdoor 雇主评分变化预测收益(Green-Huang-Wen-Zhou 2019 JFE,t≈3 待核;抓取难+ToS 风险),Google Trends 搜索量(Da-Engelberg-Gao 2011 JF — 注意力效应,历史可回取、PIT 干净),Wikipedia 浏览量、app 排名。**共同瓶颈**:免费源的历史 PIT 可得性参差 — 登记时按"Trends/Wiki 先行(历史可回取)"打成本 4。**拥挤**:Trends 高,Glassdoor 中。 |
| I12 | `congress_trades` | informed-following | backlog | 2 | 2 | 2 | 1.0 | 2.0 | **I11 instinct 的可救部分**:不看重要人物说什么(推特),看他们**被强制披露的交易**。**机制**:议员对政策/拨款有私有信息,STOCK Act 强制 45 天内披露 → 跟单;45 天披露滞后是天然 PIT 可用性(同 13F,比推特时间戳干净得多)。对手盘 = 不读披露的市场。**数据**:`scripts/fetch_congress_trades.py` 已在仓库(有 test,从未评估)— 成本 2。**文献(先验一般)**:Ziobrowski 2004 — 参议员组合 1993–98 超额 ~85bp/月(待核);但 Eggers-Hainmueller 后续样本 ≈ 0;STOCK Act 后的跟单研究多数发现披露滞后之后无 alpha(待核)。媒体热度高(Pelosi tracker 等)= 拥挤度近年陡增。**容量**:2 — 事件稀疏、宽度有限,可能踩 C16 的 MIN_XS 坑,登记时先想好 judge 用事件研究还是 XS-IC。**正交**:与 Form 4 insider family 机制同源(跟知情人)但主体不同,1.0。 |
| I11 | `public_figure_posts` | public-figure | backlog | 1 | 1 | 2 | 1.0 | 0.5 | **覆盖特朗普推特/Truth Social + Musk 等市场敏感人物的帖子。机制**:点名公司的帖子(关税、订单、批评)引发即时重定价;可能有过度反应→反转或漂移。**数据(不是瓶颈)**:Trump Twitter Archive 完整带时间戳(至 2021 封号),Truth Social 2022+ 有第三方存档,免费;点名→ticker 匹配量小。**文献(是反证)**:事件研究(Ge-Kurov-Wolfe 等,待核)结论一致 — 效应真实但集中在**分钟级**且多数当日回吐,EOD 无漂移;2017 年起就有秒级跟单 bot(Trump2Cash)。**这正是 C15/C16 的教训重演**:信号存在但 EOD t+1 入场时已结束。**样本**:点名公司的帖子每年几十条、政权依赖(只在任期内有效)→ 容量 1。**结论**:队尾。唯一回路 = 若日后为 vol_smile 建了盘中数据(CLAUDE.md 已排队的 intraday NBBO),分钟级事件研究才可测;届时另行注册。可救部分(跟知情公众人物的**交易**而非言论)→ I12。 |
| I7 | `pentagon_pizza` | alt-activity | backlog | 1 | 1 | 5 | 1.0 | 0.2 | **机制**:地缘军事行动前 Pentagon 加班 → 附近披萨店 Google Maps 实时繁忙度上升 → 先于新闻的 risk-off 信号(目标是市场级 timing:油/防务/VIX,不是横截面)。**数据**:Google Maps popular times 只有实时值,无公开历史档案,ToS 禁爬 — 样本只能前瞻采集数年。**文献**:无同行评审;2025 年起是公开 meme(@PenPizzaReport),事件样本个位数;若有信号,公开后已被分钟级消化。**拥挤**:极高(正因为出名)。**边**:无。**成本**:5。**结论**:队尾;instinct 的可救部分已泛化为 I8 — 机制保留("公开活动数据先于官方信息"),换成有历史、有截面、有文献的数据源。 |

## 基建(不按因子公式排序,横向改变所有 family 先验)

| ID | name | status | 内容 |
|----|------|--------|------|
| U1 | `universe_sharadar` | backlog | 扩到 Sharadar 全市场含退市 PIT universe(2000+ 名)。修掉每行 C 因子的 survivorship caveat;把栖息地搬到异象实际存活的小盘/难套利段(McLean-Pontiff 残余所在)。第一个标定测试:C17 SUE/PEAD 全市场重跑(PEAD 小盘集中,兼作 C17 的 out-of-design 确认)。成本:数据订阅 + `sharadar.py`/`sharadar_qa.py` 扩展 + 面板重建,约数天到一周。**这是当前最高杠杆的一项投资,优先级高于队列中任何单因子。** |

## Family 覆盖图(kill log 当数据读;测过哪里、结论是什么)

| family | 状态 | 证据 |
|--------|------|------|
| 10-K/10-Q 文本相似度 | **CLOSED**(v0,永久) | C1–C7, C21/C22/C24:10 变体全灭,paper-faithful +0.06,freshness 修正 +1.09 < 2.0 重开门槛 |
| LLM graph / economic links | closed;唯一回路 = SEC 披露客户 | C8–C10 rejected;C19 parked(用户决定,3-look 一次性协议已钉) |
| 8-K 事件 | 研究上有真效应,live 关闭 | C11 −2.0(candidate,月度);C15 gate-2 成本 ≈ gross;item/tone 轴死(C12–C14) |
| Form 4 insider | 关闭待新构造 | C16 = 成分+慢 regime,无可变现 timing;C20 rejected(episodic 读法未测,需另注册) |
| 12b-25 / SC 13D | 一次性测毕 | C18、C23 rejected |
| XBRL 基本面 | OPEN | C17 +2.01 candidate;C25–C27 evaluated → 全部 rejected 2026-07-15(sn incr −1.22/+0.36/+0.47 < 1.5)— 大盘栖息地对经典基本面因子同样不友好,U1 的又一条证据 |
| options-xs | 未开,队列头部 | I1–I3;数据在手 |
| liquidity-volume | 未开 | I9;数据在手但半数被 B4/B6 controls 张成,S&P 500 栖息地先验低,U1 后上调 |
| ownership-flow | 未开 | I5、I6;I6 先验依赖 U1 |
| alt-activity | 未开,先验中低 | I7、I8 |
| news | 未开,数据是瓶颈 | I10;强制披露子集(8-K)已测完,增量须来自 filings 外的媒体报道 |
| informed-following | 未开,数据在手 | I12;congress trades 已抓取未评估 |
| public-figure | 队尾,EOD 不可测 | I11;文献说效应是分钟级,回路 = 盘中数据 |

## Inbox(原始想法,一行一条,不排序)

想法随手丢这里(手机上丢 Notion Experiments Inbox 也行,定期搬运)。
补全六字段后领 I-ID 进队列。

- (空)
