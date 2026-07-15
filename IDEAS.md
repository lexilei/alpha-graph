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
| I3 | `iv_rv_spread_xs` | options-xs | backlog | 3 | 3 | 2 | 1.2 | 5.4 | **机制**:单名 VRP 截面差异 = 对冲需求/彩票偏好定价差;比 I1/I2 更偏风险溢价而非信息。**数据**:在手(RV 管道 vol_smile 已有,`signals/rv`)。**文献**:Bali-Hovakimian 2009 — implied−realized spread 预测截面收益(待核)。**拥挤**:中等。**注意**:与 vol_smile 的时序 VRP 负结果不冲突(那是市场级、卖方策略;这是截面相对定价)。 |
| I6 | `short_interest_xs` | ownership-flow | backlog | 3 | 3 | 2 | 1.2 | 5.4 | **机制**:卖空约束下负面信息进价慢;高 SI = 知情空头聚集 → 低收益;对手盘 = 借券成本盲的多头。**文献**:Boehmer-Jones-Zhang 等,发表后在难借券段存活(待核 OSAP)。**数据**:FINRA 双周 SI 免费;Sharadar 含 SI 字段(待核)。**拥挤**:高,但残余集中在小盘难借段。**注意**:S&P 500 大盘易借,先验偏低 — 此条在 universe 扩展(见基建 U1)之后先验显著上调,当前分按现 universe 打。 |
| I5 | `breadth_13f` | ownership-flow | backlog | 3 | 4 | 3 | 1.1 | 4.4 | **机制**:Miller 分歧+卖空约束 — 机构持有广度下降 = 悲观者被挤出价格 → 高估 → 低收益。**数据**:EDGAR 13F 免费,45 天滞后是天然 PIT 可用性(比文本家族干净)。**文献**:Chen-Hong-Stein 2002 JFE — breadth 变化预测截面收益(待核发表后衰减)。**拥挤**:13F 被大量使用,该构造中等。**边**:EDGAR 管道现成。**成本**:3 — 13F 解析出名地脏(CUSIP↔ticker 映射、份额单位错报),QA 量大。与已关的文本家族不同源:这是 ownership 数据不是文本。 |
| I8 | `employer_reviews` / activity-nowcast 泛化 | alt-activity | backlog | 3 | 3 | 4 | 1.1 | 2.5 | **I7 的可救部分泛化**:"某处公开活动数据"→"公开活动代理 nowcast 公司基本面"。付费端有确证文献:卫星停车场车流预测零售 earnings surprise(Katona-Painter-Patatoukas-Zeng,数据贵、机构已买断)。免费端可测:Glassdoor 雇主评分变化预测收益(Green-Huang-Wen-Zhou 2019 JFE,t≈3 待核;抓取难+ToS 风险),Google Trends 搜索量(Da-Engelberg-Gao 2011 JF — 注意力效应,历史可回取、PIT 干净),Wikipedia 浏览量、app 排名。**共同瓶颈**:免费源的历史 PIT 可得性参差 — 登记时按"Trends/Wiki 先行(历史可回取)"打成本 4。**拥挤**:Trends 高,Glassdoor 中。 |
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
| XBRL 基本面 | **OPEN,在测** | C17 +2.01 candidate;C25–C27 已注册(2026-07-15) |
| options-xs | 未开,队列头部 | I1–I3;数据在手 |
| ownership-flow | 未开 | I5、I6;I6 先验依赖 U1 |
| alt-activity | 未开,先验中低 | I7、I8 |

## Inbox(原始想法,一行一条,不排序)

想法随手丢这里(手机上丢 Notion Experiments Inbox 也行,定期搬运)。
补全六字段后领 I-ID 进队列。

- (空)
