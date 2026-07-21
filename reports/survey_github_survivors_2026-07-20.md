# GitHub 生态调查 + OSAP 存活者完备性判决(2026-07-20,4 agents)

## A. 存活者完备性(用户问题:"post-pub 还有效的都过过了吗")

独立重算 212 个 OSAP predictor(2011+ t ≥ +2.0 且符号一致 = 存活):

```
212 → 208 eligible(4 个 post-pub 月数不足)
    → 53 存活者:
        7 已测(C25/C27/C29/C30/C32/C33/C34)
        2 已是 controls(Mom12m/Mom6m → B1/B5)
       14 已在队列/Inbox(I1/I2 期权 3 个、I22 分红 2 个、post-U1 基本面 9 个)
        9 数据锁死(IBES 4:REV6/ChangeInRecommendation/ForecastDispersion/
          UpRecomm;13F 4:RIO_Volatility/RIO_Turnover/RIO_Disp/
          IO_ShortInterest(微盘 artifact 已排除);上市地 1:ExchSwitch)
       21 残余
```

**残余判决:实质上已挖尽。**21 个残余里 20 个是 EW 全 universe(微盘栖息地)
且全部落在本面板已测已杀的家族内(基本面 C25-C29、earnings-events
C32/C33、动量 controls 张成)。真正未测且在栖息地上的只有 3 个边缘名:
**OperProfRD**(唯一 VW 存活者,但旗舰 GP=C29 已拒)、**ShareRepurchase**、
**DivOmit**(两个事件构造,EW 且容量有限)→ 已记 Inbox。结构性结论:
**新因子只能来自新数据(期权/13F/IBES)或新 universe(U1),不是再刷
基本面表。**面板校准佐证:HXZ-VW passer 在本面板 5 测 1 活,EW 残余
的基率更低。

## B. 数据购买决策的量化基础

- **13F 管道(免费,EDGAR,工程成本)**:解锁 4 个 gated 存活者
  (RIO_*)+ I5 DelBreadth = 5 个条目。**性价比最高的下一个数据工程。**
- **IBES(个人几乎买不到,走 WRDS/Zacks 替代)**:解锁 4 个 gated 存活者
  + C17 的 forecast-SUE 升级。排 U1 与 13F 之后。
- **U1 Sharadar**:让 21 个 EW 残余 + 15 个 post-U1 Inbox 名单全部
  进入其实际栖息地——横向杠杆不变,仍是最高优先购买。

## C. GitHub 可采纳项(三份调查合并,按价值排)

1. **JKP 13-cluster 分类(已抓,data/reference/jkp/)**:153 特征 × 13
   经验聚类 → 家族定义的对照系;免费美股序列(Dec-2025)作独立实现
   交叉验证。License:代码 MIT、数据 CC BY-NC(私用 cross-check 可)。
2. **CZ Placebos 作 judge 假阳性校准**:现成"应该测不出"信号集,跑
   我们的 judge 实测 FDR——最便宜的 judge 校准法(读码重写,GPL 不拷)。
3. **Judge 诊断升级(alphalens-reloaded 约定,Apache-2)**:IC-decay
   曲线(1/5/10/21d)、rank 自相关半衰期、分位单调性检验、分 sector IC
   ——都是"单一月度 IC 藏得住"的失败模式。基建不算 look。
4. **DSR 接线**:emax_null 就是 Bailey-LdP 的 SR*;喂进偏度/峰度修正的
   PSR 即得 DSR。与 purgedcv 参考实现互验。
5. **期权引擎约定(LEAN,Apache-2)**:NBBO touch 成交(买 ask 卖 bid,
   mid→touch 单列成本)、提前指派三元组(临期 ∧ ~5% ITM ∧ 净费后有利)、
   实物/现金交割分流。options-pnl 开族时进评价设计。
6. **Intraday 落地后**:SLMolenaar 冲击曲线实证法(重建 metaorder →
   拟合 I(Q) → OOS 比较函数形)取代拍死的 3bp。
7. **确认证据**:GTJA-191 生存研究(87% OOS 死、幸存者 gross IR 0.27)
   → WQ101/I24 继续压队尾;practitioner 全景的唯一值得注册信号 =
   Cremers-Weinbaum cp spread = **我们的 I2 队头**(独立收敛)。

## D. 陷阱清单(存档)

qlib Alpha158/360 与 GTJA-191 作为 alpha(日频 A 股三重违规);
"2026 还活着的 GTJA top-10"(191 路多重检验的产物);factor timing
(借它的分样严谨,不借信号);awesome-* 清单广度优先浏览(N 膨胀诱惑)。
