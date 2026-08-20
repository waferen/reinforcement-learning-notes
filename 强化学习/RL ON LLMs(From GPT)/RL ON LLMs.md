**PPO/GRPO 明明是强化学习算法，按理说应该训练一个 agent 的策略；但大语言模型看起来只是一个预测下一个 token 的模型，它到底哪里来的“策略”？环境又在哪里？**

最关键的一句话是：

> **在 LLM 强化学习里，语言模型本身就是策略（policy），而“环境”通常是一个给回答打分的过程。**

所以不是“PPO 用来训练 LLM 之外的一个策略”，而是：

$$
\boxed{\text{LLM} \equiv \text{Policy}}
$$

然后通过 PPO / GRPO，让这个 policy 更倾向于产生高奖励的回答。

下面我从传统强化学习开始，一层一层把它对应到 LLM。

---

# 1. 先回到你熟悉的强化学习

传统强化学习里：

$$
\text{Agent} \leftrightarrow \text{Environment}
$$

在某个状态 $s_t$ 下：

$$
a_t \sim \pi_\theta(a|s_t)
$$

环境接收到动作：

$$
(s_t,a_t)\rightarrow s_{t+1},r_t
$$

然后 agent 根据奖励更新自己的策略：

$$
\pi_\theta \rightarrow \pi_{\theta'}
$$

例如下棋：

- 状态 (s)：当前棋盘
    
- 动作 (a)：下哪一步
    
- 策略 ($\pi_\theta(a|s)$)：这个局面下每一步的概率
    
- 环境：棋盘规则
    
- 奖励：赢棋 +1，输棋 -1
    

---

# 2. 那 LLM 的“策略”是什么？

这里是第一个非常重要的认知转换。

你可能会觉得：

> “LLM 不就是一个语言模型吗？它怎么会有 policy？”

实际上，语言模型天然就定义了一个策略。

假设输入：

> 今天天气很好，我决定去

模型给出：

$$
P(\text{公园}\mid \text{今天天气很好，我决定去})
$$

也可以是：

$$
P(\text{散步}\mid x)
$$

$$
P(\text{旅游}\mid x)
$$

所以对于 LLM：

$$
\boxed{  
\pi_\theta(a_t|s_t)
=
P_\theta(\text{token}_t|x,\text{previous tokens})  
}
$$

也就是说：

> **LLM 每次选择下一个 token，本质上就是在执行一个 action。**

---

# 3. LLM 强化学习里的 state、action 是什么？

把语言模型套进 RL：

## State

当前已经生成的文本：

$$
s_t=(x,y_1,\ldots,y_{t-1})
$$

其中：

- (x)：用户 prompt
    
- ($y_1,\ldots,y_{t-1}$)：已经生成的 token
    

例如：

> 用户：什么是牛顿第二定律？

模型生成：

> 牛顿第二定律指出

那么当前状态就是：

$$
s_t=  
(\text{用户问题},\text{“牛顿第二定律指出”})
$$

---

## Action

下一个 token：

$$
a_t=y_t
$$

例如：

$$
a_t=\text{“物体”}
$$

---

## Policy

LLM：

$$
\pi_\theta(a_t|s_t)
$$

也就是：

$$
P_\theta(y_t|x,y_{<t})
$$

---

所以：

$$
\boxed{  
\text{LLM就是policy}  
}
$$

这一点是理解 RLHF / GRPO / RLVR 的钥匙。

---

# 4. 那 LLM 的环境在哪里？

这是第二个关键点。

传统 RL：

$$
\text{Agent}  
\rightarrow  
\text{Environment}  
\rightarrow  
\text{Reward}
$$

例如机器人：

$$
\text{动作}  
\rightarrow  
\text{物理世界}  
\rightarrow  
\text{状态变化 + 奖励}
$$

但是 LLM 不一定存在这样一个真实物理环境。

所以 LLM RL 的“环境”可以非常抽象。

最简单的情况下：

$$
\text{LLM回答}  
\rightarrow  
\text{Reward Model}  
\rightarrow  
r
$$

这个 [Reward Model](Reward%20Model.md) 就承担了部分“环境”的作用。

---

# 5. 一个完整的 LLM RL 环境

比如我们训练模型解数学题：

Prompt：

> 求方程 $2x+3=7$ 的解。

LLM 生成：

> (2x=4)，因此 (x=2)。

然后系统检查答案。

如果正确：

$$
r=1
$$

如果错误：

$$
r=0
$$

这里：

### Agent

$$
\text{LLM}
$$

### State

$$
\text{当前 prompt + 已生成 token}
$$

### Action

$$
\text{下一个 token}
$$

### Environment

$$
\text{数学题验证器 / reward function}
$$

### Reward

$$
r\in{0,1}
$$

---

# 6. 但这里有一个有趣的地方

传统 RL 的环境会产生：

$$
s_{t+1}
$$

但是对于 LLM：

$$
s_{t+1}
=
(s_t,a_t)
$$

也就是：

> **LLM 生成 token 之后，新的 state 就是“原来的文本 + 新 token”。**

所以环境实际上非常简单。

例如：

$$
s_0=\text{“解释牛顿第二定律"}
$$

LLM 选择：

$$
a_1=\text{“牛顿"}
$$

于是：

$$
s_1=  
\text{“解释牛顿第二定律 牛顿"}
$$

然后：

$$
a_2=\text{“第二"}
$$

得到：

$$
s_2=  
\text{“解释牛顿第二定律 牛顿 第二"}
$$

继续下去。

---

# 7. 最后得到完整回答

假设生成：

> 牛顿第二定律指出，物体的加速度与所受合力成正比……

完整 trajectory：

$$
\tau

(s_0,a_1,s_1,a_2,\ldots,s_T)
$$

即：

$$
y_1,y_2,\ldots,y_T
$$

---

# 8. 那 Reward 在哪里？

一般是在回答生成完成后。

比如：

Prompt：

> 证明勾股定理

LLM 输出：

> ……

然后 reward model / verifier 看答案。

比如：

$$
R(x,y)=0.87
$$

那么整个回答得到一个奖励。

所以 LLM RL 很常见的一种形式是：

$$
\boxed{  
r_1=r_2=\cdots=r_{T-1}=0,\quad  
r_T=R(x,y)  
}
$$

也就是：

> **前面的 token 通常没有即时奖励，整个回答结束之后才知道好不好。**

这就是一种非常典型的 sparse reward。

---

# 9. 所以 LLM RL 本质上是在干什么？

整个过程其实非常直观：

$$
\boxed{  
Prompt  
\rightarrow  
LLM生成回答  
\rightarrow  
评估回答  
\rightarrow  
得到Reward  
\rightarrow  
更新LLM  
}
$$

然后不断重复。

---

# 10. 为什么 PPO 能训练 LLM？

因为 PPO 根本不在乎：

> “你的 policy 是机器人还是语言模型。”

PPO 只需要：

$$
\pi_\theta(a|s)
$$

而 LLM 天然就有这个东西。

LLM 给每个 token 一个概率：

$$
P_\theta(y_t|x,y_{<t})
$$

这就是 policy。

所以 PPO 完全可以直接作用在 token level 上。

---

# 11. PPO 在 LLM 中到底怎么操作？

假设旧模型：

$$
\pi_{\theta_{\text{old}}}
$$

生成一个回答：

> 因此答案是 42。

我们记录每个 token 当时的概率：

$$
\pi_{\theta_{\text{old}}}(y_t|s_t)
$$

之后修改模型参数，得到新模型：

$$
\pi_\theta
$$

那么重新计算：

$$
\pi_\theta(y_t|s_t)
$$

PPO 看：

$$
r_t(\theta)

\frac{  
\pi_\theta(y_t|s_t)  
}{  
\pi_{\theta_{\text{old}}}(y_t|s_t)  
}
$$

这个东西叫：

$$
\boxed{\text{Probability Ratio}}
$$

---

# 12. 为什么要看这个 ratio？

因为我们希望：

> “让好答案的概率增加一点。”

但是不能一下子增加太多。

否则模型可能发生：

> reward hacking / policy collapse

所以 PPO 会限制更新幅度：

$$
L^{CLIP}

\mathbb E  
\left[  
\min(  
r_tA_t,  
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t  
)  
\right]
$$

意思就是：

> **提高好答案的概率，但别突然提高得太夸张。**

---

# 13. 一个具体例子

假设模型回答数学题：

> $2+2=5$

Reward：

$$
R=0
$$

而另一个回答：

> $2+2=4$

Reward：

$$
R=1
$$

训练很多次之后，模型应该逐渐变成：

$$
P(4|2+2)\uparrow
$$

而：

$$
P(5|2+2)\downarrow
$$

这其实就是：

$$
\boxed{\text{Reward 高的 trajectory → 提高概率}}
$$

$$
\boxed{\text{Reward 低的 trajectory → 降低概率}}
$$

---

# 14. 那是不是每一个 token 都要单独奖励？

通常不需要。

假设：

> 我们要求模型证明一个数学题。

模型生成：

$$
y_1,y_2,\ldots,y_{100}
$$

最后得到：

$$
R=1
$$

那么这个 reward 可以通过 advantage 等方法传播回各个 token。

所以：

$$
A_t
$$

可以理解为：

> **“在当前这个位置选择这个 token，对最终结果到底有多大贡献？”**

这就让 policy gradient 能够知道应该强化哪些生成行为。

---

# 15. 于是整个 LLM PPO 流程就是

可以把它画成：

$$
\boxed{  
\text{Prompt}  
\rightarrow  
\pi_{\theta_{\text{old}}}  
\rightarrow  
\text{Generate}  
\rightarrow  
\text{Answer}  
\rightarrow  
\text{Reward}  
\rightarrow  
\text{Advantage}  
\rightarrow  
\text{PPO}  
\rightarrow  
\pi_\theta  
}
$$

不断循环。

---

# 16. 那 GRPO 又是什么？

GRPO 的核心思想可以理解为：

> **我不一定非要维护 PPO 那么完整的一套 value model，可以通过同一个 prompt 下生成多个回答，直接做相对比较。**

比如：

Prompt：

> $2+2=?$

模型采样 4 个回答：

$$
y_1,y_2,y_3,y_4
$$

得到 reward：

$$
r_1=1
$$

$$
r_2=0
$$

$$
r_3=1
$$

$$
r_4=0
$$

那么平均奖励：

$$
\bar r=\frac{1+0+1+0}{4}=0.5
$$

于是可以计算相对 advantage：

$$
A_i=r_i-\bar r
$$

得到：

$$
A_1=0.5
$$

$$
A_2=-0.5
$$

$$
A_3=0.5
$$

$$
A_4=-0.5
$$

于是模型知道：

> 回答 1、3 比这一批其他回答好 → 提高概率

> 回答 2、4 比这一批其他回答差 → 降低概率

这就是 GRPO 非常重要的思想。

---

# 17. 为什么 GRPO 对 LLM 特别合适？

因为 LLM 很容易：

$$
\text{同一个 prompt}  
\rightarrow  
\text{采样多个回答}
$$

比如：

> 证明这个数学题

可以生成：

### Answer 1

正确

### Answer 2

错误

### Answer 3

正确

### Answer 4

错误

然后比较：

$$
r_1,r_2,r_3,r_4
$$

这种“同 prompt 多采样”的事情，在 LLM 上特别自然。

---

# 18. 那这里的“环境”到底是什么？

这时候要进一步理解：

LLM 的 environment 并不一定是一个真实世界。

它可能是：

### 1. Rule-based Verifier

比如数学：

$$
\text{答案}  
\rightarrow  
\text{检查器}  
\rightarrow  
{0,1}
$$

---

### 2. Reward Model

例如：

$$
(x,y)  
\rightarrow  
RM_\phi(x,y)  
\rightarrow  
r
$$

这个 reward model 本身可能是另外一个神经网络。

---

### 3. 人类反馈

比如：

$$
y_1,y_2
$$

人类认为：

$$
y_1>y_2
$$

然后转化成训练信号。

---

### 4. 外部工具

例如：

LLM：

> 我要计算 $2387\times928$

调用 calculator：

$$
2387\times928=...
$$

工具返回结果。

这个工具就可以看作 environment 的一部分。

---

### 5. Agent 环境

以后更复杂的 LLM agent：

$$
LLM  
\rightarrow  
Browser  
\rightarrow  
Web  
\rightarrow  
Observation  
\rightarrow  
LLM
$$

这时候环境就真的很像传统 RL 了。

---

# 19. 所以其实存在两个层次

这是你理解 LLM RL 时非常值得区分的。

## 第一种：文本环境

环境非常简单：

$$
s_{t+1}

s_t+a_t
$$

最终：

$$
R(x,y)
$$

比如：

$$
\text{数学题}  
\rightarrow  
\text{答案验证}
$$

这类非常常见。

---

## 第二种：真实交互环境

例如 coding agent：

$$
\text{LLM}  
\rightarrow  
\text{写代码}  
\rightarrow  
\text{编译器}  
\rightarrow  
\text{测试}  
\rightarrow  
\text{Observation}  
\rightarrow  
\text{LLM}
$$

这里环境就明显复杂了。

例如：

$$
s_t=  
\text{当前代码 + terminal状态}
$$

动作：

$$
a_t=  
\text{执行shell命令}
$$

环境：

$$
\text{Linux + compiler + filesystem}
$$

reward：

$$
R=  
\text{测试通过率}
$$

这就越来越像传统 RL。

---

# 20. SFT 在这里又处于什么位置？

这又是 LLM 训练流程里特别重要的一环。

通常不会一上来就 RL。

一般是：

$$
\boxed{  
Pretraining  
\rightarrow  
SFT  
\rightarrow  
RL  
}
$$

---

# 21. 第一阶段：Pretraining

训练：

$$
P_\theta(x_t|x_{<t})
$$

数据：

> 大量互联网文本、书籍、代码……

目标：

$$
\mathcal L_{LM}

-\sum_t\log P_\theta(x_t|x_{<t})
$$

最后得到一个：

$$
\boxed{\text{Base Model}}
$$

比如它知道很多东西，但：

> 不一定会很好地回答用户问题。

---

# 22. 第二阶段：SFT

给它：

$$
(x,y^*)
$$

其中：

- (x)：prompt
    
- (y^*)：人工写出的高质量答案
    

直接做监督学习：

$$
P_\theta(y^*|x)
$$

也就是：

> “正确答案应该长这样。”

这样把 base model 变成 instruction-following model。

---

# 23. 第三阶段：RL

现在模型已经会：

> 听懂问题并回答。

接下来希望它进一步优化：

- 正确性
    
- helpfulness
    
- reasoning
    
- safety
    
- style
    
- tool use
    
- 长程任务能力
    

于是：

$$
\boxed{  
\text{SFT model}  
\rightarrow  
\text{RL}  
}
$$

---

# 24. 一个典型 PPO-RLHF 架构

经典 RLHF 大致有：

$$
\text{Prompt}  
\rightarrow  
\text{Policy Model}  
\rightarrow  
\text{Response}
$$

然后：

$$
\text{Response}  
\rightarrow  
\text{Reward Model}  
\rightarrow  
R
$$

此外通常还有：

$$
\text{Reference Model}
$$

用来约束模型不要偏离原来的 SFT 模型太远。

所以 reward 常常不是简单：

$$
R=RM(x,y)
$$

而会加入 KL penalty：

$$
R'

R
-

\beta  
D_{KL}  
(  
\pi_\theta  
\Vert  
\pi_{\text{ref}}  
)
$$

这意味着：

> “回答得好，同时别把模型改得太离谱。”

---

# 25. 为什么需要 Reference Model？

假设 reward model 有漏洞。

LLM 可能发现：

> “只要我疯狂说某种话，就能拿高分。”

于是模型开始 reward hacking。

比如：

$$
R_{RM}(y)=10
$$


于是需要：

$$
D_{KL}(\pi_\theta||\pi_{ref})
$$

限制模型。

可以理解成：

$$
\boxed{  
\text{RL objective}

\text{maximize reward}  
+  
\text{stay close to original model}  
}
$$

---

# 26. 到这里你就可以理解 DeepSeek 这类 reasoning RL 为什么成立了

比如一个数学问题：

$$
x^2-5x+6=0
$$

给模型一个 prompt。

模型自己采样：

$$
y_1,y_2,\ldots,y_G
$$

每一个可能是不同 reasoning trajectory。

例如：

### Response 1

$$
x^2-5x+6=(x-2)(x-3)
$$

所以：

$$
x=2,3
$$

reward：

$$
1
$$

### Response 2

推导错误：

$$
reward=0
$$

---

# 27. GRPO 就会比较这些回答

假设：

$$
r=[1,0,1,0]
$$

然后：

$$
A_i=  
\frac{r_i-\mu}{\sigma}
$$

或者类似的组内归一化。

于是：

$$
A_1>0
$$

$$
A_2<0
$$

模型就逐渐学习：

$$
P(\text{正确 reasoning trajectory})\uparrow
$$

这就是为什么 RL 可以让模型表现出更强的推理能力。

---

# 28. 一个很容易误解的点

你可能会问：

> “奖励明明是整个答案的，为什么能告诉模型具体哪个 token 好？”

这就是 policy gradient 的核心。

因为：

$$
P_\theta(y|x)

\prod_t  
P_\theta(y_t|x,y_{<t})
$$

取 log：

$$
\log P_\theta(y|x)

\sum_t  
\log P_\theta(y_t|x,y_{<t})
$$

所以：

$$
\nabla_\theta \log P_\theta(y|x)

\sum_t  
\nabla_\theta  
\log P_\theta(y_t|x,y_{<t})
$$

最后 reward：

$$
R
$$

会乘上这个梯度。

所以：

$$
\boxed{  
R  
\rightarrow  
\text{整条trajectory的policy gradient}  
\rightarrow  
\text{所有token对应的概率}  
}
$$

---

# 29. 这其实就是“语言模型版的 trajectory”

传统 RL：

$$
s_0  
\xrightarrow{a_1}  
s_1  
\xrightarrow{a_2}  
s_2  
\rightarrow\cdots  
\rightarrow  
R
$$

LLM：

$$
prompt  
\xrightarrow{token_1}  
token_1  
\xrightarrow{token_2}  
token_1token_2  
\rightarrow\cdots  
\rightarrow  
answer  
\rightarrow R
$$

两者结构其实几乎完全一样。

---

# 30. 最终把 PPO/GRPO 映射到 LLM

|传统 RL|LLM RL|
|---|---|
|State (s)|当前上下文 + 已生成 token|
|Action (a)|下一个 token|
|Policy (\pi_\theta)|LLM|
|Trajectory|一次完整回答|
|Environment|verifier / reward model / tool / world|
|Reward|回答质量|
|Value|对当前生成状态的未来 reward 估计|
|Policy update|修改 LLM 参数|
|PPO / GRPO|更新 LLM policy|

---

# 31. 最完整的 LLM RL 流程

现在把所有东西串起来：

```text
                         ┌──────────────────┐
                         │   Pretraining    │
                         │ 大规模文本数据     │
                         └────────┬─────────┘
                                  ↓
                         ┌──────────────────┐
                         │   Base Model     │
                         └────────┬─────────┘
                                  ↓
                         ┌──────────────────┐
                         │       SFT        │
                         │  高质量问答数据    │
                         └────────┬─────────┘
                                  ↓
                         ┌──────────────────┐
                         │  Policy Model    │
                         │      LLM         │
                         └────────┬─────────┘
                                  │
                              Prompt
                                  ↓
                         ┌──────────────────┐
                         │  Generate y      │
                         │ token by token    │
                         └────────┬─────────┘
                                  ↓
                         ┌──────────────────┐
                         │   Environment    │
                         │                  │
                         │ Verifier / RM /  │
                         │ Tool / Simulator │
                         └────────┬─────────┘
                                  ↓
                               Reward
                                  ↓
                         ┌──────────────────┐
                         │ Advantage / GRPO │
                         │      / PPO       │
                         └────────┬─────────┘
                                  ↓
                         ┌──────────────────┐
                         │ Update θ         │
                         │ 更新 LLM 参数      │
                         └────────┬─────────┘
                                  │
                                  └────→ 再次生成
```

---

# 32. 最后回答你最开始的三个疑问

### 疑问 1：PPO / GRPO 训练的不是策略吗？

**是。**

而：

$$
\boxed{\text{LLM本身就是policy}}
$$

因为：

$$
\pi_\theta(a_t|s_t)
=
P_\theta(token_t|context)
$$

---

### 疑问 2：LLM 需要策略吗？

也需要。

只不过传统 RL 的策略是：

$$
\pi(a|s)
$$

LLM 的策略是：

$$
\pi(token|context)
$$

本质上完全一致。

---

### 疑问 3：LLM 强化学习的环境是什么？

没有唯一答案。

可以是：

$$
\boxed{  
\text{Reward Model}  
}
$$

也可以是：

$$
\boxed{  
\text{数学验证器}  
}
$$

也可以是：

$$
\boxed{  
\text{代码编译器 + 单元测试}  
}
$$

也可以是：

$$
\boxed{  
\text{浏览器 / 游戏 / 机器人 / 真实世界}  
}
$$

所以“LLM RL 没有环境”这个说法不太准确，更准确的是：

> **LLM RL 的环境可以被设计成一个非常简单的 reward/verifier，也可以是真正可交互的环境。**

---

# 33. 你可以记住这一张“核心对应关系”

$$
\boxed{  
\begin{aligned}  
\text{State} &\leftrightarrow \text{当前上下文}\\
\text{Action} &\leftrightarrow \text{下一个 token}\\
\text{Policy} &\leftrightarrow \text{LLM}\\
\text{Trajectory} &\leftrightarrow \text{完整回答}\\
\text{Environment} &\leftrightarrow \text{Verifier / RM / Tool / World}\\
\text{Reward} &\leftrightarrow \text{回答质量}\\
\text{PPO/GRPO} &\leftrightarrow \text{更新 LLM}  
\end{aligned}  
}
$$

真正值得你继续往下理解的是：

$$
\boxed{  
\text{LLM为什么可以把“一个最终 reward”反向传播成每个 token 的训练信号？}  
}
$$

这一步一旦搞明白，你就会彻底看懂 **Policy Gradient → PPO → GRPO → LLM reasoning RL** 这条线。