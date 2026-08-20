从一张图我们就能明显看出 [PPO](../PPO.md)与 GRPO 的不同。
![Pasted image 20260820121932](附件/Pasted%20image%2020260820121932.png)
我们从：

> **“LLM 为什么需要 PPO → PPO 为什么要 Critic → GRPO 怎么把 Critic 去掉 → 它到底怎么更新模型”**

这条线理解。

---

# 1. GRPO 是什么？

**GRPO（Group Relative Policy Optimization）**，中文通常叫：

> **组相对策略优化**

它是 DeepSeek 在 DeepSeekMath 等工作中使用的一种强化学习算法。

最核心的思想一句话：

> **让模型针对同一个问题生成一组答案，然后用这一组答案内部的相对奖励，来判断哪些答案应该增加概率、哪些应该降低概率。**

也就是说：

**PPO：**

> “这个答案到底值多少分？”

**GRPO：**

> “同一道题生成的这些答案里，谁表现得更好？”

例如：

```text
问题：
2 + 2 = ?

模型生成 4 个回答：

y₁ = "4"       reward = 1
y₂ = "5"       reward = 0
y₃ = "4"       reward = 1
y₄ = "3"       reward = 0
```

GRPO 会利用这一组答案的奖励：

```text
[1, 0, 1, 0]
```

计算它们的**相对优势**：

```text
好的答案 → 正优势
差的答案 → 负优势
```

然后：

```text
增加好答案的概率
降低差答案的概率
```

---

# 2. 先回忆 PPO

你如果理解 PPO，GRPO 就很好理解。

PPO 通常有：

```text
Actor / Policy
      ↓
生成回答
      ↓
Reward
      ↓
Critic
      ↓
Advantage
      ↓
PPO 更新 Policy
```

其中最重要的问题是：

> **怎么知道某个 action 到底好不好？**

PPO 通常通过：

$$
A_t = Q(s_t,a_t)-V(s_t)
$$

来衡量。

直觉：

```math
\boxed{
A_t>0
\Rightarrow
这个 action 比预期好
}
```

```math
\boxed{
A_t<0
\Rightarrow
这个 action 比预期差
}
```

所以 PPO 需要一个 **Value Model / Critic** 来估计：V(s)

---

# 3. LLM 中 PPO 是怎么工作的？

假设：

```text
Prompt:
证明勾股定理。
```

LLM：

$$
\pi_\theta
$$

生成：

```text
回答：
设直角三角形两直角边为 a,b...
```

然后 Reward Model 或规则奖励系统给它一个 reward：

$$
r=0.8
$$

PPO 要解决：

> 这 0.8 到底意味着这条回答好还是不好？

它不能只看：

$$
r=0.8
$$

因为不同问题的 reward 分布可能完全不同。

所以 PPO 引入：V(s)

例如：

$$
V(s)=0.5
$$

于是：

$$
A=r-V(s)
$$

得到：

$$
A=0.8-0.5=0.3
$$

说明：

> 这次生成比 Critic 原本预期的表现好。

于是增加这条回答的概率。

---

# 4. GRPO 的关键思想

GRPO 发现：

> **对于 LLM 的很多任务，我们其实不一定需要训练一个单独的 Critic。**

怎么办？

让模型针对**同一个 prompt 生成多个回答**。

例如：

```text
Prompt x

        ↓
 ┌──────┼──────┐
 ↓      ↓      ↓
y₁     y₂     y₃ ... y_G
 ↓      ↓      ↓
r₁     r₂     r₃ ... r_G
```

比如：

$$
G=4
$$

得到：

$$
r_1=1.0
$$

$$
r_2=0.2
$$

$$
r_3=0.8
$$

$$
r_4=0.4
$$

那么平均奖励：

$$
\bar r=\frac{1}{4}(1.0+0.2+0.8+0.4)=0.6
$$

标准差：

$$
\sigma_r\approx0.316
$$

于是 GRPO 可以直接定义一种**组内相对优势**：

```math
\boxed{
A_i=
\frac{r_i-\bar r}{\sigma_r}
}
```

因此：

```text
y₁: reward = 1.0 → A₁ > 0
y₂: reward = 0.2 → A₂ < 0
y₃: reward = 0.8 → A₃ > 0
y₄: reward = 0.4 → A₄ < 0
```

这样就知道：

```text
y₁、y₃ → 应该鼓励
y₂、y₄ → 应该抑制
```

**完全没有训练 Critic。**

这就是 GRPO 最核心的思想。

---

# 5. 为什么这样就能代替 Critic？

这是理解 GRPO 最重要的地方。

PPO：

$$
A_i \approx R_i - V(s)
$$

而 GRPO：

```math
\boxed{
A_i=
\frac{R_i-\operatorname{mean}(R)}
{\operatorname{std}(R)}
}
```

你可以发现：

PPO：

> 和一个**学出来的 baseline** 比。

GRPO：

> 和**同组样本的平均表现**比。

所以：

| PPO              | GRPO                  |
| ---------------- | --------------------- |
| 需要 Critic        | 不需要独立 Critic          |
| 学习 (V(s))        | 使用组内平均 reward         |
| $A=R-V(s)$       | $A=(R-\bar R)/\sigma$ |
| 一个回答一个 advantage | 一组回答一起计算              |
| Actor + Critic   | 主要训练 Actor            |

所以 GRPO 的名字：

> **Group Relative**

就非常直白了：

**Group = 一组回答**

**Relative = 相对比较**

**Policy Optimization = 更新策略**

---

# 6. 这里的 Group 到底是什么？

这是很多人第一次看 GRPO 时容易搞混的地方。

不是：

```text
问题 A
问题 B
问题 C
问题 D
```

而是：

```text
同一个问题 x
       ↓
 ┌─────┼─────┬─────┐
 ↓     ↓     ↓     ↓
 y₁    y₂    y₃    y₄
```

例如：

```text
问题：

小明有 3 个苹果，又买了 5 个，
一共有多少个苹果？
```

模型采样：

```text
回答1：8
回答2：7
回答3：8
回答4：9
```

Reward：

```text
1     0     1     0
```

然后：

$$
\mu=0.5
$$

$$
\sigma=0.5
$$

所以：

$$
A=[1,-1,1,-1]
$$

这就是一个 group。

---

# 7. 那 GRPO 到底在优化什么？

这里就开始和 PPO 接上了。

假设：

$$
\pi_{\theta_{\text{old}}}
$$

是旧模型。

经过一次更新之后：

$$
\pi_\theta
$$

变成新模型。

对于生成的 token：

$$
y_{i,t}
$$

计算概率比：

```math
r_{i,t}(\theta)
=

\frac{
\pi_\theta(y_{i,t}|x,y_{i,<t})
}{
\pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,<t})
}
```

这和 PPO 的 ratio 是一样的。

如果：

$$
A_i>0
$$

那么希望：

```math
\pi_\theta(y_i|x)
\uparrow
```

如果：

$$
A_i<0
$$

那么希望：

```math
\pi_\theta(y_i|x)
\downarrow
```

---

# 8. 为什么一个回答只有一个 Reward，却可以给每个 token 算 Advantage？

这是 LLM 强化学习里非常重要的一点。

假设：

```text
回答：

The answer is 42
```

最终只有一个 reward：

$$
R=1
$$

但是回答包含：

```text
The
answer
is
42
```

多个 token。

GRPO 通常把**整个回答的 group-relative advantage**应用到这个回答中的 token 上。

也就是说：

$$
A_i
$$

是针对第 $i$ 个 response 的优势。

但在 token level 上：

$$
A_{i,t}=A_i
$$

因此：

```text
回答 y₁ reward 很高
        ↓
A₁ > 0
        ↓
这个回答里的 token
都倾向于提高概率
```

反过来：

```text
回答 y₂ reward 很低
        ↓
A₂ < 0
        ↓
这个回答里的 token
整体倾向于降低概率
```

当然，实际 GRPO 实现还会加入 KL 项、mask、长度处理等细节。

---

# 9. GRPO 的 Loss

可以先把核心结构理解成：

```math
\boxed{
L_{\mathrm{GRPO}}
=

L_{\mathrm{PPO-like}}
+
\beta L_{\mathrm{KL}}
}
```

其中 PPO-like 部分本质还是：

```math
\min
\left(
r_t(\theta)A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
```

这和 PPO 非常像。

区别主要在：

```math
\boxed{
A_t
}
```

怎么得到。

PPO：

```math
A_t
\approx
R_t-V_\phi(s_t)
```

GRPO：

```math
\boxed{
A_i=
\frac{R_i-\operatorname{mean}(R)}
{\operatorname{std}(R)}
}
```

---

# 10. GRPO 为什么还需要 KL？

这是另外一个非常关键的地方。

假设当前模型：

$$
\pi_\theta
$$

为了得到高 reward，开始疯狂钻 reward model 的漏洞。

比如：

```text
Reward Model：
喜欢详细答案
```

模型发现：

```text
写得越长 → reward 越高
```

于是：

```text
回答：
非常非常非常非常非常非常……
```

reward 可能变高。

这就是典型的：

> **Reward Hacking**

所以需要限制模型不要离原来的语言模型太远。

通常会引入 reference model：

$$
\pi_{\mathrm{ref}}
$$

于是加入 KL penalty：

```math
D_{KL}
\left(
\pi_\theta
\parallel
\pi_{\mathrm{ref}}
\right)
```

直觉就是：

> **你可以为了 reward 改变，但不能把原来的语言能力整个改崩。**

因此：

```text
Reward
  ↑
  │
  │       想提高
  │         ↑
  │         │
  │     Policy
  │         │
  │         ↓
  │      KL约束
  │
Reference Model
```

---

# 11. 所以 GRPO 的完整流程

现在把整个过程串起来。

### Step 1：拿一个 prompt

$$
x
$$

例如：

```text
Solve:
3x + 5 = 20
```

---

### Step 2：模型生成 G 个答案

$$
y_1,y_2,\cdots,y_G
$$

例如：

```text
y₁ = x=5
y₂ = x=4
y₃ = x=5
y₄ = x=6
```

---

### Step 3：Reward Model / Reward Function 打分

$$
R_1,R_2,\cdots,R_G
$$

例如：

$$
[1,0,1,0]
$$

---

### Step 4：计算组内平均值

```math
\mu_R=
\frac{1}{G}\sum_{i=1}^{G}R_i
```

这里：

$$
\mu_R=0.5
$$

---

### Step 5：计算组内相对优势

```math
\boxed{
A_i=
\frac{R_i-\mu_R}
{\sigma_R+\epsilon}
}
```

得到：

$$
[+1,-1,+1,-1]
$$

---

### Step 6：计算新旧 Policy 概率比

```math
r_{i,t}(\theta)
=

\frac{
\pi_\theta(y_{i,t}|x,y_{i,<t})
}{
\pi_{\theta_{\mathrm{old}}}(y_{i,t}|x,y_{i,<t})
}
```

---

### Step 7：PPO-style clipping

让：

$$
r_{i,t}
$$

不能一下子变化太大。

---

### Step 8：加入 KL 约束

限制：

$$
\pi_\theta
$$

不要偏离：

$$
\pi_{\mathrm{ref}}
$$

太远。

---

### Step 9：反向传播

最终：

$$
\nabla_\theta L_{\mathrm{GRPO}}
$$

更新 LLM：

$$
\theta\leftarrow\theta-\alpha\nabla_\theta L
$$

然后进入下一轮。

---

# 12. 你可以把 PPO 和 GRPO 画成这个结构

```text
                 PPO
                  │
        ┌─────────┴─────────┐
        ↓                   ↓
     Policy               Critic
        │                   │
        ↓                   ↓
    Generate             V(s)
        │                   │
        ↓                   │
      Reward                │
        └────────┬──────────┘
                 ↓
             Advantage
                 ↓
              PPO Loss
                 ↓
             Update Policy
```

而 GRPO：

```text
                 GRPO
                  │
                  ↓
               Policy
                  │
                  ↓
         同一个 Prompt
                  │
        ┌─────────┼─────────┐
        ↓         ↓         ↓
       y₁        y₂        ... yG
        ↓         ↓
       R₁        R₂        ... RG
        └─────────┬─────────┘
                  ↓
          Group Relative
             Advantage
                  ↓
          PPO-style Loss
                  +
                KL
                  ↓
             Update Policy
```

最大的变化其实就一个：

```math
\boxed{
\text{Critic}
\quad\longrightarrow\quad
\text{Group Relative Reward}
}
```

---

# 13. 为什么 GRPO 特别适合数学推理？

这就涉及 DeepSeekMath 为什么使用它。

数学题有一个非常好的特点：

> **很多时候可以自动验证答案是否正确。**

比如：

```text
2x + 4 = 10
```

模型生成：

```text
回答 A：x=3
回答 B：x=4
回答 C：x=3
回答 D：x=5
```

可以直接通过程序验证：

```text
A → 正确 → 1
B → 错误 → 0
C → 正确 → 1
D → 错误 → 0
```

根本不一定需要人工标注。

所以：

```math
\boxed{
\text{可验证任务}
\rightarrow
\text{非常适合 GRPO}
}
```

比如：

* 数学
* 编程
* 有明确答案的推理任务
* 可以写 verifier 的任务

---

# 14. GRPO 和你刚学的 GAE 有什么关系？

这个地方非常值得注意。

你刚才问过 **GAE（Generalized Advantage Estimation）**。

GAE 是：

> **怎么估计 Advantage。**

GRPO 则是：

> **不用 Critic，而通过 group reward 来构造 Advantage。**

所以二者不是同一个层级。

可以这样看：

```text
              Advantage Estimation
                       │
          ┌────────────┴────────────┐
          ↓                         ↓
        PPO                      GRPO
          │                         │
          ↓                         ↓
       GAE / Critic           Group Relative
                              Reward
```

PPO 中：

$$
A_t
$$

可能通过 GAE 得到：

$$
\hat A_t^{GAE}
$$

而 GRPO：

```math
\hat A_i
=

\frac{R_i-\mu_R}{\sigma_R}
```

所以：

> **GAE 和 GRPO 不是竞争关系。**

而是：

**GAE 是 Advantage Estimation 方法；GRPO 是一种 Policy Optimization 方法。**

---

# 15. 最后给你一个非常重要的理解

你可以把整个 LLM RL 的演化先记成：

```text
REINFORCE
   │
   │  加 baseline / Critic
   ↓
Actor-Critic
   │
   ↓
PPO
   │
   │  PPO需要Critic
   │
   │  → Critic训练成本高
   │  → Value估计可能不稳定
   ↓
GRPO
   │
   │  同一个问题生成多个答案
   │
   │  用组内相对Reward
   ↓
不需要独立 Critic
```

所以 **GRPO 最核心的创新并不是“把 PPO 的公式改了一点”**。

真正值得记住的是：

```math
\boxed{
\text{PPO：}
\quad
\text{“这个回答比 Critic 预测的好多少？”}
}
```

而：

```math
\boxed{
\text{GRPO：}
\quad
\text{“这个回答比同组其他回答好多少？”}
}
```

这就是为什么叫 **Group Relative Policy Optimization**。

如果把你最近正在学的 **PPO → GAE → GRPO → DeepSeek-R1** 串起来，下一步最值得讲的是 **“GRPO 里的 reward 到底怎么变成 token-level loss，以及它和 PPO 的 GAE 在数学上到底差在哪”**。这一步搞明白，基本就能真正看懂 DeepSeek-R1 的 RL 训练代码了。
