> **Reward Model 是“裁判”，Value Model 是“预测未来能拿多少分的人”，Reference Model 是“防止运动员跑偏的旧自己”，Policy Model 才是真正被训练的运动员。**

---

# 1. 先把四个模型摆在一起

在经典 PPO-RLHF 中，经常会看到：

$$
\boxed{  
\text{Policy}  
+  
\text{Value}  
+  
\text{Reward}  
+  
\text{Reference}  
}
$$

它们的职责完全不同。

|模型|干什么|参数是否训练|
|---|---|---|
|Policy Model|生成回答|✅ 训练|
|Reward Model|给完整回答打分|❌ 通常冻结|
|Value Model / Critic|预测当前状态未来能拿多少 reward|✅ 训练|
|Reference Model|作为原始模型，约束 Policy 不要跑太远|❌ 冻结|

最容易混淆的是：

$$
\boxed{\text{Reward} \neq \text{Value}}
$$

这是今天最重要的一点。

---

# 2. Reward 和 Value 到底有什么区别？

先回到传统 RL。

假设你在下棋。

当前局面：

$$
s_t
$$

你走了一步：

$$
a_t
$$

最后赢棋：

$$
R=1
$$

那么：

### Reward

回答：

> **“这次最终到底拿了多少分？”**

它是一个实际得到的结果。

---

### Value

回答：

> **“站在现在这个状态看，未来大概能拿多少分？”**

也就是：

$$
V^\pi(s)

\mathbb E_\pi  
\left[  
\sum_{k=t}^{T}\gamma^{k-t}r_k  
\mid s_t=s  
\right]
$$

所以：

$$
\boxed{  
R=\text{实际结果}  
}
$$

$$
\boxed{  
V(s)=\text{对未来结果的预测}  
}
$$

---

# 3. 用下棋最容易理解

假设你现在有一个棋局状态：

$$
s
$$

最后：

> 赢了。

因此：

$$
R=1
$$

但是在你还没下完的时候，Value Model 可能估计：

$$
V(s)=0.7
$$

意思：

> “从这个局面出发，按照现在的策略，我估计你最终赢棋的期望是 0.7。”

所以：

**Reward 是回头看的。**

> 最后到底赢没赢？

**Value 是向前看的。**

> 从现在这个状态看，未来大概会怎么样？

---

# 4. 放到 LLM 里面

现在把棋盘换成文本。

Prompt：

> 解释为什么天空是蓝色的。

模型开始生成：

$$
y_1,y_2,\ldots,y_T
$$

假设完整回答最后得到：

$$
R=0.9
$$

Reward Model 说：

> “这整个回答挺不错，给 0.9 分。”

但是在生成到一半的时候，比如：

> “由于太阳光……”

此时 Value Model 会预测：

$$
V(s_t)=0.82
$$

它是在说：

> “根据目前已经写出的内容，接下来继续生成的话，我预测最终能得到大约 0.82 的回报。”

---

# 5. 为什么 PPO 需要 Value Model？

这是因为 PPO 不是简单地：

> 好答案增加概率，坏答案减少概率。

它需要知道：

> **这个 action 到底“比预期好多少”？**

这就是：

$$
\boxed{Advantage}
$$

---

# 6. Advantage 是什么？

最经典的定义：

$$
\boxed{  
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)  
}
$$

意思是：

> 执行动作 $a$ 之后的表现，比“我本来对这个状态的预期”好多少。

例如：

当前状态：

$$
s
$$

Value Model 预测：

$$
V(s)=0.5
$$

然后你选择了 token：

> “所以”

最终结果很好：

$$
Q(s,a)=0.9
$$

那么：

$$
A(s,a)=0.9-0.5=0.4
$$

说明：

> 这个动作比预期好。

所以：

$$
A>0
$$

→ 应该增加这个 action 的概率。

反之：

$$
A<0
$$

→ 应该降低这个 action 的概率。

---

# 7. 这就解释了 PPO 是怎么训练 LLM 的

假设模型生成：

> 因此答案是 42。

最后 Reward：

$$
R=1
$$

Value Model 在生成过程中不断估计：

$$
V(s_1),V(s_2),...,V(s_T)
$$

于是计算 Advantage：

$$
A_1,A_2,\ldots,A_T
$$

然后 PPO 根据：

$$
A_t
$$

判断：

> 哪些 token 对这次成功回答贡献较大？

最终更新：

$$
\pi_\theta
$$

---

# 8. 你可以把它想象成一个老师

Policy：

> 学生

Reward：

> 最后的考试成绩

Value：

> 班主任根据你现在的表现预测你最后能考多少分

Advantage：

> “你这次这个操作，比老师原来预计的表现好多少？”

例如：

老师原来估计：

$$
70
$$

结果：

$$
90
$$

那么：

$$
A>0
$$

说明：

> 这次策略不错，应该多做类似的事情。

---

# 9. 那 Reward Model 又是什么？

Reward Model 更像：

> **阅卷老师 / 裁判**

它直接看完整回答：

$$
(x,y)
$$

然后：

$$
R_\phi(x,y)=0.9
$$

所以三者职责：

$$
\boxed{  
\begin{aligned}  
Reward &= \text{最终裁判分数}\\
Value &= \text{未来得分预测}\\
Advantage &= \text{实际表现相对预测好多少}  
\end{aligned}  
}
$$

---

# 10. 为什么不能直接拿 Reward 当 Advantage？

这是 PPO 中一个很关键的问题。

假设：

$$
R=1
$$

你知道最终赢了。

但是你不知道：

> **是哪一步做得好。**

对于一个长度 1000 token 的回答：

$$
y_1,y_2,\ldots,y_{1000}
$$

最后统一：

$$
R=1
$$

如果直接把：

$$
A_t=1
$$

那么所有 token 都被同等强化。

这太粗糙了。

Value Model 就是帮助我们判断：

> 某个 token 出现在这个状态下，到底比正常预期好多少。

---

# 11. 这就是 Credit Assignment

也就是：

$$
\boxed{\text{Credit Assignment}}
$$

中文可以理解成：

> **到底是谁为最终成功“立了功”？**

这是 RL 的核心难题之一。

例如：

```text
Prompt
 ↓
token 1
 ↓
token 2
 ↓
token 3
 ↓
token 4
 ↓
……
 ↓
最终 reward = 1
```

我们只有：

$$
R=1
$$

但需要估计：

$$
A_1,A_2,A_3,\ldots,A_T
$$

这就是 Value / GAE 等技术发挥作用的地方。

---

# 12. GAE 又是什么？

你之前问过 Bootstrap，这里正好接上。

PPO 中很常用：

$$
\boxed{\text{GAE = Generalized Advantage Estimation}}
$$

先定义 TD error：

$$
\delta_t

r_t+\gamma V(s_{t+1})-V(s_t)
$$

然后：

$$
A_t^{GAE(\gamma,\lambda)}

\sum_{l=0}^{T-t}  
(\gamma\lambda)^l\delta_{t+l}
$$

你暂时不用死记公式。

直觉是：

> **利用 Value Model 的预测，一步一步估计当前动作到底比预期好多少。**

---

# 13. 这里又出现你之前问过的 Bootstrap

还记得 Bootstrap 吗？

这里就是：

$$
r_t+\gamma V(s_{t+1})
$$

直接用：

$$
V(s_{t+1})
$$

去估计未来，而不是等整个 episode 全部结束。

所以：

$$
\boxed{  
\text{Value Model + TD + GAE}  
}
$$

共同解决：

> **最终 reward 如何分配到每个 token？**

---

# 14. 那 Reference Model 又是什么？

这个模型和 Reward / Value 完全不是一回事。

Reference Model 通常就是：

$$
\boxed{  
\text{SFT Model}  
}
$$

复制一份出来，然后冻结。

例如：

```text
SFT Model
   ├────→ Reference Model（冻结）
   │
   └────→ Policy Model（继续训练）
```

---

# 15. 为什么需要 Reference？

假设 Reward Model 喜欢某一种答案。

PPO 就疯狂优化：

$$
R_\phi(x,y)
$$

模型可能越来越偏。

极端情况下：

$$
\pi_\theta
$$

和最初的模型已经完全不是一个东西了。

甚至语言能力都可能崩掉。

所以加一个：

$$
\pi_{ref}
$$

约束：

$$
\pi_\theta  
\approx  
\pi_{ref}
$$

---

# 16. 怎么约束？

最典型的就是 KL divergence：

$$
D_{KL}  
\left(  
\pi_\theta  
\Vert  
\pi_{ref}  
\right)
$$

于是总体 reward 可以写成：

$$
\boxed{  
R_{total}

R_{RM}
-

\beta D_{KL}  
}
$$

含义：

> 奖励高很好，但是别离原来的模型太远。

所以 Reference Model 更像：

> **保险丝 / 刹车。**

---

# 17. 于是经典 PPO-RLHF 的完整结构就是

现在你可以看到整个系统：

```text
                   Prompt
                      ↓
               ┌─────────────┐
               │   Policy    │
               │    LLM      │
               └──────┬──────┘
                      ↓
                   Response
                      ↓
              ┌───────┴───────┐
              ↓               ↓
       Reward Model      Reference Model
              ↓               ↓
          Reward R          KL penalty
              └───────┬───────┘
                      ↓
                 Total Reward
                      ↓
               Advantage A_t
                      ↑
                      │
                Value Model
                      ↓
                PPO Update
                      ↓
                   Policy
```

---

# 18. 那为什么 GRPO 可以不要 Value Model？

这就是 GRPO 最值得理解的地方。

你之前已经知道：

GRPO 会针对同一个 prompt 生成：

$$
G
$$

个回答：

$$
y_1,y_2,\ldots,y_G
$$

然后得到：

$$
r_1,r_2,\ldots,r_G
$$

例如：

$$
[1,0,1,0]
$$

---

# 19. GRPO 的关键思想

既然这几个回答面对的是：

$$
\boxed{\text{同一个 Prompt}}
$$

那么可以直接比较它们。

例如：

$$
\mu=\frac1G\sum_i r_i
$$

然后：

$$
A_i

\frac{r_i-\mu}{\sigma}
$$

也就是说：

> 不需要一个 Value Model 来告诉我“这个状态通常应该拿多少分”。

我直接看：

> **这一批回答里面，你比其他人好还是差。**

---

# 20. 举个例子

同一个数学题：

$$
x^2-5x+6=0
$$

采样 4 个回答：

|回答|Reward|
|---|--:|
|A|1|
|B|0|
|C|1|
|D|0|

平均：

$$
\mu=0.5
$$

所以：

$$
A_A=+0.5
$$

$$
A_B=-0.5
$$

$$
A_C=+0.5
$$

$$
A_D=-0.5
$$

于是：

> A、C 应该强化。

> B、D 应该削弱。

---

# 21. 这和 Value Model 在做什么事情？

传统 PPO：

$$
A_t  
\approx  
R-V(s_t)
$$

它需要：

$$
V(s_t)
$$

也就是：

> “这个状态本来应该值多少？”

而 GRPO：

$$
A_i  
\approx  
r_i-\operatorname{mean}(r_1,\ldots,r_G)
$$

它相当于说：

> “不用预测一个绝对值，我直接做组内比较。”

所以：

$$
\boxed{  
\text{PPO：依赖 Value Model 做 baseline}  
}
$$

$$
\boxed{  
\text{GRPO：用 group statistics 做 baseline}  
}
$$

---

# 22. 这就是为什么 GRPO 可以省掉 Critic

经典 PPO：

$$
\text{Policy}  
+  
\text{Value}  
+  
\text{Reward}  
+  
\text{Reference}
$$

GRPO 大致：

$$
\text{Policy}  
+  
\text{Reward}  
+  
\text{Reference}
$$

至少在“通过组内奖励做 advantage”这个核心设计上，不需要单独训练一个 critic/value model。

于是训练系统更简单。

这对于超大语言模型非常有吸引力，因为：

> **训练一个和 LLM 差不多规模的 Value Model，本身就是非常贵的。**

---

# 23. 但 GRPO 是不是完全不需要“价值”的思想？

不能简单这么说。

它不是说：

> Value Function 这个概念从 RL 世界消失了。

而是：

> **它换了一种方式估计 baseline / relative advantage。**

也就是说：

$$
\text{PPO}  
\rightarrow  
V(s)  
\rightarrow  
A
$$

而 GRPO 更接近：

$$
\text{Group Rewards}  
\rightarrow  
\text{Relative Advantage}
$$

---

# 24. 现在再看 PPO 和 GRPO

你就会发现它们真正的区别其实不是：

> “一个是 RL，一个不是。”

而是：

$$
\boxed{  
\text{如何估计 Advantage}  
}
$$

---

## PPO

先学：

[  
V(s)  
]

然后：

$$
A\approx R-V(s)
$$

---

## GRPO

同一个 prompt 生成多个 response：

$$
r_1,\ldots,r_G
$$

直接比较：

$$
A_i  
\approx  
\frac{r_i-\mu}{\sigma}
$$

---

# 25. 最后一张图把所有概念彻底串起来

```text
                    ┌─────────────────┐
                    │     Prompt      │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │     Policy      │
                    │      LLM        │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │    Response     │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Reward / Verifier│
                    └────────┬────────┘
                             ↓
                       Actual Reward
                             │
                ┌────────────┴────────────┐
                │                         │
                ↓                         ↓
          PPO / Critic                 GRPO
                │                         │
                ↓                         ↓
       Value Model V(s)          Group Rewards
                │                         │
                ↓                         ↓
        Advantage A_t          Relative Advantage
                │                         │
                └────────────┬────────────┘
                             ↓
                    ┌─────────────────┐
                    │ Policy Gradient │
                    │    PPO/GRPO     │
                    └────────┬────────┘
                             ↓
                       Update LLM
```

同时旁边还有一根“刹车线”：

$$
\boxed{  
\text{Policy}  
\leftrightarrow  
\text{Reference Model}  
\rightarrow  
KL\ penalty  
}
$$

---

# 26. 你现在可以把四个东西用一句话记住

$$
\boxed{  
\begin{aligned}  
\text{Policy} &: \text{我怎么回答？}\\
\text{Reward} &: \text{这次回答到底好不好？}\\
\text{Value} &: \text{现在这个状态未来大概能有多好？}\\
\text{Reference} &: \text{你别和原来的自己跑得太远。}  
\end{aligned}  
}
$$

而：

$$
\boxed{  
\text{Advantage}

\text{实际表现}
-

\text{预期表现}  
}
$$

所以你前面学的 **Bootstrap、Q、V、Policy、PPO、GRPO** 现在其实已经串起来了：

$$
\boxed{  
Q(s,a)  
\rightarrow  
V(s)  
\rightarrow  
A(s,a)  
\rightarrow  
\text{Policy Update}  
}
$$

只是到了 LLM 这里：

$$
s=\text{上下文}
$$

$$
a=\text{token}
$$

$$
\pi=\text{LLM}
$$

而 GRPO 的一个核心变化就是：

$$
\boxed{  
\text{不用专门训练 }V(s)，  
\text{直接用同一 prompt 的多个回答做相对比较。}  
}
$$

下一层其实就到了一个特别有意思的问题：**“为什么 GRPO 只看最终 reward，却能够把一个复杂的长 reasoning 过程学出来？”** 这里会真正碰到 **token-level advantage、sequence-level reward、credit assignment、长度偏置，以及为什么 DeepSeek-R1 这类 reasoning RL 能自己长出更长的思维链**。