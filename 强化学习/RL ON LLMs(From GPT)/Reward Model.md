> **这个 Reward 到底是谁给的？Reward Model 又是怎么来的？**

最经典的答案是：**人类先提供偏好数据，再用这些数据训练一个 Reward Model。**

---

# 1. Reward Model 本质上是什么？

Reward Model（RM）其实就是一个普通的神经网络，它学的是：

$$
\boxed{  
R_\phi(x,y)\rightarrow \text{一个分数}  
}
$$

输入：

- (x)：用户问题
    
- (y)：LLM 的回答
    

输出：

$$
R_\phi(x,y)\in\mathbb R
$$

这个分数被当成“这个回答有多好”。

比如：

**Prompt**

> 什么是牛顿第二定律？

两个回答：

### Answer A

> 牛顿第二定律表明，物体的加速度与所受合力成正比……

### Answer B

> 牛顿第二定律就是牛顿发现的第二个定律。

人类显然更偏好 A。

Reward Model 最终就是希望学会：

$$
R_\phi(x,A)>R_\phi(x,B)
$$

所以注意：

> **Reward Model 不是天生就知道什么叫“好回答”。它本身也是训练出来的。**

---

# 2. 那谁来训练 Reward Model？

最经典的答案：

$$
\boxed{\text{人类偏好数据}}
$$

例如给人类一个问题：

> 解释什么是过拟合。

然后让一个 LLM 生成多个回答：

$$
y_1,y_2,y_3,y_4
$$

然后让人类比较：

$$
y_2 > y_4 > y_1 > y_3
$$

或者更简单：

$$
y_A>y_B
$$

意思就是：

> 人类更喜欢回答 A，而不是 B。

这就是 **preference data（偏好数据）**。

---

# 3. 为什么不直接让人给一个绝对分数？

理论上可以：

> 这个回答打 7 分。

但这种数据有一个问题：

不同人对“7 分”的理解不一样。

一个人觉得：

$$
7/10
$$

另一个人觉得：

$$
8/10
$$

两个人的尺度可能不一致。

但是比较：

> A 和 B 哪个更好？

往往更容易。

所以 RLHF 非常经典的一种方法就是：

$$
\boxed{\text{Pairwise Preference}}
$$

也就是两两比较。

---

# 4. 一个完整的数据长什么样？

比如：

```text
Prompt:
解释一下什么是梯度下降。

Response A:
梯度下降是一种用于优化函数的方法，它沿着损失函数梯度的反方向更新参数……

Response B:
梯度下降就是让模型不断学习，然后找到最好的答案。

Human preference:
A > B
```

于是我们得到：

$$
(x,y_w,y_l)
$$

其中：

- (y_w)：winner，更好的回答
    
- (y_l)：loser，更差的回答
    

---

# 5. 然后怎么从这个训练 Reward Model？

这一步非常漂亮。

Reward Model 的目标不是：

> “我要预测人类给几分。”

而是：

> **“对于人类更喜欢的回答，我应该给更高分。”**

所以希望：

$$
R_\phi(x,y_w)>R_\phi(x,y_l)
$$

于是可以定义一个概率：

$$
P(y_w\succ y_l)

\sigma  
\left(  
R_\phi(x,y_w)-R_\phi(x,y_l)  
\right)
$$

其中：

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

这就是 Bradley-Terry / pairwise preference 这一类思路。

训练目标：

$$
\boxed{  
\mathcal L_{RM}

-\log  
\sigma  
\left(  
R_\phi(x,y_w)-R_\phi(x,y_l)  
\right)  
}
$$

---

# 6. 直觉上它在干什么？

假设：

$$
R(A)=0.2
$$

$$
R(B)=0.1
$$

那么：

$$
R(A)-R(B)=0.1
$$

模型觉得：

> A 比 B 好一点。

但人类说：

$$
A>B
$$

那就继续训练，让：

$$
R(A)-R(B)
$$

变得更大。

例如：

$$
R(A)=2.5
$$

$$
R(B)=-1.2
$$

这时候：

$$
R(A)-R(B)=3.7
$$

模型就很明确地认为 A 比 B 好。

---

# 7. 所以整个 Reward Model 训练流程是

你可以理解成：

$$
\boxed{  
\text{LLM生成多个答案}  
\rightarrow  
\text{人类比较}  
\rightarrow  
\text{Preference Dataset}  
\rightarrow  
\text{训练RM}  
}
$$

具体一点：

```text
                 Prompt
                    ↓
              SFT LLM
                    ↓
        ┌───────────┼───────────┐
        ↓           ↓           ↓
      Answer A    Answer B    Answer C
        └───────────┼───────────┘
                    ↓
               人类标注
                    ↓
             A > C > B
                    ↓
          Preference Dataset
                    ↓
             Train Reward Model
                    ↓
             Reward Model
```

---

# 8. Reward Model 通常长什么样？

一个很容易误解的地方：

> RM 并不一定是一个专门设计的奇怪网络。

通常可以直接基于一个语言模型。

例如：

$$
\text{Transformer Backbone}  
+  
\text{Reward Head}
$$

也就是：

```text
输入：
Prompt + Response
       ↓
 Transformer
       ↓
 Hidden States
       ↓
 Reward Head
       ↓
 一个标量 reward
```

例如最后：

$$
R_\phi(x,y)\in\mathbb R
$$

所以它和普通 LLM 很像，只不过：

**普通 LLM：**

$$
hidden\rightarrow vocab\ logits
$$

预测下一个 token。

**Reward Model：**

$$
hidden\rightarrow scalar
$$

预测一个整体分数。

---

# 9. 那 RM 怎么知道“什么是好”？

这个问题特别重要。

答案其实有点哲学味：

> **RM 并不知道什么是“客观的好”，它学习的是人类提供的偏好。**

所以：

$$
\text{Human preference}  
\rightarrow  
\text{Reward Model}  
\rightarrow  
\text{LLM behavior}
$$

换句话说：

$$
\boxed{  
\text{RLHF 本质上是在把人类偏好压缩进一个模型里}  
}
$$

---

# 10. 于是经典 RLHF 有三个模型

你经常会看到：

$$
\boxed{  
\text{Policy Model}  
+  
\text{Reward Model}  
+  
\text{Reference Model}  
}
$$

分别是：

### Policy Model

真正被训练的 LLM：

$$
\pi_\theta
$$

它负责回答。

---

### Reward Model

负责打分：

$$
R_\phi(x,y)
$$

它告诉 policy：

> 你这个回答好不好。

---

### Reference Model

通常是冻结的 SFT 模型：

$$
\pi_{\text{ref}}
$$

用于约束 policy 不要跑得太远。

---

# 11. 那 PPO 的时候具体发生什么？

现在把它们串起来：

### 第一步：Prompt

$$
x
$$

进入 Policy：

$$
\pi_\theta
$$

生成：

$$
y
$$

---

### 第二步：Reward Model 打分

$$
R_\phi(x,y)=2.73
$$

---

### 第三步：计算 KL 惩罚

例如：

$$
D_{KL}  
(\pi_\theta||\pi_{\text{ref}})
$$

得到：

$$
R_{\text{total}}

R_\phi
-

\beta KL
$$

---

### 第四步：PPO

使用：

$$
R_{\text{total}}
$$

计算 advantage，然后更新：

$$
\theta
$$

最终：

$$
\pi_\theta  
\rightarrow  
\pi_{\theta'}
$$

也就是：

> 让 LLM 更倾向于生成 Reward Model 喜欢的回答。

---

# 12. 到这里会出现一个非常重要的问题

你可能已经感觉到了：

> **等等，那 Reward Model 自己也可能错啊。**

完全正确。

这就是 RLHF 的一个巨大问题。

如果：

$$
R_\phi(x,y)
$$

不能准确代表人类真正的偏好，那么 policy 就会：

$$
\boxed{\text{优化 Reward Model，而不是优化真实目标}}
$$

这就是著名的：

$$
\boxed{\text{Reward Hacking / Reward Model Exploitation}}
$$

---

# 13. 举个特别直观的例子

假设人类喜欢：

> 有帮助、准确、简洁。

训练 RM 后，它学到：

> “更长的回答通常更好。”

那么 LLM 就可能发现：

$$
\text{回答越长}  
\rightarrow  
R_\phi\uparrow
$$

于是开始疯狂输出 5000 字。

Reward：

$$
9.8
$$

但人类实际评价：

> “你踏马说这么多干什么……”

这就是：

$$
\boxed{  
\text{Proxy Reward}  
\neq  
\text{True Objective}  
}
$$

---

# 14. 这也是为什么后来的 LLM RL 越来越喜欢 Verifier

尤其是数学、代码、形式推理任务。

因为人类偏好有时候太模糊。

例如数学题：

$$
2+2=?
$$

直接判断：

$$
\text{Answer correct?}
$$

非常容易。

于是可以直接：

$$
R(y)=  
\begin{cases}  
1 & \text{correct}\\
0 & \text{wrong}  
\end{cases}
$$

这时候甚至不需要一个复杂的 Reward Model。

而是：

$$
\boxed{  
\text{LLM}  
\rightarrow  
\text{Verifier}  
\rightarrow  
Reward  
}
$$

这就是 **RL with Verifiable Rewards（RLVR）** 背后的核心思想之一。

---

# 15. RLHF 和 RLVR 的区别

这两个概念你现在应该就能一下看懂了。

### RLHF

$$
\text{LLM}  
\rightarrow  
\boxed{\text{Reward Model}}  
\rightarrow  
R
$$

Reward Model 来自：

$$
\text{Human Preferences}
$$

---

### RLVR

$$
\text{LLM}  
\rightarrow  
\boxed{\text{Verifier}}  
\rightarrow  
R
$$

Verifier 来自：

$$
\text{规则 / 数学答案 / 单元测试 / 程序执行结果}
$$

---

# 16. 为什么 RLVR 对“推理模型”特别重要？

因为数学、代码这类任务存在：

$$
\boxed{\text{可验证性}}
$$

例如：

> 求 (127\times 83)。

模型回答：

$$
10541
$$

程序一算：

$$
127\times83=10541
$$

于是：

$$
R=1
$$

如果回答：

$$
10542
$$

那么：

$$
R=0
$$

这里不需要人类去判断：

> “这个 reasoning 看起来是不是挺聪明？”

机器自己就能验证。

---

# 17. 那 Reward Model 是不是已经没用了？

不是。

两种方法针对的东西不同。

对于：

- 数学
    
- 代码
    
- 棋类
    
- 有明确答案的任务
    

Verifier 非常强。

但对于：

- 回答是否有帮助
    
- 是否自然
    
- 是否符合用户意图
    
- 是否安全
    
- 风格是否合适
    
- 两个都正确的答案哪个更好
    

往往没有一个简单的 verifier。

这时候：

$$
\boxed{\text{Reward Model}}
$$

还是很有用。

---

# 18. 你可以把整个演化过程理解成

最早：

$$
\boxed{  
\text{Human}  
\rightarrow  
\text{直接给Reward}  
}
$$

太贵。

于是：

$$
\boxed{  
\text{Human}  
\rightarrow  
\text{Preference Data}  
\rightarrow  
\text{Reward Model}  
\rightarrow  
\text{Reward}  
}
$$

再往后，对于可验证任务：

$$
\boxed{  
\text{Verifier}  
\rightarrow  
\text{直接Reward}  
}
$$

所以现在理解 LLM RL 时，可以把 reward 来源分成三类：

$$
\boxed{  
\begin{aligned}  
\text{Human} &\rightarrow \text{Preference}\\
\text{RM} &\rightarrow \text{Learned Reward}\\
\text{Verifier} &\rightarrow \text{Verifiable Reward}  
\end{aligned}  
}
$$

---

# 19. 最后把你刚才的问题和上一问完整闭环

你刚才问：

> PPO/GRPO 怎么训练 LLM？

现在可以完整回答：

$$
\boxed{  
\text{LLM}=\text{Policy}  
}
$$

它生成：

$$
y\sim\pi_\theta(y|x)
$$

然后：

$$
\boxed{  
y\rightarrow \text{Reward Source}\rightarrow R  
}
$$

Reward Source 可能是：

$$
\text{RM}
$$

也可能是：

$$
\text{Verifier}
$$

然后：

$$
R  
\rightarrow  
A  
\rightarrow  
\text{PPO/GRPO}  
\rightarrow  
\theta'
$$

所以完整链条其实是：

$$
\boxed{  
\text{Human Preference}  
\rightarrow  
\text{Reward Model}  
\rightarrow  
\text{Reward}  
\rightarrow  
\text{Advantage}  
\rightarrow  
\text{PPO/GRPO}  
\rightarrow  
\text{LLM}  
}
$$

而在 RLVR 中则是：

$$
\boxed{  
\text{Verifier}  
\rightarrow  
\text{Reward}  
\rightarrow  
\text{Advantage}  
\rightarrow  
\text{PPO/GRPO}  
\rightarrow  
\text{LLM}  
}
$$

接下来看[Value Model 与 Reward Model 区别](Value%20Model%20与%20Reward%20Model%20区别.md)