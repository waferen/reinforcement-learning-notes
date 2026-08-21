# Double DQN与Dueling DQN区别

我们可以通过比较 Dueling DQN 和 Double DQN 来引入。它们其实是**两个完全不同维度的改进**。这个地方特别容易混，因为 Dueling DQN 和 Double DQN 都属于 “DQN Tricks”，但它们解决的问题不一样。

最核心的一句话：

$$
\boxed{\text{Dueling DQN：改网络结构}}
$$

$$
\boxed{\text{Double DQN：改 Q 值目标的计算方式}}
$$

---

## 1. 先看普通 DQN 在干什么

普通 DQN 有一个网络：

$$
Q_\theta(s,a)
$$

输入状态 (s)，输出所有动作的 Q 值：

$$
[Q(s,a_1),Q(s,a_2),\cdots,Q(s,a_n)]
$$

训练目标通常是：

$$
y=r+\gamma\max_{a'}Q_{\theta^-}(s',a')
$$

然后最小化：

$$
L=(y-Q_\theta(s,a))^2
$$

这里实际上存在两个问题：

### 问题 A：Q 网络的结构不够合理

网络直接：

$$
s\rightarrow Q(s,a_1),\cdots,Q(s,a_n)
$$

没有显式区分：

- 当前状态本身有多好
    
- 不同动作之间有什么区别
    

这就是 **Dueling DQN** 要解决的问题。

---

### 问题 B： $\max$ 容易导致 Q 值高估

看这个：

$$
\max_{a'}Q_{\theta^-}(s',a')
$$

假设真实 Q 值都是：

$$
[10,10,10,10]
$$

但是网络估计有误差：

$$
[9.8,10.3,9.7,10.1]
$$

那么：

$$
\max Q=10.3
$$

你本来真实值是 10，却因为估计误差选到了一个偏高的值。

这就是 **Overestimation Bias**。

Double DQN 就是解决这个问题的。

---

# 2. Dueling DQN：解决“怎么表示 Q”

Dueling 把原来的：

$$
Q(s,a)
$$

拆成：

$$
Q(s,a)=V(s)+A(s,a)
$$

网络结构变成：

```text
                 State s
                    │
              Shared Network
                    │
             ┌──────┴──────┐
             ↓             ↓
          V Stream      A Stream
             │             │
           V(s)      A(s,a1),...,A(s,an)
             └──────┬──────┘
                    ↓
              Combine
                    ↓
                 Q(s,a)
```

所以 Dueling 关注的是：

> **Q 函数应该如何参数化、如何表示。**

它改善的是网络的**表示能力和学习效率**。

---

# 3. Double DQN：解决“怎么计算 target”

Double DQN 不需要把网络拆成 $V$ 和 $A$。

它甚至可以直接使用普通 DQN 的网络：

$$
Q_\theta(s,a)
$$

变化的是 target。

普通 DQN：

```math
\boxed{  
y=r+\gamma\max_{a'}Q_{\theta^-}(s',a')  
}
```

这里：

> **选动作**和**评价动作**都是同一个网络。

Double DQN 把它拆成两步：

### 第一步：Online Network 负责选动作

```math
a^*
=
\arg\max_{a'}Q_\theta(s',a')
```

### 第二步：Target Network 负责评价这个动作

```math
Q_{\theta^-}(s',a^*)
```

所以 target 变成：

```math
\boxed{  
y=  
r+\gamma Q_{\theta^-}  
\left(  
s',  
\arg\max_{a'}Q_\theta(s',a')  
\right)  
}
```

这就是 Double DQN。

---

# 4. 为什么叫 Double？

因为它把原来：

$$
\max_a Q(s,a)
$$

这一件事情拆成了：

```math
\boxed{  
\text{Action Selection}  
+  
\text{Action Evaluation}  
}
```

即：

```math
\underbrace{  
\arg\max_aQ_\theta(s,a)  
}_{\text{选择}}
```

和：

```math
\underbrace{  
Q_{\theta^-}(s,a)  
}_{\text{评价}}
```

分别交给两个网络。

所以叫：

> **Double Q-learning**

它并不是说“用了两个 DQN 网络就叫 Double DQN”。

更准确地说，是：

> **让两个相对独立的估计过程分别承担 action selection 和 action evaluation。**

---

# 5. 两者可以同时使用吗？

不仅可以，而且**完全可以同时使用**。

这是非常重要的一点。

因为：

- Dueling 改的是 **网络结构**
    
- Double 改的是 **TD target**
    

所以二者并不冲突。

可以得到：

$$
\boxed{\text{Dueling Double DQN}}
$$

网络本身：

```math
Q_\theta(s,a)

V_\theta(s)+A_\theta(s,a)-\text{normalization}
```

然后 target 使用 Double DQN：

```math
y

r+\gamma  
Q_{\theta^-}  
\left(  
s',  
\arg\max_{a'}Q_\theta(s',a')  
\right)
```

于是：

> **Dueling 决定“Q 怎么表示”，Double 决定“Q 怎么更新”。**

---

# 6. 你可以用一个表把它们彻底分开

|                     | Dueling DQN | Double DQN              |
| ------------------- | ----------- | ----------------------- |
| 解决什么                | Q 表示效率      | Q 值高估                   |
| 修改哪里                | 网络结构        | TD Target               |
| 核心思想                | $Q=V+A$     | Selection/Evaluation 分离 |
| 是否需要双网络             | 不需要额外增加     | 利用 Online + Target      |
| 是否改变 Q 输出结构         | 是           | 不一定                     |
| 是否改变 Bellman target | 否           | 是                       |
| 能否和另一个结合            | 可以          | 可以                      |

---

# 7. 一个很重要的误区

你现在看到 Dueling 里面也有： $V(s)$  

然后 Double DQN 又有：

$$
Q_\theta,\quad Q_{\theta^-}
$$

很容易产生一种错觉：

> “是不是 Dueling 的 $V$ 和 $A$ 就对应 Double DQN 的两个网络？”

**完全不是。**

它们的“两个东西”根本不是一个维度。

### Dueling 的两个分支

$V(s)$ 和 $A(s,a)$

表示的是：

> **同一个 Q 函数的两种组成部分。**

---

### Double DQN 的两个网络

$$
Q_\theta
$$

和

$$
Q_{\theta^-}
$$

表示的是：

> **两个不同参数状态下的 Q 函数估计器。**

所以：

```math
\boxed{  
V/A\neq Online/Target  
}
```

这是理解 DQN 系列时非常关键的一点。

---

# 8. 把整个 DQN 体系放到一起，你就会很清楚

你前面笔记里的几个 trick，其实分别在攻击不同问题：

```text
                         DQN
                          │
          ┌───────────────┼────────────────┐
          ↓               ↓                ↓
     Multi-step TD   Experience Replay   Dueling
          │               │                │
      降低 TD 偏差      打破样本相关性      改善表示
                                          Q = V + A

                          │
                          ↓
                     Double DQN
                          │
                    减少 Q 高估
                          │
             Selection ≠ Evaluation
```

甚至可以进一步整理成：

```math
\boxed{  
\begin{array}{c|c}  
\text{方法}&\text{主要解决的问题}\\
\hline  
\text{Experience Replay}&\text{样本相关性、数据利用率}\\
\text{Target Network}&\text{训练目标不断变化}\\
\text{Multi-step TD}&\text{TD 的偏差}\\
\text{Prioritized ER}&\text{哪些经验更值得学习}\\
\text{Dueling}&\text{Q函数表示效率}\\
\text{Double DQN}&\text{Q值过估计}  
\end{array}  
}
```

所以从学习 DQN 的角度，你现在可以把两者记成：

> **Dueling：把 Q“拆开学”。**

> **Double：把 Q“分开选、分开评”。**


# Double DQN 完整训练流程

先给你一张总图：

```text
                    初始化
                      │
          ┌───────────┴───────────┐
          ↓                       ↓
     Online Network          Target Network
        Qθ(s,a)                 Qθ⁻(s,a)
          │                       │
          └───────────┬───────────┘
                      ↓
                  与环境交互
                      │
              得到 (s,a,r,s')
                      │
                      ↓
                Replay Buffer
                      │
                      ↓
                 随机采样 Batch
                      │
                      ↓
              ┌───────┴────────┐
              │                │
              ↓                ↓
        Qθ 选择动作        Qθ⁻ 评价动作
              │                │
              ↓                ↓
   a* = argmax Qθ(s',a)    Qθ⁻(s',a*)
              └───────┬────────┘
                      ↓
                 TD Target
                      │
                      ↓
              更新 Online Network
                      │
                      ↓
                定期同步参数
                  θ⁻ ← θ
                      │
                      └──────→ 重复
```

下面一步一步看。

---

## 1. 初始化两个网络

首先建立两个结构完全一样的 Q 网络：

$$
Q_\theta(s,a)
$$

和

$$
Q_{\theta^-}(s,a)
$$

其中：

$$
\boxed{Q_\theta=\text{Online Network}}
$$

$$
\boxed{Q_{\theta^-}=\text{Target Network}}
$$

初始化时让它们参数相同：

$$
\boxed{\theta^- \leftarrow \theta}
$$

例如：

```text
Online:
θ  = [0.2, 0.5, 0.8, ...]

Target:
θ⁻ = [0.2, 0.5, 0.8, ...]
```

同时建立 Replay Buffer：

$$
D = ReplayBuffer()
$$

并设置：

- 学习率 $\alpha$
    
- 折扣因子 $\gamma$
    
- batch size
    
- target network 更新周期
    
- $\epsilon$ -greedy 的探索率
    

---

## 2. 当前处于状态 $s_t$

智能体从环境获得：

$$
s_t
$$

例如：

```text
游戏当前画面
      ↓
   状态 s_t
```

然后把状态送入 Online Network：

$$
Q_\theta(s_t,a)
$$

假设有 4 个动作：

```math
Q_\theta(s_t,\cdot)
=
[1.2,2.5,1.8,0.9]
```

对应：

```text
a1 → 1.2
a2 → 2.5
a3 → 1.8
a4 → 0.9
```

---

## 3. 用 $\epsilon$ -greedy 选择动作

Double DQN 通常仍然使用：

$$
\epsilon\text{-greedy}
$$

即：

以概率 ($1-\epsilon$)：

$$
a_t=\arg\max_aQ_\theta(s_t,a)
$$

以概率 ($\epsilon$)：

$$
a_t\sim\text{Random}
$$

例如：

$$
\epsilon=0.1
$$

那么大部分情况下：

$$
a_t=a_2
$$

因为：

$$
Q_\theta(s_t,a_2)=2.5
$$

---

## 4. 执行动作，与环境交互

执行：

$$
a_t
$$

环境返回：

$$
(s_{t+1},r_{t+1},done)
$$

于是得到一条经验：

```math
\boxed{  
(s_t,a_t,r_{t+1},s_{t+1})  
}
```

例如：

```text
当前状态：
s_t

执行：
a2

得到：
reward = +1

下一状态：
s_{t+1}
```

---

## 5. 把经验存入 Replay Buffer

把：

$$
(s_t,a_t,r_{t+1},s_{t+1})
$$

存入：

$$
D
$$

例如：

```text
Replay Buffer

------------------------------------------------
(s1, a2, +1, s2)
(s2, a1,  0, s3)
(s3, a3, -1, s4)
...
------------------------------------------------
```

这样旧经验之后还可以重新使用。

---

## 6. 随机采样一个 mini-batch

假设 batch size：

$$
B=32
$$

从 Replay Buffer 中随机采样：

$$
{(s_i,a_i,r_{i+1},s_{i+1})}_{i=1}^{32}
$$

这一步的意义主要是：

> 打破连续时间步之间的相关性。

---

## 7. 计算当前 Q 值

对于 batch 中每条经验，我们首先用 **Online Network** 计算实际执行动作的 Q：

```math
\boxed{  
Q_\theta(s_i,a_i)  
}
```

注意：

这里是当前状态 (s_i)，以及当时真正执行的动作 (a_i)。

---

## 8. Double DQN 最关键的一步：选择下一动作

现在来到下一状态：

$$
s_{i+1}
$$

先让 **Online Network** 来选择：

```math
\boxed{  
a_i^*
=
\arg\max_{a'}  
Q_\theta(s_{i+1},a')  
}
```

注意：

### 这里只负责“选择”

例如：

```math
Q_\theta(s_{i+1},\cdot)
=
[2.1,3.7,3.0,2.8]
```

那么：

```math
a_i^*=a_2
```

因为：

$$
3.7=\max_aQ_\theta(s_{i+1},a)
$$

---

## 9. 但不能直接拿 3.7 当 target

这就是 Double DQN 和普通 DQN 的关键区别。

普通 DQN 会：

$$
\max_{a'}Q_{\theta^-}(s_{i+1},a')
$$

直接取 Target Network 的最大值。

而 Double DQN：

> Online Network 负责选动作。

所以现在已经知道：

```math
a_i^*=a_2
```

接下来交给 Target Network：

```math
\boxed{  
Q_{\theta^-}(s_{i+1},a_i^*)  
}
```

也就是：

> “你觉得 $a_2$ 最好，那我让另一个网络来评价 $a_2$ 到底值多少。”

---

## 10. 计算 Double DQN 的 TD Target

于是：

```math
\boxed{  
y_i
=
r_{i+1}  
+  
\gamma  
Q_{\theta^-}  
\left(  
s_{i+1},  
\arg\max_{a'}  
Q_\theta(s_{i+1},a')  
\right)  
}
```

如果下一状态已经是终止状态，那么没有未来回报：

```math
\boxed{  
y_i=r_{i+1}  
}
```

通常写成：

```math
y_i
=
r_{i+1}  
+  
\gamma(1-done_i)  
Q_{\theta^-}  
\left(  
s_{i+1},  
\arg\max_{a'}  
Q_\theta(s_{i+1},a')  
\right)
```

---

## 11. 举一个具体数字例子

假设：

$$
r=1
$$

$$
\gamma=0.99
$$

Online Network 给出：

```math
Q_\theta(s',\cdot)
=
[4.1,5.2,4.8]
```

所以：

```math
a^*=\arg\max_aQ_\theta(s',a)=a_2
```

然后 Target Network 给出：

```math
Q_{\theta^-}(s',\cdot)
=
[4.5,4.7,5.0]
```

注意这里：

```math
a^*=a_2
```

所以我们只取：

$$
Q_{\theta^-}(s',a_2)=4.7
$$

最终：

```math
y
=
1+0.99\times4.7
```

得到：

$$
\boxed{y=5.653}
$$

---

## 12. 然后更新 Online Network

我们当前网络对于实际动作的预测是：

$$
Q_\theta(s,a)
$$

假设：

$$
Q_\theta(s,a)=4.8
$$

而 target：

$$
y=5.653
$$

那么 TD error：

```math
\boxed{  
\delta=y-Q_\theta(s,a)  
}
```

即：

$$
\delta=5.653-4.8=0.853
$$

损失函数可以写成：

```math
\boxed{  
L(\theta)
=
\left(  
y-Q_\theta(s,a)  
\right)^2  
}
```

实际实现里通常使用 Huber Loss，而不是简单平方误差。

然后反向传播：

$$
\nabla_\theta L
$$

更新：

```math
\boxed{  
\theta  
\leftarrow  
\theta-\alpha\nabla_\theta L  
}
```

注意：

$$
\boxed{\theta^- \text{ 不更新}}
$$

这一轮只更新 Online Network。

---

## 13. Target Network 到底什么时候更新？

这是 DQN 的另一个关键机制。

经过若干次 Online Network 更新后：

$$
\theta
$$

已经和：

$$
\theta^-
$$

不一样了。

例如：

```text
刚开始：

θ  = [0.2,0.5,0.8]
θ⁻ = [0.2,0.5,0.8]


训练很多次：

θ  = [0.8,0.3,1.2]
θ⁻ = [0.2,0.5,0.8]
```

然后每隔 $C$ 次训练：

```math
\boxed{  
\theta^-\leftarrow\theta  
}
```

例如：

$$
C=1000
$$

意味着每 1000 次 gradient update，把 Online Network 完整复制给 Target Network。

---

## 14. 然后继续训练

参数同步后：

$$
\theta^-=\theta
$$

然后继续：

```text
环境交互
   ↓
Replay Buffer
   ↓
采样 Batch
   ↓
Online Network 选动作
   ↓
Target Network 评价动作
   ↓
计算 TD Target
   ↓
更新 Online Network
   ↓
每隔 C 次：
θ⁻ ← θ
   ↓
继续
```

整个训练过程不断循环。

---

## 15. 所以完整算法可以浓缩成 8 步

你可以直接记成：

### Step 1：初始化

$$
Q_\theta,\quad Q_{\theta^-}
$$

并：

$$
\theta^-=\theta
$$

---

### Step 2：获取状态

$$
s_t
$$

---

### Step 3：Online Network 选择动作

```math
a_t
=
\epsilon\text{-greedy}(Q_\theta(s_t,\cdot))
```

---

### Step 4：与环境交互

```math
(s_t,a_t)  
\rightarrow  
(r_{t+1},s_{t+1})
```

---

### Step 5：存入 Replay Buffer

$$
(s_t,a_t,r_{t+1},s_{t+1},done)
$$

---

### Step 6：采样 Batch，并计算 Double DQN Target

先选：

```math
\boxed{  
a^*
=
\arg\max_{a'}  
Q_\theta(s',a')  
}
```

再评：

```math
\boxed{  
y
=
r+\gamma Q_{\theta^-}(s',a^*)  
}
```

合在一起：

```math
\boxed{  
y
=
r+  
\gamma  
Q_{\theta^-}  
\left(  
s',  
\arg\max_{a'}Q_\theta(s',a')  
\right)  
}
```

---

### Step 7：更新 Online Network

```math
L=  
\left(  
y-Q_\theta(s,a)  
\right)^2
```

然后：

$$
\theta\leftarrow\theta-\alpha\nabla_\theta L
$$

---

### Step 8：定期更新 Target Network

```math
\boxed{  
\theta^-\leftarrow\theta  
}
```

然后回到 Step 2。

---
