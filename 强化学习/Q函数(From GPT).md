## 1. 整体架构：强化学习到底在学什么？

强化学习的核心问题可以概括为：

> 给定环境，如何通过与环境交互，学习一个能够获得高长期回报的策略 $\pi$。

对于离散动作问题，可以从 **Q 函数**出发理解这一过程。

---

## 2. Q函数

Q函数定义为：

```math
Q^\pi(s,a)=

\mathbb{E}_\pi  
\left[  
\sum_{k=0}^{\infty}  
\gamma^k r_{t+k+1}  
\mid  
s_t=s,a_t=a  
\right]  
```

它表示：

> 在状态 $s$ 下采取动作 $a$，之后按照策略 $\pi$ 行动，能够获得的期望累积回报。

因此：

$$  
Q^\pi(s,a)  
$$

描述的是：

```math
\boxed{  
\text{状态 }s+\text{动作 }a  
\rightarrow  
\text{未来价值}  
}  
```

---

# 3. 从Q函数得到策略

如果已经知道 $Q(s,a)$，就可以根据 Q 值选择动作。

最简单的是 **Greedy Policy**：

```math
\pi(s)=

\arg\max_a Q(s,a)  
```

即：

> 在当前状态下选择 Q 值最大的动作。

因此存在：

```math
\boxed{  
Q\rightarrow\pi  
}  
```

的关系。

但强化学习真正困难的问题是：

> 一开始我们并不知道 $Q$ 函数。

因此问题变成：

```math
\boxed{  
\text{如何通过与环境交互得到的数据学习 }Q？  
}  
```

这就是 DP、MC、TD、SARSA、Q-Learning 等方法要解决的问题。

---

# 4. 学习Q函数的总体框架

学习 Q 函数时，可以从两个维度理解。

## 4.1 是否知道环境模型？

### Model-based

如果知道环境转移模型：

$$  
P(s'|s,a)  
$$

就可以利用模型计算未来状态和价值。

典型方法：

- Dynamic Programming（DP）

### Model-free

如果不知道：

$$  
P(s'|s,a)  
$$

只能通过与环境交互获得：

$$  
(s_t,a_t,r_{t+1},s_{t+1})  
$$

典型方法：

- Monte Carlo（MC）
    
- Temporal Difference（TD）
    
- SARSA
    
- Q-Learning
---

## 4.2 如何估计未来价值？

主要有三种思想：

$$  
\boxed{\text{DP：利用模型进行 Bellman Backup}}  
$$  
$$  
\boxed{\text{MC：利用完整轨迹回报}}  
$$  
$$  
\boxed{\text{TD：利用 Bootstrap}}  
$$


其中需要特别注意：

> **Q-Learning 并不是和 TD 完全并列的方法。**

更准确地说：

```math
\boxed{  
\text{Q-Learning}\subset\text{TD Learning}  
}  
```

同样：

```math
\boxed{  
\text{SARSA}\subset\text{TD Learning}  
}  
```

---

# 5. DP：利用环境模型学习Q

DP（Dynamic Programming）假设我们知道：

$$  
P(s'|s,a)  
$$

以及奖励函数。

对于给定策略 $\pi$，Q函数满足 Bellman expectation equation：

```math
Q^\pi(s,a)=

r(s,a)  
+  
\gamma  
\sum_{s'}  
P(s'|s,a)  
\sum_{a'}  
\pi(a'|s')Q^\pi(s',a')  
```

因此可以不断进行 Bellman Backup：

```math
Q_{k+1}^\pi  
\leftarrow  
\mathcal{T}^\pi Q_k^\pi  
```

最终：

$$  
Q_k^\pi\rightarrow Q^\pi  
$$

### DP的特点

- **Model-based**
    
- 已知环境模型
    
- 不需要实际采样轨迹
    
- 可以遍历状态和动作
    
- 理论上可以精确求解
    
- 状态空间巨大时计算成本很高

DP的核心思想：

```math
\boxed{  
\text{利用已知模型计算未来价值}  
}  
```

---

# 6. MC：利用完整轨迹学习Q

MC（Monte Carlo）不需要知道：

$$  
P(s'|s,a)  
$$

而是直接与环境交互。

得到完整轨迹：

$$  
(s_t,a_t,r_{t+1},s_{t+1},\ldots,s_T)  
$$

然后计算从 $(s_t,a_t)$ 开始的完整回报：

```math
G_t=

r_{t+1}  
+  
\gamma r_{t+2}  
+  
\cdots  
+  
\gamma^{T-t-1}r_T  
```

然后更新：

```math
Q(s_t,a_t)  
\leftarrow  
Q(s_t,a_t)  
+  
\alpha  
\left[  
G_t-Q(s_t,a_t)  
\right]  
```

其中：

$$  
\boxed{G_t}  
$$

是真实轨迹产生的完整回报。

因此 MC：

- **Model-free**
    
- 不需要环境模型
    
- 不使用 Bootstrap(自举)
    
- 通常需要等待 Episode 结束
    
- 完整回报方差较大

MC 的核心思想：

```math
\boxed{  
\text{“这一局真正走完之后，我到底获得了多少？”}  
}  
```

---

# 7. TD：利用Bootstrap(自举)学习Q

TD（Temporal Difference）的核心思想：

> 不需要等整个 Episode 结束，而是利用当前的价值估计来预测未来价值。

一步 TD 的目标可以写成：

```math
\text{Target}=

r+\gamma Q(s',a')  
```

然后：

```math
Q(s,a)  
\leftarrow  
Q(s,a)  
+  
\alpha  
\left[  
r+\gamma Q(s',a')  
-Q(s,a)  
\right]  
```

其中：

```math
\boxed{  
r+\gamma Q(s',a')  
}  
```

称为 **TD Target**。

TD 使用了：

$$  
\boxed{\text{Bootstrap}}  
$$

即：

> 使用当前的价值估计来更新另一个价值估计。

---

# 8. MC与TD的区别

两者最大的区别在于对未来价值的估计方式。

### MC

使用完整真实回报：

```math
\boxed{  
G_t  
}  
```

因此：

```math
Q(s,a)  
\leftarrow  
Q(s,a)  
+  
\alpha  
[  
G_t-Q(s,a)  
]  
```

特点：

- 不 Bootstrap
    
- 必须等待 Episode 结束
    
- 方差较大
    
- 回报目标相对真实

---

### TD

只使用一步经验：

```math
\boxed{  
r+\gamma Q(s',a')  
}  
```

因此：

```math
Q(s,a)  
\leftarrow  
Q(s,a)  
+  
\alpha  
[  
r+\gamma Q(s',a')-Q(s,a)  
]  
```

特点：

- Bootstrap
    
- 不需要等待 Episode 结束
    
- 可以一步一步更新
    
- 方差通常更低，但存在 Bootstrap bias

因此可以记为：

```math
\boxed{  
\text{MC：真实回报}  
}  
```

```math
\boxed{  
\text{TD：奖励 + 当前价值估计}  
}  
```

---

# 9. 从价值估计到策略控制

前面的讨论主要解决：

> 给定策略 $\pi$，如何学习它的 Q 函数？

也就是：

```math
\boxed{  
\pi\rightarrow Q^\pi  
}  
```

这叫：

### Policy Evaluation

但是强化学习最终并不只是想评价一个策略，而是希望：

> 不断改进策略，使其越来越好。

因此还需要：

### Policy Improvement

给定 Q 函数，可以构造新的策略：

```math
\pi_{\text{new}}(s)=

\arg\max_a Q^\pi(s,a)  
```

即：

> 根据当前 Q 函数，在每个状态选择价值最高的动作。

于是形成：

```math
\boxed{  
\text{Policy Evaluation}  
\rightarrow  
\text{Policy Improvement}  
\rightarrow  
\text{Policy Evaluation}  
\rightarrow\cdots  
}  
```

这就是强化学习中非常重要的 **Policy Iteration** 思想。

---

# 10. SARSA：On-policy TD Control

SARSA 可以理解为：

```math
\boxed{  
\text{SARSA}

\text{TD}  
+  
\text{Q函数}  
+  
\text{On-policy Control}  
}  
```

SARSA 使用：

$$  
(S_t,A_t,R_{t+1},S_{t+1},A_{t+1})  
$$

因此名字来自：

```math
\boxed{  
S-A-R-S-A  
}  
```

更新公式：

```math
Q(s_t,a_t)  
\leftarrow  
Q(s_t,a_t)  
+  
\alpha  
\left[  
r_{t+1}  
+  
\gamma Q(s_{t+1},a_{t+1})
-
Q(s_t,a_t)  
\right]  
```

关键在于：

$$  
\boxed{a_{t+1}}  
$$

它是**行为策略实际选择出来的动作**。

如果行为策略为：

$$  
\pi=\epsilon\text{-greedy}(Q)  
$$

那么：

```math
a_{t+1}  
\sim  
\pi(\cdot|s_{t+1})  
```

因此 SARSA 学习的是当前行为策略：

```math
\boxed{  
Q^{\pi_\epsilon}  
}  
```

所以：

```math
\boxed{  
\text{SARSA是On-policy}  
}  
```

---

# 11. Q-Learning：Off-policy TD Control

Q-Learning 同样是 TD Control 方法。

它的更新公式：

```math
Q(s_t,a_t)  
\leftarrow  
Q(s_t,a_t)  
+  
\alpha  
\left[  
r_{t+1}  
+  
\gamma  
\max_{a'}  
Q(s_{t+1},a')
-
Q(s_t,a_t)  
\right]  
```

与 SARSA 最大的区别：

```math
\boxed{  
Q(s',a')  
\quad\rightarrow\quad  
\max_{a'}Q(s',a')  
}  
```

SARSA 使用：

$$  
a_{t+1}  
$$

即行为策略实际选择的动作。

而 Q-Learning 使用：

$$  
\arg\max_{a'}Q(s',a')  
$$

即当前 Q 函数认为最好的动作。

---

# 12. 为什么Q-Learning是Off-policy？

假设实际交互使用：

$$  
\mu=\epsilon\text{-greedy}(Q)  
$$

那么实际采取的动作：

$$  
a_t\sim\mu(\cdot|s_t)  
$$

但是 Q-Learning 更新时使用：

$$  
\max_{a'}Q(s',a')  
$$

相当于目标策略为：

```math
\pi(s')
=
\arg\max_{a'}Q(s',a')  
```

因此：

```math
\boxed{  
\mu\neq\pi  
}  
```

所以：

```math
\boxed{  
\text{Q-Learning是Off-policy}  
}  
```

---

# 13. Q-Learning中的Greedy

Q-Learning 经常被称为使用 **Greedy** 的方法，但需要区分：

## 13.1 行为策略

为了探索，通常使用：

$$  
\mu=\epsilon\text{-greedy}(Q)  
$$

因此实际行动不一定是 Greedy。

---

## 13.2 目标策略

Q-Learning 使用：

```math
\pi(s)
=
\arg\max_aQ(s,a)  
```

因此目标策略是 Greedy。

所以：

```math
\boxed{  
\text{Q-Learning不是“行为完全Greedy”}  
}  
```

而是：

```math
\boxed{  
\text{行为策略可以探索，TD Backup使用Greedy选择}  
}  
```

这也是它成为 Off-policy 方法的关键。

---

# 14. SARSA与Q-Learning的核心区别
|            | SARSA                | Q-Learning                  |
| ---------- | -------------------- | --------------------------- |
| 类型         | TD Control           | TD Control                  |
| Model-free | ✅                    | ✅                           |
| Policy     | On-policy            | Off-policy                  |
| 学习对象       | $Q^\pi$              | $`Q^*`$                       |
| TD Target  | $r+\gamma Q(s',a')$  | $r+\gamma\max_{a'}Q(s',a')$ |
| 下一动作       | 实际采取的 $a'$           | Q值最大的动作                     |
| 行为策略       | 通常 $\epsilon$ -greedy | 通常 $\epsilon$ -greedy        |
| 目标策略       | 行为策略本身               | Greedy Policy               |
| 是否使用实际探索动作 | ✅                    | ❌                           |

可以用一句话理解：

```math
\boxed{  
\text{SARSA：按照“我实际怎么走”来评价}  
}  
```

```math
\boxed{  
\text{Q-Learning：不管我怎么探索，按照“最优应该怎么走”来评价}  
}  
```

---

# 15. Q-Learning最终学习什么？

Q-Learning 的目标是学习最优 Q 函数：
```math
Q^*(s,a)

\max_\pi Q^\pi(s,a)  
```

它满足 Bellman Optimality Equation：

```math
Q^*(s,a)
=
\mathbb{E}  
\left[  
r+\gamma\max_{a'}Q^*(s',a')  
\right]  
```

得到 $`Q^*`$ 后，最优策略可以直接通过 Greedy 得到：

```math
\boxed{  
\pi^*(s)
=
\arg\max_a Q^*(s,a)  
}  
```

因此：

```math
\boxed{  
Q^*  
\rightarrow  
\pi^*  
}  
```

---

# 16. 完整知识架构

整个知识体系可以整理为：

```text
强化学习
│
├── 目标：学习一个好的策略 π
│
├── 价值函数
│   ├── Vπ(s)
│   └── Qπ(s,a)
│
└── 从 Q 函数得到策略
    │
    └── Greedy：
        π(s) = argmax_a Q(s,a)

            ↓

       如何学习 Q？

            │
     ┌──────┴──────┐
     │             │
 Model-based    Model-free
     │             │
     DP        ┌───┴───┐
               │       │
              MC       TD
                       │
                ┌──────┴──────┐
                │             │
             SARSA       Q-Learning
                │             │
           On-policy      Off-policy
                │             │
             Q^π             Q*
```

---

# 17. 三个最重要的分类问题

以后判断一个 RL 算法应该问三个问题。

## 问题一：学习什么？

```math
\boxed{  
Q^\pi  
\quad\text{还是}\quad  
Q^*  
}  
```

如果是：

$$  
Q^\pi  
$$

通常是在做 **Policy Evaluation**。

如果是：

```math
Q^*  
```

通常是在做 **Control / Policy Optimization**。

---

## 问题二：Target怎么得到？

### MC

```math
\boxed{  
G_t  
}  
```

完整回报。

### TD

```math
\boxed{  
r+\gamma Q(s',a')  
}  
```

Bootstrap。

### Q-Learning

```math
\boxed{  
r+\gamma\max_{a'}Q(s',a')  
}  
```

Greedy / Optimality Backup。

---

## 问题三：谁产生数据，谁是目标策略？

行为策略：

$$  
\boxed{\mu}  
$$

目标策略：

$$  
\boxed{\pi}  
$$

如果：

$$  
\mu=\pi  
$$

那么：

$$  
\boxed{\text{On-policy}}  
$$

如果：

$$  
\mu\neq\pi  
$$

那么：

$$  
\boxed{\text{Off-policy}}  
$$

---

# 18. 最终统一理解

整个 Q-Learning 的逻辑可以压缩为：

```math
\boxed{  
\text{环境交互}  
\rightarrow  
\text{经验数据}  
\rightarrow  
Q(s,a)  
\rightarrow  
\max_a Q(s,a)  
\rightarrow  
\pi  
}  
```

其中：

### DP

```math
\boxed{  
\text{模型}  
\rightarrow  
\text{Bellman Backup}  
\rightarrow  
Q^\pi  
}  
```

### MC

```math
\boxed{  
\text{完整轨迹}  
\rightarrow  
G_t  
\rightarrow  
Q^\pi  
}  
```

### TD / SARSA

```math
\boxed{  
(s,a,r,s',a')  
\rightarrow  
r+\gamma Q(s',a')  
\rightarrow  
Q^\pi  
}  
```

### Q-Learning

```math
\boxed{  
(s,a,r,s')  
\rightarrow  
r+\gamma\max_{a'}Q(s',a')  
\rightarrow  
Q^*  
\rightarrow  
\pi^*  
}  
```

---

# 19. 与DQN的连接

传统 Q-Learning 使用 Q-table：

$$  
Q(s,a)  
$$

当状态空间非常大时，无法直接存储所有状态-动作对。

于是使用神经网络近似 Q：

$$  
Q_\theta(s,a)  
$$

于是：

```math
\boxed{  
\text{Q-Learning}  
+  
\text{Neural Network}  
\approx  
\text{DQN}  
}  
```

DQN 的核心目标仍然来自 Q-Learning：

```math
y
=
r+  
\gamma\max_{a'}Q_\theta(s',a')  
```

只是：

$$  
Q(s,a)  
$$

变成了：

$$  
Q_\theta(s,a)  
$$

因此，DQN 并不是改变了 Q-Learning 的基本思想，而是：

> **用深度神经网络代替 Q-table，学习最优 Q 函数。**

---

# 20. 总结

```math
\boxed{  
\text{先定义Q函数}  
\rightarrow  
\text{再研究如何估计Q}  
\rightarrow  
\text{再研究如何利用Q改进策略}  
}  
```

其中：

```math
\boxed{  
\text{DP：有模型}  
}  
```

```math
\boxed{  
\text{MC：完整回报}  
}  
```

```math
\boxed{  
\text{TD：Bootstrap}  
}  
```

```math
\boxed{  
\text{SARSA：On-policy TD Control}  
}  
```

```math
\boxed{  
\text{Q-Learning：Off-policy TD Control + Greedy Backup}  
}  
```

最终：

```math
\boxed{
Q^*
\rightarrow
\pi^*(s)=\arg\max_a Q^*(s,a)
}
```