## 参考课程

https://www.youtube.com/watch?v=o_g9JUMw1Oc
https://www.bilibili.com/video/BV1hhbSzjEi1?spm_id_from=333.788.videopod.episodes&vd_source=b17d4de2c32b04e437ad7699ea8a76ea&p=10

>可看完[RL Lecture](RL%20Lecture.md)再来看 DQN。

## Q 函数简介

>在学习完普通的基于 state 的 Value Function 后，我们可以学习另一种 Critic——基于 state-action 的 Value Function。注意二者的区别：
> $V^\pi(s)$：**状态价值函数**（只看状态 s）→ 在策略 $\pi$ 下，从状态s出发**后续所有动作都按 $\pi$ 执行**，能拿到的期望累计回报
> $Q^\pi(s,a)$：**状态 - 动作价值函数 / Q 函数**（同时看$`(s,a)`$）→ 在策略 $(\pi)$ 下，**先在s刻意执行动作a，之后再全部按 $(\pi)$ 执行**，能拿到的期望累计回报

Q函数定义为：

```math
Q^\pi(s,a)
=
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

关于 Q 函数具体可查看[Q函数(From GPT)](Q函数(From%20GPT).md)，里面具体介绍了 Q-Learning 的数学框架，还介绍了其与 Sarsa 的区别。

![Pasted image 20260812164814](附件/Pasted%20image%2020260812164814.png)
## Q-Learning在强化学习里的定位

>那么Q-Learning 即是要学习这样一种 Q 函数，当然学习 Q 函数的方式有很多，Q-Learning 只是 TD 算法的其中一种。它在强化学习的定位大概如下图。
```
                     Critic
                        │
             ┌──────────┴──────────┐
             ↓                     ↓
          V(s)                  Q(s,a)
             │                     │
             │              如何学习 Q？
             │                     │
             │          ┌──────────┼──────────┐
             │          ↓          ↓          ↓
             │         DP          MC         TD
             │                                │
             │                         ┌──────┴──────┐
             │                         ↓             ↓
             │                      SARSA       Q-Learning
             │                       On-policy    Off-policy
             │
```
## Q Learning算法

>Q Learning算法如下：

![Pasted image 20260817125153](附件/Pasted%20image%2020260817125153.png)

## DQN简介

>进一步 DQN 即是在 Q-Learning 的基础上将 Q函数变成了 Network。

![Pasted image 20260819140939](附件/Pasted%20image%2020260819140939.png)
>具体的训练目标（即 TD error）定义如下：

![Pasted image 20260819143314](附件/Pasted%20image%2020260819143314.png)
![Pasted image 20260819143525](附件/Pasted%20image%2020260819143525.png)


>当然只是这样单纯的去训练 DQN 效果并不好，我们需要一些 trick。接下来介绍一些 trick。

## Muti-step TD

>我们都知道 MC 是高方差低偏差，TD 是低方差高偏差。那么如何降低 TD 的偏差呢？一个方式是通过 Muti-step 来代替 One-step。

![Pasted image 20260819155819](附件/Pasted%20image%2020260819155819.png)

>要达到这个效果其实只要在原本的 Ut 基础上多展开m 项。


![Pasted image 20260819155849](附件/Pasted%20image%2020260819155849.png)
![Pasted image 20260819160016](附件/Pasted%20image%2020260819160016.png)
![Pasted image 20260819160056](附件/Pasted%20image%2020260819160056.png)

## Experience Replay

>ER在 RL 上提出的 Motivation 主要是由于经验的浪费。

![Pasted image 20260819144332](附件/Pasted%20image%2020260819144332.png)
![Pasted image 20260819144350](附件/Pasted%20image%2020260819144350.png)
>ER 本质上就是维护一个 Replay Buffer ，通过这个 buffer，对历史数据进行重复训练。这里的 n 是 buffer 大小，是超参需要调，在 ER 中，在 n 较小时，一般是n 越大效果越好。

![Pasted image 20260819144531](附件/Pasted%20image%2020260819144531.png)
>由于可以用 buffer 中的旧数据，即可以形成 batch 进行训练，可以使用 mini-batch-SGD，比直接 GD 会更快收敛。

![Pasted image 20260819144828](附件/Pasted%20image%2020260819144828.png)
>当然 ER 的方式有很多，它是持续学习的基础。本人也复现并改进过一个 CVPR2025 提出的 ER 方法，可以参考库 https://github.com/waferen/ViewBatchModel-MyThesis

![Pasted image 20260819145148](附件/Pasted%20image%2020260819145148.png)
>这里我们可以对一种 ER 方法 Prioritized ER，进行介绍。即对于更少见，更重要的经验，提高其在 buffer 中的优先级。

![Pasted image 20260819145420](附件/Pasted%20image%2020260819145420.png)
>那么基于什么评判什么是更重要的？基于TD error。

![Pasted image 20260819145620](附件/Pasted%20image%2020260819145620.png)
>由于对优先级进行了排序，高优先级的容易多次训练，那么对应要降低高优先级样本的学习率。

![Pasted image 20260819145757](附件/Pasted%20image%2020260819145757.png)
## Dueling Network

>这是第二个 trick，我们先回顾下几种价值函数。

![Pasted image 20260819151446](附件/Pasted%20image%2020260819151446.png)
![Pasted image 20260819151704](附件/Pasted%20image%2020260819151704.png)
>Dueling Network 的本质就是将DQN 原本只学习 Q 改成对 A 和 V 学习两个网络。
>接下来我们进行数学推导。先观察以下结论。

![Pasted image 20260819160438](附件/Pasted%20image%2020260819160438.png)
证明如下：
```math
\boxed{V^*(s)=\max_a Q^*(s,a)}
```

直觉上就是：

> **在状态 $s$ 下，最优状态价值 ($`V^*(s)`$)，就是选择一个最优动作之后所能得到的最大动作价值 ($`Q^*(s,a)`$)。**

下面从定义一步一步证明。

```math
V^*(s)=\max_{\pi}V^\pi(s)
```

意思是：

> 从状态 $s$ 出发，我们可以选择任何策略，能够得到的最大期望回报。

---
```math
Q^*(s,a)=\max_{\pi}Q^\pi(s,a)
```

这里的含义稍微不同：

> **已经决定当前状态 $s$ 下先执行动作 $a$ **，之后再采用最优策略，能够获得的最大期望回报。

所以：

* ($`V^*(s)`$)：**我现在还没决定做什么，看看最优情况下能拿多少。**
* ($`Q^*(s,a)`$)：**我现在强制做 (a)，之后再最优，能拿多少。**

---

在状态 $s$ 下，如果我们要获得最优回报，那么第一步一定要：

1. 选择一个动作 $a$
2. 执行动作 $a$
3. 后面继续采取最优策略

而“选择动作 (a)，然后后面采取最优策略”对应的价值恰好就是：

```math
Q^*(s,a)
```

因此，我们只需要在所有可能动作中选择最好的那个：

```math
V^*(s)=\max_a Q^*(s,a)
```

这就是这个定理。

不过这个解释比较直观，我们可以再进行一个更严格的数学证明。

---

我们从

```math
V^*(s)=\max_\pi V^\pi(s)
```

开始。

对于任意策略 ($\pi$)，在状态 $s$ 下，策略首先会选择一个动作。

把策略 $\pi$ 在状态 $s$ 下选择动作的概率记作

$$
\pi(a|s)
$$

那么根据 $V^\pi$ 和 $Q^\pi$ 的关系：

```math
V^\pi(s)
=

\sum_a \pi(a|s)Q^\pi(s,a)
```

因为 $V^\pi(s)$ 就是按照策略 $\pi$ 对所有可能动作的 $Q^\pi(s,a)$ 做加权平均。

---

令最优策略为 ($`\pi^*`$)，那么：

```math
V^*(s)=V^{\pi^*}(s)
```

于是：

```math
V^*(s)
=

\sum_a\pi^*(a|s)Q^{\pi^*}(s,a)
```

又因为最优策略之后都是最优的，所以：

```math
Q^{\pi^*}(s,a)=Q^*(s,a)
```

因此：

```math
V^*(s)
=

\sum_a\pi^*(a|s)Q^*(s,a)
```

注意这里有一个非常关键的事实：

```math
\sum_a\pi^*(a|s)Q^*(s,a)
```

是所有 $`Q^*(s,a)`$ 的**加权平均**。

一个数的加权平均值不可能超过这些数中的最大值，因此：

```math
V^*(s)\leq \max_a Q^*(s,a)
```

---

再证明反方向

设

```math
a^*=\arg\max_a Q^*(s,a)
```

也就是说 $`a^*`$ 是状态 $s$ 下最优的动作。

如果我们在状态 $s$ 直接选择这个动作 (a^*)，然后从下一步开始一直采取最优策略，那么得到的价值就是：

```math
Q^*(s,a^*)
```

因此：

```math
V^*(s)\geq Q^*(s,a^*)
```

而根据 $`a^*`$ 的定义：

```math
Q^*(s,a^*)=\max_aQ^*(s,a)
```

所以：

```math
V^*(s)\geq \max_aQ^*(s,a)
```

结合前面的：

```math
V^*(s)\leq\max_aQ^*(s,a)
```

于是得到：

```math
\boxed{V^*(s)=\max_aQ^*(s,a)}
```

证毕。

---

slides里的 Advantage 又是怎么回事？

图片使用的是：

```math
A^*(s,a)=Q^*(s,a)-V^*(s)
```

现在我们已经证明：

```math
V^*(s)=\max_aQ^*(s,a)
```

所以：

```math
A^*(s,a)
=

Q^*(s,a)-\max_{a'}Q^*(s,a')
```

因为 $`V^*(s)`$ 是所有 $`Q^*(s,a)`$ 中最大的那个，所以对于任何动作：

```math
Q^*(s,a)\leq V^*(s)
```

因此：

```math
\boxed{A^*(s,a)\leq0}
```

而对于最优动作 (a^*)：

```math
Q^*(s,a^*)=V^*(s)
```

所以：

```math
A^*(s,a^*)=0
```

因此：

```math
\boxed{\max_aA^*(s,a)=0}
```

---

整个逻辑可以压缩成一条链：

```math
\boxed{
V^*(s)=\max_aQ^*(s,a)
}
```

于是

```math
\boxed{
A^*(s,a)=Q^*(s,a)-V^*(s)\leq0
}
```

并且最优动作满足

```math
\boxed{
A^*(s,a^*)=0
}
```

所以

```math
\boxed{
\max_aA^*(s,a)=0
}
```

---

这里有一个容易和普通 Advantage 混淆的地方：

对于**任意策略** ($\pi$)，

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

并不意味着它一定 ($\leq0$)。

例如一个非最优策略可能在状态 $s$ 下主要选择了一个很差的动作，那么另一个动作完全可能有：

$$
Q^\pi(s,a)>V^\pi(s)
$$

所以：

$$
A^\pi(s,a)>0
$$

真正满足

```math
A^*(s,a)\leq0
```

的是**最优 Advantage**。

这也是为什么slides强调的是 ($`A^*`$)，而不是一般的 ($A^\pi$)。

>那么我们回到 slides，我们对等式进行一个变换，可以将$`Q^*`$表示成以下式子。

![Pasted image 20260819162119](附件/Pasted%20image%2020260819162119.png)
>那么我们训练两个网络，一个 V 和一个 A 即可得到 Q。

![Pasted image 20260819162444](附件/Pasted%20image%2020260819162444.png)

>为什么要加一个 max 项。因为如果不加会出现 Non-identifiability 的问题。

![Pasted image 20260819162647](附件/Pasted%20image%2020260819162647.png)
>**为什么Dueling Network 的“要把一个 Q 网络硬拆成两个网络?”**

> 在很多状态下，**“这个状态本身值多少钱”** 和 **“不同动作之间到底差多少”** 是两种不同的信息。

所以它把

$$
Q(s,a)
$$

重新参数化成

$$
Q(s,a)=V(s)+A(s,a)
$$

然后分别学习 $V$ 和 $A$。

---

先直观理解：Q 其实包含两类信息

假设你在玩游戏，现在到了一个状态 (s)。

这个状态可能是：

> “我已经占据了很有利的位置，基本快赢了。”

那么这里首先存在一个问题：

这个状态本身值多少钱？

这就是 $V(s)$

比如：

$$
V(s)=100
$$

但即使这个状态很好，不同动作可能仍然有一些区别：

|动作|Q(s,a)|
|---|--:|
|向左|98|
|向右|100|
|攻击|99|
|防御|97|

这些动作相对于状态平均水平的“好坏差异”，就是 Advantage：

$$
A(s,a)=Q(s,a)-V(s)
$$

所以可以理解成：

```math
\boxed{  
Q(s,a)
=
\underbrace{V(s)}_{\text{这个状态本身有多好}}  
+  
\underbrace{A(s,a)}_{\text{这个动作相对来说有多好}}  
}
```

---
为什么这样拆有意义？

关键在于：

> **很多时候，动作之间其实没什么区别，但状态本身的信息非常重要。**

举个特别典型的例子。

假设一个游戏当前状态是：

> 你已经稳赢了，敌人只剩一点血，而且你站在一个绝对安全的位置。

这时候：

- 向左
    
- 向右
    
- 原地不动
    
- 攻击
    
- 防御
    

可能都不会改变最终结果太多。

于是：

$$
Q(s,a_1)\approx Q(s,a_2)\approx Q(s,a_3)
$$

也就是说：

> **这个状态很好，比“哪个动作最好”更加重要。**

但一个普通 DQN 需要直接学习：

$$
Q(s,a_1),Q(s,a_2),Q(s,a_3),\dots
$$

相当于要分别学习很多个几乎一样的东西。

而 Dueling Network 会学习：

$$
V(s)\approx 100
$$

再学习：

$$
A(s,a_1)\approx0
$$

$$
A(s,a_2)\approx0
$$

$$
A(s,a_3)\approx0
$$

于是：

$$
Q(s,a_i)=V(s)+A(s,a_i)
$$

这就非常自然。

---
这其实是在利用“共享信息”

这就是 Dueling Network 最重要的优势。

普通 DQN：

$$
s\rightarrow Q(s,a_1),Q(s,a_2),\dots,Q(s,a_n)
$$

Dueling：

```math
s  
\rightarrow  
\begin{cases}  
V(s)\\
A(s,a_1),A(s,a_2),\dots,A(s,a_n)  
\end{cases}  
\rightarrow Q(s,a)
```

其中： $V(s)$  

是一个**与动作无关的状态信息**。

一旦学会了它，就可以让所有动作共享这部分信息。

所以它实际上是在做：

> **把“状态信息”和“动作差异信息”解耦。**

---
一个特别经典的例子：赛车

想象你在玩赛车。

当前状态：

> 车辆在直道上高速行驶，前方没有障碍。

此时：

- 左转
    
- 右转
    
- 轻微转向
    
- 保持方向
    

可能暂时都差不多。

因此：

$$
Q(s,a_1)\approx Q(s,a_2)\approx Q(s,a_3)
$$

真正重要的是：

> “我现在是不是处于一个好的状态？”

也就是： $V(s)$  

只有当出现：

> 前面有障碍物

或者：

> 马上要过弯

的时候，不同动作的重要性才突然变得不同。

也就是说： $V(s)$  

可以描述：

> **当前局面整体怎么样**

而

$$
A(s,a)
$$

描述：

> **在这个局面下，选哪个动作更好**

这个分工非常符合很多决策任务的结构。

---
那为什么普通 DQN 做不到？

其实普通 DQN **理论上当然也能学到这些东西**。

它不是说：

> “DQN 无法学习状态价值。”

而是：

> **Dueling 提供了一种更加容易学习的参数化方式。**

普通 DQN 直接学习：

$$
Q(s,a)
$$

例如有 10 个动作，就要直接输出：

$$
Q(s,a_1),\dots,Q(s,a_{10})
$$

而 Dueling 学： $V(s)$  和 $A(s,a_1),\dots,A(s,a_{10})$

于是：

$$
Q(s,a)=V(s)+A(s,a)
$$

这样网络会更明确地利用：

- 状态共享信息 → $V$
    
- 动作区别信息 → $A$
    

---
这也解释了为什么 Dueling 对“动作很多”的问题特别有帮助

假设有 100 个动作。

普通 DQN：

$$
Q(s,a_1),Q(s,a_2),\dots,Q(s,a_{100})
$$

每个动作都需要输出一个 Q 值。

但很多情况下：

> 大部分动作的价值其实差不多。

例如：

$$
Q(s,a_1)\approx Q(s,a_2)\approx\dots\approx Q(s,a_{100})
$$

那么真正需要学习的东西其实只有：

一个“这个状态怎么样”的量 $V(s)$  

再加：
少量“动作之间的区别” $A(s,a)$

因此参数化会更加高效。
所以 Dueling 的真正优势可以概括成一句话

不是：

> “两个网络比一个网络厉害。”

而是：

> **把 Q 函数中“状态本身的价值”和“动作之间的相对优势”分离出来，让网络能够更有效地共享状态信息。**

也就是：

```math
\boxed{  
Q(s,a)

\underbrace{V(s)}_{\text{状态有多好}}  
+  
\underbrace{A(s,a)}_{\text{动作有多好}}  
}
```

---
后续还有关于 [Double DQN(From GPT)](Double%20DQN(From%20GPT).md) 的介绍，就不放在这个文档中了。