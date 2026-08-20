## 参考课程

https://www.bilibili.com/video/BV1TMcUzKENZ?spm_id_from=333.788.videopod.episodes&vd_source=b17d4de2c32b04e437ad7699ea8a76ea
![Pasted image 20260812112632.png](附件/Pasted%20image%2020260812112632.png)

## What is RL？

>**强化学习本质上也可以用机器学习的通用范式来解释。即目标函数，损失函数，优化方法三部分。**

>**Step1**：即目标是学一个 Function，这个 Function 在强化学习里是一个 Policy Network，它接受 Environment 输出的 Observation，然后输出 Action 的概率分布。最终 Action 是基于这个概率分布即 Scores 的采样（Sample）。

![Pasted image 20260811130127.png](附件/Pasted%20image%2020260811130127.png)
>**Step2**：强化学习里的损失函数可以看作是总奖励的取负，强化学习的优化目标即是Total reward 最大。

![Pasted image 20260811130213.png](附件/Pasted%20image%2020260811130213.png)

>**Step3**：强化学习的 $\{s_1,a_1,s_2,a_2,...\}$ 序列即轨迹 Trajectory（$`\tau`$）。优化目标即找到这样的 Trajectory 来让 $R(\tau)$ 最大。这里的 Actor 即是目标的策略网络，类似 GAN 中的 Generator。Env 和 Reward 其实类似 [GAN](../GAN/GAN.md) 中的 Discriminator。但是强化学习里的最大问题在于 Env 和 Reward 输出的不稳定性，即同样的 $s_1$ 同样的 $a_1$，但是所产生的下一步 $s_2$ 完全可能是不一样的。这会导致整个优化过程的不稳定。

![Pasted image 20260811130329.png](附件/Pasted%20image%2020260811130329.png)
## Policy Gradient

>下面是一个经典的 Policy Gradient 循环，可以看到 Data Collection 在 For 循环内部，这表示每次训练 Actor 的时候都需要 Data Collection 一遍，这种就是 On-Policy 类型的强化学习，比如 Alphazero 就是如此。而 Off-Policy 可以不用每次进行 Data Collection 。 [PPO](PPO.md) 即是Off-Policy。


![Pasted image 20260811142941.png](附件/Pasted%20image%2020260811142941.png)
![Pasted image 20260811143037.png](附件/Pasted%20image%2020260811143037.png)
## Actor-Critic

>当我们知道了 Actor 怎么更新后，我们需要思考如何评判Actor 的输出。即在我们难以直接求出 G1‘时我们应该如何去预测。典型的就是 Value Function 这在 Alphazero 中也有，即输入的是 state，输出接下来可能得到的 discounted cumulated reward 即经过折扣因子计算后的总奖励。

![Pasted image 20260811155434.png](附件/Pasted%20image%2020260811155434.png)

>如何对 Critic 进行估计如今主要有两个方法。一是通过 MC 方式，一个是通过 TD 方式。

>MC 方式很好理解，即通过模拟（仿真）来直接搜索出 Ga‘ 来，然后只要学习让模型输出与 Ga‘ 保持一致即可。

![Pasted image 20260811160339.png](附件/Pasted%20image%2020260811160339.png)
>相对来说 TD 更加适合没有尽头的游戏，无法模拟出结果的游戏，无法获取 Ga‘ 自然无法训练 V。所以 TD 的思路本质是拟合贝尔曼方程的两边。让两边保持一致。

![Pasted image 20260811160842.png](附件/Pasted%20image%2020260811160842.png)
![Pasted image 20260812163646.png](附件/Pasted%20image%2020260812163646.png)
>那么 Critic 有什么用呢？我们看以下slides，描述了我们如何一步步改善对 Actor 的控制的。主要关注 Advantage Function 的变化。

>最初的版本我们可设置每一个状态-动作对一个标签，告诉模型什么动作是好的，什么是不好的。类似监督学习。

![Pasted image 20260811164855.png](附件/Pasted%20image%2020260811164855.png)
>进一步我们继续思考，每一个状态-动作对奖励应该是不同的。所以在上面的交叉熵损失的基础上我们增加权重系数。

![Pasted image 20260811165858.png](附件/Pasted%20image%2020260811165858.png)
>进一步我们把权重设置为 Reward

![Pasted image 20260811164601.png](附件/Pasted%20image%2020260811164601.png)
>进一步我们引入 Gt 作为权重系数，即 t 时刻后 Reward 总和。

![Pasted image 20260811170030.png](附件/Pasted%20image%2020260811170030.png)
>进一步我们引入折扣因子来降低长程依赖，得到G1’。

![Pasted image 20260811170103.png](附件/Pasted%20image%2020260811170103.png)

>由于 Reward 是相对的，可以对 G1‘ 进行标准化操作，让 Reward 有正有负。Advantage Function变为 Gt‘-b。

![Pasted image 20260811170154.png](附件/Pasted%20image%2020260811170154.png)
>进一步我们引入 Critic 后，Value Network 输出在状态s，按当前策略平均能拿到多少回报，相减 = **这个动作，比该状态下的平均水平好多少 / 差多少**。

![Pasted image 20260811164515.png](附件/Pasted%20image%2020260811164515.png)

>从 TD 视角我们可以继续将 Gt' 换为 $r_t+V(s_{t+1})$，到这里我们已经得到基础的 [GAE](GAE.md) 了。

![Pasted image 20260812100217.png](附件/Pasted%20image%2020260812100217.png)
>Actor 和 Critic 可以用同样的主干网络。

![Pasted image 20260812102145.png](附件/Pasted%20image%2020260812102145.png)
## Learning From Demonstration

>现实中的 Reward 往往是复杂的，不易被定义，或者稀疏的。我们可以人为地去设置一些 Reward，但应用场景往往有限。我们可以采用Imitation Learning。

![Pasted image 20260812104753.png](附件/Pasted%20image%2020260812104753.png)
>如果 Imitation Learning 只是 Supervise Learning (Behavior Cloning)，我们可以发现，这样会导致机器无法独自面对一些特殊情况，并且单纯地模仿会学到一些不必要的人类特有的行为。

![Pasted image 20260812111410.png](附件/Pasted%20image%2020260812111410.png)
>由此引出 IRL(Inverse Reinforcement Learning)。

![Pasted image 20260812112253.png](附件/Pasted%20image%2020260812112253.png)

>可以看出 IRL 的整个流程跟 GAN 很像。

![Pasted image 20260812112354.png](附件/Pasted%20image%2020260812112354.png)
![Pasted image 20260812112429.png](附件/Pasted%20image%2020260812112429.png)