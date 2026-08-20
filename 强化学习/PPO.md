## 参考课程

https://www.youtube.com/watch?v=OAKAZhFmYoI
https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

## PPO简介
![Pasted image 20260812113014.png](附件/Pasted%20image%2020260812113014.png)

>先回顾下 Policy Gradient 的计算过程。

![Pasted image 20260812131157.png](附件/Pasted%20image%2020260812131157.png)
>$R(\tau^n)$我们称之为 Advantage Function，它的演化过程参考[RL Lecture](RL%20Lecture.md)。

>接下来我们思考如何从 On-policy 到 Off-policy。先看一个数学推导，当我们想得多$\mathbb{E}_{x\sim p}\big[f(x)\big]$时，我们如果无法对p(x)进行采样，可以通过以下手段，通过对q(x)采样代替对p(x)采样。

![Pasted image 20260812140451.png](附件/Pasted%20image%2020260812140451.png)
>但是这种替代不是完美的，他们的 Var 不同。

![Pasted image 20260812142146.png](附件/Pasted%20image%2020260812142146.png)
![Pasted image 20260812142405.png](附件/Pasted%20image%2020260812142405.png)
>下面我们将上面的数学推导引入 Policy Gradient，其中$\pi_{\theta'}$表示旧策略。新策略的分布不清楚，可以用旧策略的分布去估计。其中$d_{\pi_\theta}(s) \approx d_{\pi_{\theta'}}(s) \implies \frac{p_\theta(s)}{p_{\theta'}(s)}\approx 1$为 PPO 核心假设，即认为旧策略与新策略状态访问分布几乎一致。这里的$\boldsymbol{J^{\theta'}(\theta) = \mathbb{E}_{(s_t,a_t)\sim\pi_{\theta'}} \left[\;\frac{p_\theta(a_t|s_t)}{p_{\theta'}(a|s_t)}\; A^{\theta'}(s_t,a_t)\;\right]}$其实就是 Gradient 对$\theta$的积分，即$J^{\theta'}(\theta)$求导就得到Policy Gradient，所以$J^{\theta'}(\theta)$即是目标函数(Objective Function)。

>这里的 $A^{\theta'}(s_t,a_t)$可以参考[GAE](GAE.md)中的内容理解。

![Pasted image 20260812144448.png](附件/Pasted%20image%2020260812144448.png)
>我们对比下 PPO 与 TRPO，TRPO 是在上述优化目标的基础上加入了一个 KL 约束，待有约束的优化比较难做，所以在这个基础上 PPO 将约束放入了优化目标当中。需要注意的是，这里的 KL Divergence 并不是两个策略网络参数之间的 KL 散度，而是二者在同一个 state 所做的 action 之间的 KL 散度。

![Pasted image 20260812151750.png](附件/Pasted%20image%2020260812151750.png)
>下面是原版的 PPO 算法，它与 On-Policy 的算法之间的区别在于对于一组由$\theta_k$ collect的数据，可以进行多次的对$\theta$的更新。这里的 several times 一般在 3～10 次，即每次的数据可以复用 3～10 次，从这种意义上来说，PPO 不算完全的 Off-Policy 算法，其仍然是遵循 Data Collection and Network Training 这个循环。


![Pasted image 20260812152814.png](附件/Pasted%20image%2020260812152814.png)
>以下是 PPO 的一个变形，通过 clip 函数取代了 KL 散度。可以看到蓝色虚线代表后一项，绿色虚线代表前一项，最终我们取 min，得到红色线。本质上目标是约束$\theta$与$\theta_k$不能差别太大。

![Pasted image 20260812154035.png](附件/Pasted%20image%2020260812154035.png)
PPO多用于 LLMs 的后训练中，这点可以在[RL ON LLMs](RL%20ON%20LLMs(From%20GPT)/RL%20ON%20LLMs.md)中了解，在
