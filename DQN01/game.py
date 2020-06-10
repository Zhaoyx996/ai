import gym, torch, random
from torch import nn, optim

class QNet(nn.Sequential):
    def __init__(self):
        super(QNet, self).__init__(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 输出向左向右两个动作
        )

class Game:
    def __init__(self, exp_pool_size, explore=0.9):  # explore为探索值
        self.exp_pool = []
        self.exp_pool_size = exp_pool_size
        self.evn = gym.make('CartPole-v1')
        self.explore = explore
        self.loss_fn = nn.MSELoss()
        self.net = QNet()
        self.opt = optim.Adam(self.net.parameters())

    def __call__(self):
        is_render = False
        avg = 0

        # 数据采样，游戏的状态
        while True:
            R = 0
            state = self.evn.reset()
            while True:
                if is_render:
                    self.evn.render()

                if len(self.exp_pool) >= self.exp_pool_size:
                    self.exp_pool.pop(0)  # 经验池满了就把最旧的数据删去
                    self.explore += 0.000001
                    if random.random() > self.explore:
                        action = self.evn.action_space.sample()  # 当探索值大于阈值就随机采样动作
                    else:
                        _state = torch.tensor(state).float()
                        Qs = self.net(_state[None, ...])
                        action =Qs.argmax(dim=1)[0].item()
                else:
                    action = self.evn.action_space.sample()

                next_state, reward, done, _ = self.evn.step(action)
                R += reward
                self.exp_pool.append([state, reward, action, next_state, done])
                state = next_state
                if done:
                    avg = 0.95 * avg + 0.05 * R  # 当前状态对后续的影响较大，之前的状态影响较小
                    if avg > 300:
                        is_render=True
                    break

            if len(self.exp_pool) >= self.exp_pool_size:
                exps = random.choices(self.exp_pool, k=100)
                _state = torch.tensor([exp[0].tolist() for exp in exps])  # 状态时一个数组，有多种状态存在
                _reward = torch.tensor([[exp[1]] for exp in exps])
                _action = torch.tensor([[exp[2]] for exp in exps])
                _next_state = torch.tensor([exp[3].tolist() for exp in exps])
                _done = torch.tensor([[int(exp[4])] for exp in exps])

                # 得到估计值
                _Qs = self.net(_state)
                _Q = torch.gather(_Qs, 1, _action)

                # 目标值
                _next_Qs = self.net(_next_state)
                _max_Q = _next_Qs.max(dim=1, keepdim=True)[0]
                _target_Q = _reward + (1 - _done) * 0.9 * _max_Q

                loss = self.loss_fn(_Q, _target_Q.detach())

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                print(loss)

if __name__ == '__main__':
    game = Game(10000)
    game()








