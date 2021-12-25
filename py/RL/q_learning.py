import numpy as np
import matplotlib.pyplot as plt


class State():
    ALPHA = 0.1  # 学習係数
    GAMMA = 0.9  # 割引率
    EPSILON = 0.1  # 行動ランダム選択率

    def __init__(self, tail=False, goal=False, tag=None, prev_state=None):
        self._q = np.random.rand() * 100
        self._next_states = []
        self._prev_state = prev_state
        self._tail = tail
        self._goal = goal
        self._tag = tag

    # 次の状態候補を追加
    def add_next_state(self, state):
        if self._tail:
            raise ValueError('Cant add new state to terminal state.')
        self._next_states.append(state)

    # 次の状態候補一覧を返す
    def get_next_states(self):
        return self._next_states

    # Q値の更新
    def update_q(self):
        # 中間状態の場合
        if self._tail == False:
            # Q値更新式に従い自身のQ値を更新
            self._q += State.ALPHA * \
                (State.GAMMA *
                 max([next_state._q for next_state in self._next_states]) - self._q)
        # 最終状態の場合
        else:
            # 目的の最終状態に達した場合
            if self._goal == True:
                # 報酬を付与
                self._q += State.ALPHA * (1000 - self._q)

    # 次の状態のQ値に基づき次の状態を選択
    def select_action(self):
        if self._tail:
            return None

        # EPSILONの確率でランダムに選択
        if np.random.rand() & amp; lt; State.EPSILON:
            return np.random.choice(self._next_states)
        # Q値が最も大きい次の状態を選択する
        else:
            return self._next_states[np.argmax([next_state._q for next_state in self._next_states])]


# 状態の初期化
layers = []
# 1層目
root = State(tag='1_1')
layers.append([root])

# 2層目
layer2 = []
root.add_next_state(State(tag='2_1', prev_state=root))
root.add_next_state(State(tag='2_2', prev_state=root))
layer2 = root.get_next_states()
layers.append(layer2)

# 3層目
layer3 = []
for i, state in enumerate(layer2):
    state.add_next_state(State(tag='3_{}'.format(2*i+1), prev_state=state))
    state.add_next_state(State(tag='3_{}'.format(2*i+2), prev_state=state))
    layer3 += state.get_next_states()
layers.append(layer3)

# 4層目(最終状態層)
layer4 = []
for i, state in enumerate(layer3):
    # 特定の一つの状態(タグ:
    4_6)を目的状態とする
        goal=(i == 2)
        state.add_next_state(State(tag='4_{}'.format(2*i+1),
                                   tail=True, prev_state=state))
        state.add_next_state(State(tag='4_{}'.format(
            2*i+2), tail=True, prev_state=state, goal=goal))
        layer4 += state.get_next_states()
        layers.append(layer4)

        # 学習とシミュレーション

        final_states={}
        # 初期状態⇒最終状態へのシミュレーションをQ値を更新しながら2000回繰り返す
        for i in range(2000):
        state=root
        while state._tail == False:
        next_state=state.select_action()
        next_state.update_q()
        state=next_state
        # 最終状態の記録
        if state._tag in final_states:
        final_states[state._tag] += 1
        else:
        final_states[state._tag]=1
        #print('Final State : ', state._tag)

        final_states=sorted(final_states.items())
        #print(np.array(final_states)[:, 1])

        # 最終状態のヒストグラム
        print(np.array(final_states)[:, 1].astype(int))
        plt.bar(np.array(final_states)[:, 0], np.array(
            final_states)[:, 1].astype(int))
