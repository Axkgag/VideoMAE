import numpy as np
from collections import deque

class StateMachine():
    """
        pattern: 标准序列
        state_cache: 缓冲区，存满一定数量才会将动作放入状态序列
        state_list: 动作序列
    """
    def __init__(self, pattern, frame_thres=5) -> None:
        self.pattern = pattern + pattern
        self.num_stat = len(pattern)

        self.frame_thres = frame_thres
        self.state_cache = []
        self.state_list = deque(maxlen=self.num_stat)

    def getStateList(self, new_state):
        if not self.state_cache:  
            # 缓冲区为空，则先将动作加入缓冲区
            self.state_cache.append(new_state)
        elif new_state != self.state_cache[-1] and len(self.state_cache) < self.frame_thres:
            # 当前动作与缓冲区内动作不一致时，清空缓冲区后再加入缓冲区
            self.state_cache.clear()
            self.state_cache.append(new_state)
        elif new_state == self.state_cache[-1] and len(self.state_cache) < self.frame_thres:
            # 当前动作与缓冲区内动作一致时，直接加入缓冲区
            self.state_cache.append(new_state)

        if len(self.state_cache) == self.frame_thres:
            # 当缓冲区满时，将缓冲区内动作放入动作序列，并清空缓冲区
            if not self.state_list or self.state_cache[-1] != self.state_list[-1]:
                self.state_list.append(new_state)
            self.state_cache.clear()
        
    def checkState(self, new_state):
        if new_state == None:
            return True
        
        self.getStateList(new_state)

        if not self.state_list:
            return True
        
        stat_list = list(self.state_list)
        win_size = len(stat_list)

        for i in range(self.num_stat):
            if self.pattern[i: i + win_size] == stat_list:
                return True
            
        self.state_list.clear()
        return False
        
    
if __name__ == "__main__":
    pattern = ['1', '2', '0', '1', '5']
    
    stat_list = ['1', '2', '0', '2', '5']

    FSM = StateMachine(pattern, 1)


    for state in stat_list:
        print(state, FSM.checkState(state))


