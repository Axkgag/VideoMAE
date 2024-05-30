import numpy as np
from collections import deque

class StateMachine():
    """
        current_state: 当前动作
        next_state: 等待动作
        pattern: 标准序列
        frame_thres: 等待时长（帧）
        cnt: 等待计数器
    """
    def __init__(self, pattern, time_thres=100, frame_thres=1) -> None:
        self.pattern = pattern
        self.num_stat = len(pattern)

        self.current_state = None
        self.next_state = None

        self.cnt = 0
        self.time_thres = time_thres

        self.frame_thres = frame_thres
        self.state_cache = []

    def get_CurrentState(self, new_state):
        if self.frame_thres <= 1 or new_state == None:
            return new_state
        
        if not self.state_cache:
            self.state_cache.append(new_state)
        elif new_state != self.state_cache[-1] and len(self.state_cache) < self.frame_thres:
            self.state_cache.clear()
            self.state_cache.append(new_state)
        elif new_state == self.state_cache[-1] and len(self.state_cache) < self.frame_thres:
            self.state_cache.append(new_state)

        if len(self.state_cache) == self.frame_thres:
            current_state = self.state_cache[-1]
            self.state_cache.clear()
            return current_state
        
        return None
                    
    def checkState(self, new_state):
        new_state = self.get_CurrentState(new_state)

        if new_state == None:
            return True
        
        if self.current_state == None or self.need_state == None:
            self.current_state = new_state
            state_id = self.pattern.index(new_state)
            next_id = (state_id + 1) % self.num_stat
            self.need_state = self.pattern[next_id]

        if new_state == self.current_state:
            self.cnt = 0
        elif new_state == self.need_state:
            self.cnt = 0
            self.current_state = new_state
            state_id = self.pattern.index(new_state)
            next_id = (state_id + 1) % self.num_stat
            self.need_state = self.pattern[next_id]
        else:
            self.cnt += 1
        
        if self.cnt == self.time_thres:
            self.cnt = 0
            self.current_state = None
            self.next_state = None
            return False
        
        return True
    
if __name__ == "__main__":
    pattern = ['1', '2', '0', '1', '5']
    
    stat_list = ['1', '2', '0', '2', '5']

    FSM = StateMachine(pattern, 1)


    for state in stat_list:
        print(state, FSM.checkState(state))


