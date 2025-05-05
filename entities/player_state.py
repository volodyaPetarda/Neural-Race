class PlayerState:
    def __init__(self, rewards_collected: int, deaths: int, time_elapsed: float, delta_time: float, last_reward_collected: float):
        self.rewards_collected = rewards_collected
        self.deaths = deaths
        self.time_elapsed = time_elapsed
        self.delta_time = delta_time
        self.last_reward_collected = last_reward_collected