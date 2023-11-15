import numpy as np

# 定义状态空间和技能空间
states = np.array(
    [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [0.2, 0.1, 0.4, 0.3], [0.3, 0.4, 0.1, 0.2], [0.1, 0.2, 0.1, 0.6],
     [0.7, 0.1, 0.1, 0.1], [0.8, 0.05, 0.05, 0.1], [0.05, 0.9, 0.05, 0], [0.05, 0.05, 0.8, 0.1], [0.2, 0.2, 0.2, 0.4]])
skill_1_indices = [0, 1, 2, 3]
skill_2_indices = [4, 5, 6, 7, 8, 9]


# 计算每个技能的状态熵
def get_entropy(state_indices):
    state_probs = states[state_indices]
    probs_sum = np.sum(state_probs, axis=0)
    normalized_probs = state_probs / probs_sum
    entropy = -np.sum(normalized_probs * np.log(normalized_probs + 1e-6))
    return entropy


skill_1_entropy = get_entropy(skill_1_indices)
skill_2_entropy = get_entropy(skill_2_indices)

# 测试智能体所处的技能：
test_state = np.array([0.4, 0.3, 0.05, 0.25])
test_entropy = -np.sum(test_state * np.log(test_state + 1e-6))
if test_entropy < skill_1_entropy:
    print("The agent is currently using Skill 1.")
else:
    print("The agent is currently using Skill 2.")
