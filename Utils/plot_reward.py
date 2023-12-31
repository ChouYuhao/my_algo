import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mlb
mlb.use('TkAgg')
sns.set(style="darkgrid") #这是seaborn默认的风格

def smooth(data, sm=10):
    '''
    :param data:
    :param sm: sm表示滑动窗口大小,为2*k+1,
    :return: smooth data
    '''
    smooth_data = []
    if sm > 1:
        for d in data:
            y = np.ones(sm)*1.0/sm
            d = np.convolve(y, d, "same")
            smooth_data.append(d)
    return smooth_data
def smoothing(data, sm=20):
    '''
    :param data:
    :param sm: sm表示滑动窗口大小,
    :return: smooth data
    '''
    data_smooth = []
    for i,j in enumerate(data):
        index_left = i - int(sm / 2) if (i - int(sm / 2)) >= 0 else 0
        index_right = i + int(sm / 2) if (i + int(sm / 2)) < len(data) else len(data) -1
        data_smooth.append(sum(data[index_left:index_right])/len(data[index_left:index_right]))
    return data_smooth

#2、数据max min mean
def get_max_min_mean(data, sm):
    '''
    :param data:
    :param sm:
    :return:
    '''
    data_max = []
    data_min = []
    data_mean = []
    for i,j in enumerate(data):
        index_left = i - int(sm / 2) if (i - int(sm / 2)) >= 0 else 0
        index_right = i + int(sm / 2) if (i + int(sm / 2)) < len(data) else len(data) -1
        data_max.append(max(data[index_left:index_right]))
        data_min.append(min(data[index_left:index_right]))
        data_mean.append(sum(data[index_left:index_right])/len(data[index_left:index_right]))
    return data_max, data_min, data_mean

def draw_line(data_file, label, color, max_min_mean_sm=6, smooth_sm=30):
    #数据处理
    #Wall time	Step	Value
    data = pd.read_csv(filepath_or_buffer=data_file)
    #DDPG_predator_4.head(10)
    # data_x = data['Step']
    data_x = data['Step']
    # data_y = data['Value']
    data_y = data['Value']

    data_max_y, data_min_y, data_mean_y \
        = get_max_min_mean(data_y, sm=max_min_mean_sm)

    data_max_smooth_y = smoothing(data=data_max_y, sm=smooth_sm)
    data_min_smooth_y = smoothing(data=data_min_y, sm=smooth_sm)
    data_mean_smooth_y = smoothing(data=data_mean_y, sm=smooth_sm)
    #print(data_mean_y)
    #画图
    plt.plot(data_x, data_mean_smooth_y, color=color, label=label, linewidth='2')

    plt.fill_between(data_x, data_min_smooth_y, data_mean_smooth_y, facecolor=color, alpha=0.3)
    plt.fill_between(data_x, data_mean_smooth_y, data_max_smooth_y, facecolor=color, alpha=0.3)

if __name__ == '__main__':
    plt.figure()
    data_files_path = [
        r'D:\browser_download\run-Hallway_1-tag-Goal_Correct.csv',
        r'D:\browser_download\run-Hallway_2-tag-Goal_Correct.csv',
        r'D:\browser_download\run-Hallway_3-tag-Goal_Correct.csv',
        # r'D:\browser_download\ablation_hallway4.csv',
        # r'D:\browser_download\ablation_pyramid_1.csv',
        # r'D:\browser_download\ablation_pyramid_2.csv',
        # r'D:\browser_download\run-Pyramid_All2-tag-Environment_Cumulative Reward.csv',
        # r'D:\browser_download\ablation_pyramid_4.csv',
        ]
    labels = ['HAL',
              'HPLSE(Ours)',
              'Option-Critic',
              ]
    colors = ['r', 'g', 'k',]
    # colors = ['r', 'g', 'k']
    smooth_sms = [10, 10, 10, ]
    # smooth_sms = [500, 500, 500]
    for data_file_path, label, color, smooth_sm in zip(data_files_path, labels, colors, smooth_sms):
        draw_line(data_file=data_file_path,
                  label=label,
                  color=color,
                  max_min_mean_sm=20,
                  smooth_sm=smooth_sm)
    plt.xlabel("Step")  # 横坐标名字
    plt.ylabel("Goal Correct")  # 纵坐标名字
    plt.title('Hallway UGV')

    plt.legend(loc="best")#图例
    # plt.legend(loc="lower right")  # 图例
    # plt.ylim(-10, 12)
    # plt.xlim(0, 250)

    # plt.subplots_adjust(left=0.117, right=0.983, top=0.988, bottom=0.11)
    plt.savefig("hallway_goalCorrect.pdf", dpi=500)
    plt.show()
