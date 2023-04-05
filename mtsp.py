# %% [markdown]
# 问题描述：
# m个旅行商去旅游 n个城市，一种规定是都必须从同一个出发点出发，而且返回原出发点，需要将所有的城市遍历完毕，每个城市只能游历一次，但是为了路径最短可以路过这个城市多次。
# 两种问题：
# 1. 多个旅行商从不同的城市出发，遍历所有的目标点并回到自己的原点。
# 2. 多个旅行商从同一个点出发回到所有起点，还是回到同一个点但该点不是起点。
# 另外，每个旅行商访问城市的数量是任意的，还是两个旅行商需要访问的城市数目是大致相等的，还是规定了每个旅行商访问的数量具体是多少？
# 本次实现多个旅行商从同一点出发，回到同一个起点，且每人访问的城市数量相同的情形。首先，跟TSP问题一样，初始化大小为M的种群，每个个体是一个不含重复点的长度为n的序列。然后重点是设置断点，将该序列拆分成m段，也就是旅行商的数量。此后，将m段序列的首尾都添上起点，形成m个旅行商各自的环路径，然后计算各个环的距离的总和的倒数，作为适应度函数的值，后续操作按照TSP问题的遗传算法求解即可。在本例中，考虑的m取2，两位旅行商。断点取序列的中间，也就是序列平均分为两半。若读者想不限制各位旅行商的访问城市的数目相等，只需要设置断点在不同的位置即可。

# %%
import numpy as np
from matplotlib import pyplot as plt
import math
import copy
import random
import time
from tqdm import tqdm

# %% [markdown]
# ## 种群初始化
#    首先是初始化种群，TSP里的初始化种群其实就是一个长度为n=10的包含$1 - n$且不含重复元素的序列，其意义就是一个人从某个点出发，随机访问下一个未访问过的城市直到所有的城市都访问完毕，他再从最后一个城市返回出发城市。它的轨迹就是一个包含了所有城市且没有重复访问的环。种群数量设为M，那么该种群初始化的意义就是M个人独立、随机、不重复地访问一遍各个城市，单个个体轨迹构成一个环。

# %%
def init_population(length, num):
    """初始化群体产生

    Args:
        length (_type_): 一个群体中个体的长度
        num (_type_): 要产生的群体中个体的数量

    Returns:
        _type_: 产生的群体
    """
    li = list(range(length))
    # print(li)
    population = []
    for i in range(num):
        random.shuffle(li)
        population.append(copy.deepcopy(li))
    return population

# %% [markdown]
# ## 适应度计算
# m个个体的轨迹得到了，这些轨迹是随机游走得到的，当然很可能远远不包含属于最优解，而是比最优解坏的多。但是，随机游走的路径也有大有小，有好有坏。因此需要有一个函数衡量个体的好坏，也就是环路径的长短。这个函数就被称为适应度函数，其功能是衡量个体的好坏。对于TSP问题，衡量标准就是该个体序列的路径长度。路径越长，个体越坏，路径越短，个体越好。因此，该序列为x，那么该序列的路径长度便为d(x)，而适应度函数则应该取为1/d(x)，适应度越大，个体越优。

# %%
def aimFunction(entity, DMAT, break_points):
    """
    目标函数，这里计算的是环路径长度，然后得到适应度
    :param entity: 个体（种群中的个体）
    :param DMAT: 距离矩阵
    :param break_points: 切断点的位置
    :return: 适应度函数值
    """
    distance = 0
    break_points.insert(0, 0)
    break_points.append(len(entity))
    # print("当前的break_points是", break_points)
    # print("break_points的长度:", len(break_points))
    routes = []
    # 将整体路径拆成m段.break_points的长度设置为k时，则将路径拆分成了k+1段，此处的k取的1
    for i in range(len(break_points) - 1):
        routes.append(entity[break_points[i]:break_points[i + 1]])
    # print("根据旅行者数量的得到的路径:", routes)
    for route in routes:
        # 先去除随机路径中的0点
        if 0 in route:
            route.remove(0)
        # print("去0点后的路径:", route)
        # 此处给每段路径添上首尾点0，因为本例所有旅行商都从0这个城市出发，具体从哪里出发根据实际问题修改该设定
        # 将m段序列的首尾都添上起点，形成m个旅行商各自的环路径
        route.insert(0, 0)
        route.append(0)
        # print("插入首尾起点的环路径:", route)
        for i in range(len(route) - 1):
            # print("当前是第",route[i],"行",route[i + 1],"列")
            distance += DMAT[route[i], route[i + 1]]
    # 返回种群中单个个体的适应度
    return 1.0/distance


# %%
# 返回种群所有个体的适应度列表
def fitness(population, DMAT, break_points, aimFunction):
    """适应度
    Args:
        population (_type_): 种群
        DMAT (_type_): 距离矩阵
        break_points (_type_): 切断点
        aimFunction (_type_): 目标函数
        return: 种群所有个体的适应度列表
    """
    value = []
    for i in range(len(population)):
        value.append(aimFunction(population[i], DMAT, copy.deepcopy(break_points)))
        # weed out negative value
        if value[i] < 0:
            value[i] = 0
    return value

# %% [markdown]
# ## 选择
# 有了每个个体的适应度，就能评价每个个体的好坏。物竞天择，优胜劣汰，将优秀的个体选择出来进行交配，以期得到更好的个体，并由此不断进化，一代代传承，后代不断比前一代变得更好，最终收敛，种群中的某个个体达到了优秀的极限，便是最优解。
# 对于TSP问题，选择的具有操作是，计算出所有个体的适应度，也就是路径距离的倒数。然后将所有个体的适应度归一化，得到概率。然后从数量为n的个体中以轮盘的形式选择出若干个体，视为优秀个体。

# %%
# 这里传入的是种群和每个个体的适应度
# 这里的轮盘赌与蚁群算法的有一定区别。这里对适应度归一化得到概率后，每个个体被选中的概率就是这个概率
# 对每次被选中的个体的数目没有限制，完全随机，限制的是选择次数n与种群数目相同，使得新的种群数量与旧的种群数量一致
def selection(population, value):
    """轮盘赌选择
    说是轮盘赌选择，实际是随机选择，后续修改为轮盘赌选择

    Args:
        population (_type_): 种群
        value (_type_): 种群中所有个体的适应度

    Returns:
        _type_: 新的种群
    """
    # 轮盘赌选择
    fitness_sum = []
    for i in range(len(value)):
        if i == 0:
            fitness_sum.append(value[i])
        else:
            fitness_sum.append(fitness_sum[i - 1] + value[i])
        # print("fitness_sum[i - 1]:", fitness_sum[i - 1], "value[i]:", value[i])
        # print("fitness_sum:", fitness_sum)
    # print("sum fitness_sum:", sum(fitness_sum),"sum value:", sum(value))
    for i in range(len(fitness_sum)):
        fitness_sum[i] /= sum(value)
    # print("fitness 概率:", fitness_sum)
    
    # select new population
    population_new = []
    for i in range(len(value)):
        rand = np.random.uniform(0, 1)
        for j in range(len(value)):
            if j == 0:
                if 0 < rand and rand <= fitness_sum[j]:
                    population_new.append(population[j])
            else:
                if fitness_sum[j - 1] < rand and rand <= fitness_sum[j]:
                    population_new.append(population[j])
            # print("当前j是:", j)
            # print("当前的population_new是:", population_new)
    # print("新的种群：", population_new)
    return population_new

# %%
# 对于被选中的双亲，随机两两组合。并以pc的概率交配
# 若没有被选择交配，则直接双亲进入下一代。如果被选择，则交换中间片段。
def amend(entity, low, high):
    """修正个体

    Args:
        entity (_type_): 个体
        low (_type_): 交叉点最低处
        high (_type_): 交叉点最高处
        return: 
    """
    length = len(entity)
    # 交叉基因
    cross_gene = entity[low: high]
    # 应交叉基因，但还没有交叉
    not_in_cross = []
    # 非交叉基因，里面存在需要交叉的基因
    raw = entity[0: low] + entity[high:]
    
    for i in range(length):
        if not i in cross_gene:
            not_in_cross.append(i)
    
    error_index = []
    for i in range(len(raw)):
        if raw[i] in not_in_cross:
            not_in_cross.remove(raw[i])
        else:
            error_index.append(i)
    for i in range(len(error_index)):
        raw[error_index[i]] = not_in_cross[i]
        
    entity = raw[0 : low] + cross_gene + raw[low:]
    return entity

# %% [markdown]
# ## 交叉
# 遗传算法中的交叉属于遗传算法的核心以及关键步骤，其思想启发于生物遗传中的染色体交叉

# %%
def crossover(population_new, pc):
    """交叉

    Args:
        population_new (_type_): 种群
        pc (_type_): 交叉概率
        return:
    """
    half = int(len(population_new) / 2)
    father = population_new[: half]
    mother = population_new[half: ]
    np.random.shuffle(father)
    np.random.shuffle(mother)
    offspring = []
    for i in range(half):
        if np.random.uniform(0, 1) <= pc:
            cut1 = 0
            cut2 = np.random.randint(0, len(population_new[0]))
            if cut1 > cut2:
                cut1, cut2 = cut2, cut1
            if cut1 == cut2:
                son = father[i]
                daughter = mother[i]
            else:
                son = father[i][0:cut1] + mother[i][cut1:cut2] + father[i][cut2:]
                son = amend(son, cut1, cut2)
                daughter = mother[i][0:cut1] + father[i][cut1:cut2] + mother[i][cut2:]
                daughter = amend(daughter, cut1, cut2)
                
        else:
            son = father[i]
            daughter = mother[i]
        offspring.append(son)
        offspring.append(daughter)
    
    return offspring

# %% [markdown]
# ## 变异
# 对于交叉得到的大小为m的种群，以pm的概率选择出部分个体进行变异操作。选择出来的个体序列随机选择两个位置上的数进行交换。

# %% [markdown]
# ## 传承
# 种群的迭代更新方式还剩最后一步操作，传承。将上一轮中最优的个体（对于TSP问题，也就是路径距离最短的序列）保留下来，并用它替换掉新产生的种群中最差的个体。这一步的意义在于，上一轮中最优的个体被选择的几率也最大，但是它与其他个体交叉之后得到的新个体不一定优于自己。如果这样，那么这个最优的个体便被覆盖、迭代掉了，这样很可能造成下一代中所有个体都比上一代的最优个体差，优秀基因被淘汰。为了避免这样的情况，需要这一步传承，保证每次迭代之后的最优个体不坏于之前出现的所有个体。

# %%
# 这里的变异是最简单的变异法则，直接随机选取两个位置上的数进行交换
def mutaiton(offspring, pm):
    """变异

    Args:
        offspring (_type_): 子代个体
        pm (_type_): 交换概率

    Returns:
        _type_: _description_
    """
    for i in range(len(offspring)):
        if np.random.uniform(0, 1) <= pm:
            position1 = np.random.randint(0, len(offspring[i]))
            position2 = np.random.randint(0, len(offspring[i]))
            offspring[i][position1], offspring[i][position2] = offspring[i][position2], offspring[i][position1]
    return offspring
    

# %%
'''
Author: jia
Date: 2023-04-03 16:07:49
LastEditors: jia
LastEditTime: 2023-04-04 21:15:15
Description: 请填写简介
'''
# 主函数
if __name__ == '__main__':
    t_begin = time.time()
    # 这里的graph 为距离矩阵，需要自己定义后传入
    xpoints = np.random.randint(1000, size=(1, 22))
    ypoints = np.random.randint(1000, size=(1, 22))
    
    x2points = xpoints
    y2points = ypoints
    
    # 计算两点之间的距离，并保存在graph中
    graph = np.zeros((22,22), dtype =float)
    for i in range(len(xpoints[0])):
        for j in range(len(x2points[0])):
            graph[i][j] = math.sqrt(math.pow((xpoints[0][i] - x2points[0][j]), 2) + math.pow((ypoints[0][i] - y2points[0][j]), 2))
    # print("graph[0][1]:",graph[0][1],"graph[1][0]:",graph[1][0])
    print("距离矩阵:", graph)
    DMAT = graph
    # 这里是将所有城市从中间断开成两段。读者可以根据需求切断成任意段（每段代表一个旅行商），并且可以在任意位置切断。
    # 两位旅行商，断点取序列的中间，也就是序列平均分为两半
    break_points = [len(graph)//3, len(graph)//3*2]
    # 种群初始化，生成90个个体的种群
    population = init_population(len(graph), 90)
    
    t = []
    dic_result={}
    for i in tqdm(range(30000)):
        # 计算适应度 并 selection
        value = fitness(population, DMAT, break_points, aimFunction)
        population_new = selection(population, value)
        # crossover
        offspring = crossover(population_new, 0.65)
        # mutation
        population = mutaiton(offspring, 0.02)
        
        # 存储种群中所有个体的的距离值
        result = []
        for j in range(len(population)):
            result.append(1.0 / aimFunction(population[j], DMAT, copy.deepcopy(break_points)))
        
        # 存储当前距离值中的最小值，及其对应的个体
        t.append(min(result))
        min_entity = population[result.index(min(result))]
        
        # 将求得的路径根据旅行商个数分段
        routes = []
        break_points_plt = copy.deepcopy(break_points)
        break_points_plt.insert(0, 0)
        break_points_plt.append(len(min_entity))
        for n in range(len(break_points_plt) - 1):
            routes.append(min_entity[break_points_plt[n]:break_points_plt[n + 1]])
        for route in routes:
            if 0 in route:
                route.remove(0)
            route.insert(0,0)
            route.append(0)
        dic_result[min(result)]=routes
        # print("curretn i = ",i, "i % 400 = ", i%400)
        # if i % 400 == 0:
        #     # print("dic_result:", dic_result)
        #     min_entity = population[result.index(min(result))]
        #     routes = []
        #     break_points_plt = copy.deepcopy(break_points)
        #     break_points_plt.insert(0, 0)
        #     break_points_plt.append(len(min_entity))
        #     for i in range(len(break_points_plt) - 1):
        #         routes.append(min_entity[break_points_plt[i]:break_points_plt[i + 1]])
        #     for route in routes:
        #         if 0 in route:
        #             route.remove(0)
        #         route.insert(0,0)
        #         route.append(0)
        #     for route in routes:
        #         print(route) 
    t_end = time.time()
    print('运行时间为',t_end - t_begin,'秒')
    print('最短路径距离之和为',min(t),'米')#每次迭代的最优路径
    print("最优路径:", dic_result[min(t)])#最优路径
    
    min_route = dic_result[min(t)]
    
    minx_points = []
    miny_points = []
    
    minx2_points = []
    miny2_points = []
    
    minx3_points = []
    miny3_points = []
    
    for x in min_route[0]:
        minx_points.append(xpoints[0][x])
        miny_points.append(ypoints[0][x])
        
    for y in min_route[1]:
        minx2_points.append(xpoints[0][y])
        miny2_points.append(ypoints[0][y])
        
    for z in min_route[2]:
        minx3_points.append(xpoints[0][z])
        miny3_points.append(ypoints[0][z])
    
    minx_points_array = np.array(minx_points)
    miny_points_array = np.array(miny_points)
    minx2_points_array = np.array(minx2_points)
    miny2_points_array = np.array(miny2_points)
    minx3_points_array = np.array(minx3_points)
    miny3_points_array = np.array(miny3_points)
    
    plt.figure(1)
    plt.title("Target points")
    # plt.plot(minx_points_array, miny_points_array, 'r')
    # plt.plot(minx2_points_array, miny2_points_array, 'b')
    for i in range(len(minx_points) - 1):
        plt.annotate('',xy=(minx_points_array[i+1],miny_points_array[i+1]),xytext=(minx_points_array[i],miny_points_array[i]),arrowprops=dict(facecolor = "r",width=2,connectionstyle="arc3"))
    for i in range(len(minx2_points) - 1):
        plt.annotate('',xy=(minx2_points_array[i+1],miny2_points_array[i+1]),xytext=(minx2_points_array[i],miny2_points_array[i]),arrowprops=dict(facecolor="b",width=2,connectionstyle="arc3"))
    for i in range(len(minx3_points) - 1):
        plt.annotate('',xy=(minx3_points_array[i+1],miny3_points_array[i+1]),xytext=(minx3_points_array[i],miny3_points_array[i]),arrowprops=dict(facecolor="g",width=2,connectionstyle="arc3"))
  
    plt.plot(xpoints[0], ypoints[0], 'ok') #画出目标点
    
    plt.figure(2)
    plt.title("The loop path corresponding to the optimal individual in each iteration")
    plt.plot(t)#画出每次迭代的最优个体对应的环路程
    
    plt.show()


