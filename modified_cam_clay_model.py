#!/usr/bin/env python
# encoding: utf-8
"""

@file: modified_cam_clay_model.py
@time: 2019/11/9 21:36
@author: Luke
@email: guanshaoheng@qq.com
@application：
                 1.根据修正剑桥模型计算应力应变
                 2.常规三轴(drained & undrained)
"""

import numpy as np
import os
import math
from matplotlib import pyplot as plt


def get_tangent_modulus(load_slope, unload_slope, void_ratio, yita, M, p):
    tangent = np.ones([2, 2])
    tangent[0, 0] = load_slope*(M**2+yita**2)/((load_slope-unload_slope)*2*yita)
    tangent[1, 1] = 2*yita/(M**2-yita**2)
    tangent *= (2.0*(load_slope-unload_slope)*yita)/((1+void_ratio)*(M**2+yita**2)*p)
    return np.linalg.inv(tangent)


def plot_history(load_history):
    q = load_history[:, 0]-load_history[:, 1]
    p = (load_history[:, 0]+load_history[:, 1]+load_history[:, 2])/3.0
    pc = load_history[:, 9]
    vr_final = load_history[-1, 6]
    p_final = p[-1]

    # p-epsilon_1
    plt.figure(figsize=(16, 7))
    plt.subplot(241)
    plt.plot(load_history[:, 3], p)
    plt.title('p')
    plt.xlabel('epsilon_1')

    # q-epsilon_1
    plt.subplot(242)
    plt.plot(load_history[:, 3], q)
    plt.title('q')
    plt.xlabel('epsilon_1')

    # vr-epsilon_1
    plt.subplot(243)
    plt.plot(load_history[:, 3], load_history[:, 6])
    plt.title('vr')
    plt.xlabel('epsilon_1')

    # epsilon_v-epsilon_1
    plt.subplot(244)
    plt.plot(load_history[:, 3], load_history[:, 7])
    plt.title('epsilon_v')
    plt.xlabel('epsilon_1')

    # epsilon_q-epsilon_1
    plt.subplot(245)
    plt.plot(load_history[:, 3], load_history[:, 8])
    plt.title('epsilon_q')
    plt.xlabel('epsilon_1')

    # epsilon_q-epsilon_1
    plt.subplot(246)
    plt.plot(load_history[:, 3], load_history[:, 0])
    plt.title('sigma_1')
    plt.xlabel('epsilon_1')

    # epsilon_q-epsilon_1
    plt.subplot(247)
    plt.plot(load_history[:, 3], load_history[:, 2])
    plt.title('sigma_2')
    plt.xlabel('epsilon_1')

    # epsilon_q-epsilon_1
    plt.subplot(248)
    plt.plot(p, q, label='q-p')
    plt.plot(np.linspace(0, 1.5*(max(p))), M*np.linspace(0, 1.5*(max(p))), label='p*Mf')
    p_yield_1 = np.linspace(0, pc[0])
    q_yield_1 = M*np.sqrt((p_yield_1*pc[0]-p_yield_1*p_yield_1))
    p_yield_2 = np.linspace(0, pc[-1])
    q_yield_2 = M*np.sqrt((p_yield_2*pc[-1]-p_yield_2*p_yield_2))
    plt.plot(p_yield_1, q_yield_1, label='yield surface 1')
    plt.plot(p_yield_2, q_yield_2, label='yield surface 2')
    plt.title('q-p')
    plt.xlabel('p')
    plt.ylim([0, max(q*1.5)])
    plt.xlim([0, max(pc)])
    plt.legend()
    plt.show()

    # csl & ncl
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    p_max = 10*max(pc)
    ax.plot(np.linspace(1, p_max), N-1.0-l*np.log(np.linspace(1, p_max)), label='NCL')
    ax.plot(np.linspace(1, p_max), l*np.log(p_final)+vr_final-l*np.log(np.linspace(1, p_max)), label='CSL')
    ax.plot(p, load_history[:, 6], label='Load path')
    ax.set_xlim(1e0, p_max)
    plt.xlabel('lg(p)')
    plt.ylabel('void ratio')
    plt.legend()
    plt.show()


# initiation
# material properties
[l, k, M, poisn, N] = [0.2, 0.04, 0.95, 0.15, 2.5]
# [l, k, M, poisn] = [0.085, 0.0188, 1.36, 0.3]
# initial state
pc_origin = 200.  # consolidation pressure 前期固结压力
p0 = 150.  # confining pressure 围压
# e0 = 0.68
# N = e0+l*np.log(pc_origin)+1.0


def cam_clay():
    # initial volume 计算初始比体积
    v = N - l * np.log(pc_origin) + k * np.log((pc_origin / p0))
    vr = v-1.0
    # 计算设置
    iteration = 80*100
    input_de = 0.01/100
    # de = 0.01

    # initial stress & strain state 初始应力应变状态
    s = np.array([p0] * 3 + [0] * 3)
    strain = np.array([0.] * 6)

    # 加载增量计算
    dfds = np.zeros(6)
    dfdep = np.zeros(6)
    De = np.zeros([6, 6])
    epsv, epsd = 0., 0.
    p = sum(s[0:3]) / 3.0
    q = s[0] - s[2]
    pc = pc_origin
    load_history = [np.array(list(s[:3])+list(strain[:3])+[vr, epsv, epsd, pc])]
    for step in range(iteration):
        if step < iteration or step > 2*iteration:
            de = input_de
        else:
            de = -input_de
            if p <= p0:
                continue
        # bulk & shear modulus
        K = v*p/k
        G = (3*K*(1-2*poisn))/(2*(1+poisn))

        # 弹性矩阵
        for i in range(6):
            if i <= 2:
                De[i, i] = K + 4 / 3 * G
            else:
                De[i, i] = G
        for i in range(3):
            for j in range(3):
                if i != j:
                    De[i, j] = K-2/3*G

        # 加载模式
        # 常规三轴(drained)
        dstrain = np.array([de, -De[1, 0] / (De[1, 1] + De[1, 2]) * de, -De[2, 0] / (De[2, 1] + De[2, 2]) * de, 0., 0., 0.])
        # 常规三轴(undrained)
        # dstrain = np.array([de, -0.5 * de, -0.5 * de, 0., 0., 0.])

        # 判断加卸载
        dstress_e = np.dot(De, dstrain)
        s_try = s+dstress_e
        q_try = s_try[0]-s_try[2]
        p_try = sum(s_try[:3])/3.0
        f_yield = q_try ** 2 / M ** 2 + p_try ** 2 - p_try * pc
        f_yield = min(0, f_yield)

        # 若加载超过屈服面则更新 pc 【保证所有的应力都落在屈服面上】
        if f_yield < 0.:
            # 未达到屈服面按弹性计算
            D = De
        else:
            # 达到屈服面则计算弹塑性刚度阵
            pc = (q ** 2 / M ** 2 + p ** 2)/p
            # 计算dfds dfdep
            for n in range(6):
                for m in range(6):
                    if m <= 2:
                        if f_yield < 0:
                            dfds[m], dfdep[m] = 0., 0.
                        else:
                            dfds[m] = ((2 * p - pc) / 3.0 + 3 * (s[m] - p) / M ** 2)
                            dfdep[m] = ((-p) * pc * (1 + vr) / (l - k))

            dfds_mat = dfds.reshape([6, 1])
            dfdep_mat = dfdep.reshape([6, 1])
            D = De-np.matmul(np.matmul(De, dfds_mat), np.matmul(dfds_mat.T, De)) / \
                (-np.matmul(dfdep_mat.T, dfds_mat)+np.matmul(np.matmul(dfds_mat.T, De), dfds_mat))

        # 常规三轴(drained)
        dstrain = np.array([de, -D[1, 0] / (D[1, 1] + D[1, 2]) * de, -D[2, 0] / (D[2, 1] + D[2, 2]) * de, 0., 0., 0.])
        # 常规三轴(undrained)
        dstrain = np.array([de, -0.5 * de, -0.5 * de, 0., 0., 0.])

        depsv = sum(dstrain[:3])
        depsd = 2./3.*(dstrain[0]-dstrain[3])

        # 更新
        # 应力应变增量
        ds = np.dot(D, dstrain)
        s += ds
        strain += dstrain
        # 体应力 偏应力 体积 孔隙比
        p = sum(s[:3])/3.
        q = s[0]-s[2]
        pc = max(p, pc)   # !!!! 更新 最大历史 固结压力 pc
        v = N - l * np.log(pc) + k * np.log((pc / p))
        vr = v - 1.0
        epsv += depsv
        epsd += depsd
        load_history.append(np.array(list(s[:3])+list(strain[:3])+[vr, epsv, epsd, pc]))

    load_history = np.array(load_history)
    plot_history(load_history)
    return


if __name__ == '__main__':
    cam_clay()

