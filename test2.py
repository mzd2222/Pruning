from typing import List


class EdgeServer:
    def __init__(self, devices: List, speed: float, r_sp: float, r_sp_2: float, d_sp: float, d_sp_2: float,
                 trans_speed: float, lambda_sp2: float, lambda_sp: float):
        """
        构造函数
        :param devices: 用户设备列表
        :param speed: MEC处理器的执行速度
        :param r_sp: 可卸载任务的计算需求均值
        :param r_sp_2: 可卸载任务的计算需求二阶矩
        :param d_sp: 用户设备到边缘服务器设备之间可卸载任务通信的数据量均值
        :param d_sp_2: 用户设备到边缘服务器设备之间可卸载任务通信的数据量二阶矩
        :param trans_speed: 用户设备和边缘服务器之间的通信速度
        :param lambda_sp2: 卸载到MEC处理的任务子流到达率
        :param lambda_sp: MEC上所有任务到达率
        """

        self.devices = devices
        self.speed = speed
        self.r_sp = r_sp
        self.r_sp_2 = r_sp_2
        self.d_sp = d_sp
        self.d_sp_2 = d_sp_2
        self.trans_speed = trans_speed
        self.lambda_sp2 = lambda_sp2
        self.lambda_sp = lambda_sp

        # 本地处理的任务到达率初始化
        self.arrival_rate_ge = 0.0

    #   计算sever中的Tij即服务器相应时间综合
    def cal_Tij(self):

        Tij_0 = self.r_sp / self.speed + self.d_sp / self.trans_speed

        Tij_1_up = 0
        for device in self.devices:
            Tij_1_up += device.lambda_sp2 * (device.r_sp_2 / (self.speed ** 2) + (2 * device.r_sp * self.d_sp) / (
                    self.speed * self.trans_speed) + self.d_sp_2 / (self.trans_speed ** 2))

        Tij_1_down_0 = 0

        for device in self.devices:
            Tij_1_down_0 += device.lambda_sp2 * ((device.r_sp / self.speed) + self.d_sp / self.trans_speed)

        Tij_1_down = 2 * (1 - Tij_1_down_0)

        Tij_1 = Tij_1_up / Tij_1_down

        Tij = Tij_0 + Tij_1

        return Tij
