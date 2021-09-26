from typing import List


class UserDevice:
    """
    用户设备实体类，被视为M/G/1排队模型，排队过则为非抢占优先
    """

    def __init__(self, edgeServerList: List, lambda_i: float, speed: float, r_sp: float, r_sp_2: float, r_ge: float, r_ge_2: float,
                 lambda_sp1: float, lambda_sp2: float, lambda_ge: float):
        """
        构造函数
        :param edgeServerList: 用户设备连接到的服务器列表（多个）
        :param speed: 用户设备执行速度
        :param r_sp:可卸载任务的计算需求均值
        :param r_sp_2可卸载任务的计算需求二阶矩
        :param r_ge:不可卸载任务的计算需求均值
        :param r_ge_2:不可卸载任务的计算需求二阶矩
        :param lambda_sp1:在本地处理的可卸载任务子流到达率
        :param lambda_sp2:卸载到MEC处理的可卸载任务子流到达率
        :param lambda_ge:不可卸载任务到达率
        ##############################
        :param xi:用户设备上所有任务的执行时间
        :param ti:用户设备上所有任务平均响应时间
        """
        self.edgeServerList = edgeServerList
        self.speed = speed
        self.r_sp = r_sp
        self.r_sp_2 = r_sp_2
        self.r_ge = r_ge
        self.r_ge_2 = r_ge_2
        self.lambda_ge = lambda_ge
        self.lambda_sp1 = lambda_sp1
        self.lambda_sp2 = lambda_sp2
        self.lambda_i = lambda_i

    #  计算本地设备的Ti0
    def cal_Ti0(self):
        Ti0 = self.lambda_ge / (self.lambda_ge + self.lambda_sp1) * (self.r_ge / self.speed) \
                   + self.lambda_sp1 / (self.lambda_ge + self.lambda_sp1) * (self.r_sp / self.speed) \
                   + (self.lambda_ge * (self.r_ge_2 / (self.speed ** 2)) + self.lambda_sp1 * (
                    self.r_sp_2 / (self.speed ** 2))) / \
                   (2 * (1 - (self.lambda_ge * (self.r_ge / self.speed) + self.lambda_sp1 * (self.r_sp / self.speed))))

        return Ti0


    def cal_Ti(self):

        Ti_0 = (self.lambda_ge + self.lambda_sp1)/(self.lambda_ge + self.lambda_i) * self.cal_ti0()

        Ti_1 = 0
        for j in self.edgeServerList:
            Ti_1 += (self.lambda_sp2/(self.lambda_ge + self.lambda_i)) * j.cal_Tij()

        Ti = Ti_0 + Ti_1

        return Ti









