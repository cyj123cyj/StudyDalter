'''coding:utf8
encrypter.py
功能说明：自动检测认证状态，未经认证需要注册。注册过程是用户将程序运行后显示的机器码（卷序号）发回给管理员，管理员通过加密后给回用户。
每次登录，在有注册文件或者注册码的情况下，软件就会通过DES和base64解码，如果解码后和重新获取的机器码一致，则通过认证，进入主程序。
'''

import base64

from pyDes import *
from viewers.register import EncryptionParameters


class Encrypter(EncryptionParameters):
    def DesEncrypt(self, str):  # 加密函数，使用DES加密并使用base64编码
        k = des(self.Des_Key, CBC, self.Des_IV, pad=None, padmode=PAD_PKCS5)
        EncryptStr = k.encrypt(str)
        return base64.b64encode(EncryptStr)

    # def encrypt(self, code):  # 获取注册码，验证成功后生成注册文件
    #     '''不太清楚是什么逻辑'''
    #     if code:
    #         key = str(self.DesEncrypt(code))
    #         print('注册码为：')
    #         print(key)
