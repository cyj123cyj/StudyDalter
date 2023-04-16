import os
import base64
import win32api

from pyDes import *


class EncryptionParameters():
    def __init__(self):
        self.Des_Key = b"DGS@DKN*"  # Key
        self.Des_IV = b"\x22\x33\x35\x81\xBC\x38\x5A\xE7"  # 自定IV向量


class Register(EncryptionParameters):
    def getCVolumeSerialNumber(self):
        CVolumeSerialNumber = win32api.GetVolumeInformation("C:\\")[1]
        if CVolumeSerialNumber:
            return str(
                CVolumeSerialNumber)  # number is long type，has to be changed to str for comparing to content after.
        else:
            return 0

    def DesDecrypt(self, str):  # des解码
        k = des(self.Des_Key, CBC, self.Des_IV, pad=None, padmode=PAD_PKCS5)
        DecryptStr = k.decrypt(str)
        return DecryptStr

    def regist(self, key):  # 获取注册码，验证成功后生成注册文件
        content = self.getCVolumeSerialNumber()  # number has been changed to str type after use str()
        if key:
            key_decrypted = bytes.decode(self.DesDecrypt(base64.b64decode(key)))
            if content != 0 and key_decrypted != 0:
                if content != key_decrypted:  # 注册码无效
                    return False
                elif content == key_decrypted:  # 验证成功
                    with open('./auth', 'w') as f:
                        f.write(key)
                        f.close()
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def checkAuthored(self):  # 检查是否有auth注册码文件，如果有自动登录
        content = self.getCVolumeSerialNumber()
        if (not (os.path.exists('./auth') and os.path.isfile('./auth'))):  # 未找到注册授权文件
            checkAuthoredResult = -10
        else:  # 读写文件要加判断
            try:
                f = open('./auth', 'r')
                if f:
                    key = f.read()
                    if key:
                        try:
                            key_decrypted = self.DesDecrypt(base64.b64decode(key))
                            key_decrypted = key_decrypted.decode("ascii")
                            if key_decrypted:
                                if key_decrypted == content:  # 注册码通过验证
                                    checkAuthoredResult = 1
                                    f.close()
                                    return checkAuthoredResult
                                else:  # 注册码错误
                                    checkAuthoredResult = -1
                            else:  # 注册码还原为机器码后为空（是否必要？）
                                checkAuthoredResult = -2
                        except:
                            checkAuthoredResult = -3
                    else:  # 无法从文件中读取注册码
                        checkAuthoredResult = -4
                else:  # 文件无法打开？
                    checkAuthoredResult = -5
                f.close()
                os.remove('./auth')
            except IOError:
                print(IOError)
                checkAuthoredResult = -6
        return checkAuthoredResult


if __name__ == '__main__':
    reg = Register()
    reg.regist('+6e3nGMXDd9y/uRsR6qI9w==')
