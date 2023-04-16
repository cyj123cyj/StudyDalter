#coding:utf8
#register.py
#功能说明：用户运行程序后，自动检测认证状态，如果未经认证，就需要注册。注册过程是用户将程序运行后显示的机器码（卷序号）发回给管理员，管理员通过加密后生成加密文件或字符串给回用户。
#每次登录，在有注册文件或者注册码的情况下，软件就会通过DES和base64解码，如果解码后和重新获取的机器码一致，则通过认证，进入主程序。
import sys

import os
import base64
import win32api
from pyDes import *
#from binascii import a2b_hex    #如果需要用二进制编码保存注册码和注册文件可以使用binascii转换

class register():
    def __init__(self):
        #self.Des_Key = "BHC#@*UM" # Key
        self.Des_Key = "DGS@DKN*" # Key
        self.Des_IV = "\x22\x33\x35\x81\xBC\x38\x5A\xE7" # 自定IV向量

    #获取C盘卷序列号
    #使用C盘卷序列号的优点是长度短，方便操作，比如1513085707，但是对C盘进行格式化或重装电脑等操作会影响C盘卷序列号。
    #win32api.GetVolumeInformation(Volume Name, Volume Serial Number, Maximum Component Length of a file name, Sys Flags, File System Name)
    #return('', 1513085707, 255, 65470719, 'NTFS'),volume serial number is  1513085707.
    def getCVolumeSerialNumber(self):
        CVolumeSerialNumber=win32api.GetVolumeInformation("C:\\")[1]
        #print chardet.detect(str(CVolumeSerialNumber))
        #print CVolumeSerialNumber
        if CVolumeSerialNumber:
            return str(CVolumeSerialNumber) #number is long type，has to be changed to str for comparing to content after.
        else:
            return 0
    """
    #使用DES加base64的形式加密
    #考虑过使用M2Crypto和rsa，但是都因为在windows环境中糟糕的安装配置过程而放弃
    def DesEncrypt(self,str):
        k = des(self.Des_Key, CBC, self.Des_IV, pad=None, padmode=PAD_PKCS5)
        EncryptStr = k.encrypt(str)
        #EncryptStr = binascii.unhexlify(k.encrypt(str))
        return base64.b64encode(EncryptStr) #转base64编码返回
    """    
    #des解码
    def DesDecrypt(self,str):
        k = des(self.Des_Key, CBC, self.Des_IV, pad=None, padmode=PAD_PKCS5)
        DecryptStr = k.decrypt(str)
        #DecryptStr = a2b_hex(k.decrypt(str))
        #print DecryptStr
        return DecryptStr
    
    #获取注册码，验证成功后生成注册文件
    def regist(self,key):
        content = self.getCVolumeSerialNumber()    #number has been changed to str type after use str() 
        print('Machine Code: ' + content)
        #print('机器码：' + content)
        #print(content)
        print('说明：请将此机器码提交给软件提供方，以获取授权注册码并输入：')
        #key = raw_input('Registration Code: ')
        #由于输入类似“12”这种不符合base64规则的字符串会引起异常，所以需要增加输入判断
        #while key
        if key:
            #print chardet.detect(content)
            #print type(content)
            #print content
            #type(key_decrypted) is str
            try:
                key_decrypted=str(self.DesDecrypt(base64.b64decode(key)))
            #except ValueError, TypeError:
            except:
                #print('注册码输入错误！')
                return False
                
            #print chardet.detect(key_decrypted)
            #print key_decrypted
            #type(key_decrypted) is str
            if content!=0 and key_decrypted!=0:
                if content != key_decrypted:
                    print "注册码无效！请重新输入。"
                    #self.regist()
                    return False
                elif content==key_decrypted:
                    #print "软件注册成功！"
                    #读写文件要加判断
                    with open('./auth','w') as f:
                        f.write(key)
                        f.close()
                    return True
                else:
                    return False
            else:
                return False
        #else:
        #    self.regist()
        return False

    def checkAuthored(self):
        content=self.getCVolumeSerialNumber()
        checkAuthoredResult = 0
        if (not (os.path.exists('./auth') and os.path.isfile('./auth')) ):
            #未找到注册授权文件
            checkAuthoredResult = -10
        
        #读写文件要加判断
        else: 
            try:
                f=open('./auth','r')
                if f:
                    key=f.read()
                    if key:
                        try:
                            key_decrypted=self.DesDecrypt(base64.b64decode(key))
                            if key_decrypted:
                                if key_decrypted == content: #注册码通过验证
                                    checkAuthoredResult = 1
                                    f.close()
                                    return checkAuthoredResult
                                else: #注册码错误
                                    checkAuthoredResult = -1
                            else: #注册码还原为机器码后为空（是否必要？）
                                checkAuthoredResult = -2
                        #无法将注册码还原为机器码（注册码位数、格式等错误）
                        #except ValueError, TypeError:
                        except:
                            #print('软件授权无效！')
                            checkAuthoredResult = -3
                    else: #无法从文件中读取注册码
                        checkAuthoredResult = -4
                else: #文件无法打开？
                    #print('软件授权无效！')
                    #self.regist()
                    checkAuthoredResult = -5
                f.close()
                os.remove('./auth')
            except IOError:
                print IOError
                checkAuthoredResult = -6
        #print checkAuthoredResult
        return checkAuthoredResult


if __name__ == '__main__':
    reg=register()
    #reg.regist(key1)
