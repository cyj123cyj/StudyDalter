# -*- coding: utf-8 -*-
import os
import re
import pydicom as dicom


# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_id(path):
    f = dicom.read_file(path, stop_before_pixels=True)
    return f.StudyInstanceUID, f.SeriesInstanceUID


def is_dicom_file(path):
    """Fast way to check whether file is DICOM."""
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as f:
            return f.read(132).decode("ASCII")[-4:] == "DICM"
    except:
        return False


def alphanum_key(s):
    return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', s)]


def dicom_files_in_dir(directory="."):
    """Full paths of all DICOM files in the directory."""
    directory = os.path.expanduser(directory)
    fls = [os.path.join(directory, f) for f in os.listdir(directory)]
    inf = []
    for fname in fls:
        if is_dicom_file(fname):
            ds = dicom.read_file(fname, stop_before_pixels=True)
            try:
                inf.append(
                    {'fname': fname, 'InstanceNumber': int(ds.InstanceNumber), 'SeriesNumber': int(ds.SeriesNumber),
                     'StudyTime': float(ds.StudyTime),
                     'StudyDate': int(ds.StudyDate), 'SeriesDescription': ds.SeriesDescription})
            except:
                inf.append(
                    {'fname': fname, 'InstanceNumber': int(ds.InstanceNumber), 'SeriesNumber': int(ds.SeriesNumber),
                     'StudyTime': int(ds.StudyTime),
                     'StudyDate': int(ds.StudyDate), 'SeriesDescription': 'xx'})

    l = sorted(inf, key=lambda x: (
        x['StudyDate'], x['StudyTime'], x['SeriesNumber'], x['SeriesDescription'],
        x['InstanceNumber']))  # ,x['SeriesNumber']x['SeriesTime'],
    flist = [i['fname'] for i in l]
    return flist  # [f for f in candidates if is_dicom_file(f)]


if __name__ == '__main__':
    # path = "D:\\医学模体图像\\儿童颅骨CT\\SE1_12days"
    path = "D:\\医学模体图像\\儿童模数据12月\\test3.0系列-4.0-4.0"
    fls = dicom_files_in_dir(path)
    print(fls)