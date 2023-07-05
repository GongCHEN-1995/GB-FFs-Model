import re
import numpy as np

def read_GAFF2_dat(bond_morse=False, angle_ub=False):
    GAFF_dat = './data/gaff2.dat'
    openfile = open(GAFF_dat, 'r')
    data = openfile.read().splitlines()

    atom = {}
    for line in data[1:98]:
        atom[line[:2].replace(' ','')] = [float(line[3:8]), float(line[17:22])]
    
    bond = {}
    bond_name = []
    for line in data[100:1035]:
        K = float(line[7:12])
        r0 = float(line[16:22])
        if bond_morse:
            bond[line[:5].replace(' ','')] = [K/100, r0, 2]
        else:
            bond[line[:5].replace(' ','')] = [K/100, r0]
        bond_name.append(line[:5].replace(' ',''))
    for name in bond_name:
        split_name = re.split(r'-', name)
        new_name = split_name[1] + '-' + split_name[0]
        if new_name not in bond:
            bond[new_name] = bond[name]
    if bond_morse:
        bond['ha-ha'] = [100/100, 1.4494199, 2]
    else:
        bond['ha-ha'] = [100/100, 1.4494199]
        
    angle = {}
    angle_name = []
    for line in data[1036:6348]:
        angle[line[:8].replace(' ','')] = [float(line[10:16])/10, 10*float(line[22:28])/180]
        angle_name.append(line[:8].replace(' ',''))
    angle['ce-no-o'] = [86.0/10, 118.22/180*10] 
    angle['o-sy-f']  = [20.1/10, 105.57/180*10]

    for name in angle_name:
        split_name = re.split(r'-', name)
        new_name = split_name[2] + '-' + split_name[1] + '-' + split_name[0]
        if new_name not in angle:
            angle[new_name] = angle[name]
    
    # overwrite the paras for same periodicity(PN)
    torsion = {}
    torsion_name = []
    for line in data[6349:7476]:
        name = line[:11].replace(' ','')
        if name not in torsion:
            torsion[name] = [[float(line[14:15]), float(line[18:24]), float(line[31:38]), float(line[48:54])]]
        else:
            torsion[name].append([float(line[14:15]), float(line[18:24]), float(line[31:38]), float(line[48:54])])
    for name in torsion:
        torsion_para = [0., 0., 0., 0.]
        for all_para in torsion[name]:
            para2 = -1 if all_para[2] == 180. else 1
            para1 = all_para[1] / all_para[0] * para2
            PN = int(abs(all_para[3]))
            torsion_para[PN-1] = para1
        torsion[name] = torsion_para
    for name in torsion_name:
        split_name = re.split(r'-', name)
        new_name = split_name[3] + '-' + split_name[2] + '-' + split_name[1] + '-' + split_name[0]
        if new_name not in torsion:
            torsion[new_name] = torsion[name]
    
    imptors = {}
    for line in data[7477:7515]:
        imptors[line[:11].replace(' ','')] = [float(line[20:24])]
    
    vdw = {}
    for line in data[7520:7615]:
        vdw[line[2:4].replace(' ','')] = [float(line[14:20]), 10 * np.sqrt(float(line[22:28]))]
    vdw['hw'] = vdw['ho']
    vdw['ow'] = vdw['oh']
    
    if angle_ub:
        for name in angle:
            split_name = re.split(r'-',name)
            aa = vdw[split_name[0]]
            bb = vdw[split_name[2]]
            sigma = aa[0] + bb[0]
            angle[name] = angle[name] + [sigma, 1]

    openfile.close()
    GAFF = {'atom':atom,'vdw':vdw,'bond':bond,'angle':angle,'torsion':torsion,'imptors':imptors}
    return GAFF
