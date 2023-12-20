import numpy as np
import copy
import time

import torch
import torch.nn as nn

def Test_GAFF(model, model2, dl, target_type='single'):
    assert target_type in ['single', 'sum']
    model.eval()
    Params_Loss = np.zeros(6)
    Energy_Loss = np.zeros(6)
    loss_fn = nn.MSELoss()
    loss_bce = nn.BCELoss()
    with torch.no_grad():
        for p, data in enumerate(dl):
            inputs, targets, coord_inputs, params_index = copy.deepcopy(data)
            bsz = len(params_index['atom']) - 1
            if torch.cuda.is_available():
                for name in inputs.keys():
                    inputs[name] = inputs[name].to('cuda')
                for name in targets.keys():
                    targets[name] = targets[name].to('cuda')
                for i in range(len(coord_inputs)):
                    for name in coord_inputs[i].keys():
                        coord_inputs[i][name] = coord_inputs[i][name].to('cuda')
    
            # forward + backward + optimize
            parameters = model(inputs) 
            Params_Loss[0] += loss_fn(parameters['vdw'],targets['vdw']).item()
            Params_Loss[1] += loss_fn(parameters['bond'],targets['bond']).item()
            Params_Loss[2] += loss_fn(parameters['angle'],targets['angle']).item()
            if not np.isnan(loss_fn(parameters['imptors'],targets['imptors']).item()):
                Params_Loss[3] += loss_fn(parameters['imptors'],targets['imptors']).item()
            Params_Loss[4] += loss_fn(parameters['torsion'],targets['torsion']).item()
            Params_Loss[5] += loss_fn(parameters['charge'],targets['charge']).item()

            for i in range(len(params_index['atom'])-1):
                mol_parameters = {}
                mol_targets = {}
                mol_parameters['vdw'] = parameters['vdw'][params_index['atom'][i]:params_index['atom'][i+1]]
                mol_parameters['bond'] = parameters['bond'][params_index['bond'][i]:params_index['bond'][i+1]]
                mol_parameters['angle'] = parameters['angle'][params_index['angle'][i]:params_index['angle'][i+1]]
                mol_parameters['imptors'] = parameters['imptors'][params_index['imptors'][i]:params_index['imptors'][i+1]]
                mol_parameters['torsion'] = parameters['torsion'][params_index['torsion'][i]:params_index['torsion'][i+1]]
                mol_parameters['charge'] = parameters['charge'][params_index['atom'][i]:params_index['atom'][i+1]]
                mol_targets['vdw'] = targets['vdw'][params_index['atom'][i]:params_index['atom'][i+1]]
                mol_targets['bond'] = targets['bond'][params_index['bond'][i]:params_index['bond'][i+1]]
                mol_targets['angle'] = targets['angle'][params_index['angle'][i]:params_index['angle'][i+1]]
                mol_targets['imptors'] = targets['imptors'][params_index['imptors'][i]:params_index['imptors'][i+1]]
                mol_targets['torsion'] = targets['torsion'][params_index['torsion'][i]:params_index['torsion'][i+1]]
                mol_targets['charge'] = targets['charge'][params_index['atom'][i]:params_index['atom'][i+1]]
                
                Energy_NN, _ = model2(coord_inputs[i], mol_parameters)
                Energy_GAFF, _ = model2(coord_inputs[i], mol_targets)
                if target_type == 'sum':
                    for name in Energy_NN:
                        Energy_NN[name] = torch.sum(Energy_NN[name],axis=1)
                        Energy_GAFF[name] = torch.sum(Energy_GAFF[name],axis=1)
                        energy_cri = 1000
                else:
                    energy_cri = 100
                mask_vdw = torch.abs(Energy_GAFF['vdw']) > energy_cri
                Energy_GAFF['vdw'] = Energy_GAFF['vdw'].masked_fill(mask_vdw,0)
                Energy_NN['vdw'] = Energy_NN['vdw'].masked_fill(mask_vdw,0)
                mask_bond = torch.abs(Energy_GAFF['bond']) > energy_cri
                Energy_GAFF['bond'] = Energy_GAFF['bond'].masked_fill(mask_bond,0)
                Energy_NN['bond'] = Energy_NN['bond'].masked_fill(mask_bond,0)
                mask_angle = torch.abs(Energy_GAFF['angle']) > energy_cri
                Energy_GAFF['angle'] = Energy_GAFF['angle'].masked_fill(mask_angle,0)
                Energy_NN['angle'] = Energy_NN['angle'].masked_fill(mask_angle,0)
                mask_charge = torch.abs(Energy_GAFF['charge']) > energy_cri
                Energy_GAFF['charge'] = Energy_GAFF['charge'].masked_fill(mask_charge,0)
                Energy_NN['charge'] = Energy_NN['charge'].masked_fill(mask_charge,0)
                
                tmp = loss_fn(Energy_NN['vdw'],Energy_GAFF['vdw']).item() / bsz
                Energy_Loss[0] += 0 if np.isnan(tmp) else tmp
                tmp = loss_fn(Energy_NN['bond'],Energy_GAFF['bond']).item() / bsz
                Energy_Loss[1] += 0 if np.isnan(tmp) else tmp
                tmp = loss_fn(Energy_NN['angle'],Energy_GAFF['angle']).item() / bsz
                Energy_Loss[2] += 0 if np.isnan(tmp) else tmp
                tmp = loss_fn(Energy_NN['imptors'],Energy_GAFF['imptors']).item() / bsz
                Energy_Loss[3] += 0 if np.isnan(tmp) else tmp
                tmp = loss_fn(Energy_NN['torsion'],Energy_GAFF['torsion']).item() / bsz
                Energy_Loss[4] += 0 if np.isnan(tmp) else tmp
                tmp = loss_fn(Energy_NN['charge'],Energy_GAFF['charge']).item() / bsz
                Energy_Loss[5] += 0 if np.isnan(tmp) else tmp
    return Params_Loss/len(dl), Energy_Loss/len(dl)
        
def Train_para(model, model2, train_dl, val_dl, test_dl, epochs=1, start_lr=1e-6, end_lr=0.1, rate=0.99, step_size=1):
    loss_fn = nn.MSELoss()
    loss_bce = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=rate)
    best_score = 1e20
    bool_scheduler_step = True
    
    for epoch in range(1,epochs+1):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = np.zeros(6)
        for data in train_dl:
            inputs, targets, _, _ = copy.deepcopy(data)
            if torch.cuda.is_available():
                for name in inputs.keys():
                    inputs[name] = inputs[name].to('cuda')
                for name in targets.keys():
                    targets[name] = targets[name].to('cuda')
                          
            # forward + backward + optimize
            parameters = model(inputs)
            loss1 = loss_fn(parameters['vdw'],targets['vdw'])
            loss2 = loss_fn(parameters['bond'],targets['bond'])
            loss3 = loss_fn(parameters['angle'],targets['angle'])
            loss4 = loss_fn(parameters['imptors'],targets['imptors'])
            loss5 = loss_fn(parameters['torsion'],targets['torsion'])
            loss6 = loss_fn(parameters['charge'],targets['charge'])
            
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 
            epoch_loss += np.array([loss1.item(), loss2.item(), loss3.item(), 0, loss5.item(), loss6.item()])
            if not np.isnan(loss4.item()):
                epoch_loss[3] += loss4.item()
            loss.backward()
            optimizer.step()
#             zero the parameter gradients
            optimizer.zero_grad()
            
            if rate > 1 and optimizer.state_dict()['param_groups'][0]['lr'] > end_lr:
                bool_scheduler_step = False
            elif rate < 1 and optimizer.state_dict()['param_groups'][0]['lr'] < end_lr:
                bool_scheduler_step = False
            if bool_scheduler_step:
                scheduler.step()
                    
        epoch_loss /= len(train_dl)
        val_params_loss, _ = Test_GAFF(model, model2, val_dl)
        test_params_loss, _ = Test_GAFF(model, model2, test_dl)

        elapsed = time.time() - epoch_start_time
        print('| {:3d}/{:3d} | lr {:.1e} | {:2.0f} s |'.format(
                epoch, epochs, optimizer.state_dict()['param_groups'][0]['lr'], elapsed))
        print('RMSE for parameters (VdW/Bond/Angle/Imptors/Torsion/Charge):')
        print('Train params: {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e}'.format(epoch_loss[0],epoch_loss[1],epoch_loss[2],epoch_loss[3],epoch_loss[4],epoch_loss[5]))
        print('Val   params: {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e}'.format(val_params_loss[0],val_params_loss[1],val_params_loss[2],val_params_loss[3],val_params_loss[4],val_params_loss[5]))
        print('Test  params: {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} \n'.format(test_params_loss[0],test_params_loss[1],test_params_loss[2],test_params_loss[3],test_params_loss[4],test_params_loss[5]))
        
    print('The Training is finished')
    model.eval()
    return model

def Train_GAFF(model, model2, train_dl, val_dl, test_dl, epochs=1, start_lr=1e-6, end_lr=0.1, rate=0.99, step_size=1, target_type='single'):
    assert target_type in ['single', 'sum']
    loss_fn = nn.MSELoss()
    loss_bce = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=rate)
    best_score = 1e20
    bool_scheduler_step = True
    
    for epoch in range(1,epochs+1):
        epoch_start_time = time.time()
        model.train()
        Params_Loss = np.zeros(6)
        Energy_Loss = np.zeros(6)
        for p, data in enumerate(train_dl):
            inputs, targets, coord_inputs, params_index = copy.deepcopy(data)
            bsz = len(params_index['atom']) - 1
            if torch.cuda.is_available():
                for name in inputs.keys():
                    inputs[name] = inputs[name].to('cuda')
                for name in targets.keys():
                    targets[name] = targets[name].to('cuda')
                for i in range(len(coord_inputs)):
                    for name in coord_inputs[i].keys():
                        coord_inputs[i][name] = coord_inputs[i][name].to('cuda')

            # forward + backward + optimize
            parameters = model(inputs)
            loss1 = loss_fn(parameters['vdw'],targets['vdw'])
            loss2 = loss_fn(parameters['bond'],targets['bond'])
            loss3 = loss_fn(parameters['angle'],targets['angle'])
            loss4 = loss_fn(parameters['imptors'],targets['imptors'])
            loss5 = loss_fn(parameters['torsion'],targets['torsion'])
            loss6 = loss_fn(parameters['charge'],targets['charge'])

            Params_Loss += np.array([loss1.item(), loss2.item(), loss3.item(), 0, loss5.item(), loss6.item()])
            if not np.isnan(loss4.item()):
                Params_Loss[3] += loss4.item()
                
            loss7 = 0
            loss8 = 0
            loss9 = 0
            loss10 = 0
            loss11 = 0
            loss12 = 0
            for i in range(len(params_index['atom'])-1):
                mol_parameters = {}
                mol_targets = {}
                mol_parameters['vdw'] = parameters['vdw'][params_index['atom'][i]:params_index['atom'][i+1]]
                mol_parameters['bond'] = parameters['bond'][params_index['bond'][i]:params_index['bond'][i+1]]
                mol_parameters['angle'] = parameters['angle'][params_index['angle'][i]:params_index['angle'][i+1]]
                mol_parameters['imptors'] = parameters['imptors'][params_index['imptors'][i]:params_index['imptors'][i+1]]
                mol_parameters['torsion'] = parameters['torsion'][params_index['torsion'][i]:params_index['torsion'][i+1]]
                mol_parameters['charge'] = parameters['charge'][params_index['atom'][i]:params_index['atom'][i+1]]
                mol_targets['vdw'] = targets['vdw'][params_index['atom'][i]:params_index['atom'][i+1]]
                mol_targets['bond'] = targets['bond'][params_index['bond'][i]:params_index['bond'][i+1]]
                mol_targets['angle'] = targets['angle'][params_index['angle'][i]:params_index['angle'][i+1]]
                mol_targets['imptors'] = targets['imptors'][params_index['imptors'][i]:params_index['imptors'][i+1]]
                mol_targets['torsion'] = targets['torsion'][params_index['torsion'][i]:params_index['torsion'][i+1]]
                mol_targets['charge'] = targets['charge'][params_index['atom'][i]:params_index['atom'][i+1]]
                
                Energy_NN, _ = model2(coord_inputs[i], mol_parameters)
                Energy_GAFF, _ = model2(coord_inputs[i], mol_targets)
                if target_type == 'sum':
                    for name in Energy_NN:
                        Energy_NN[name] = torch.sum(Energy_NN[name],axis=1)
                        Energy_GAFF[name] = torch.sum(Energy_GAFF[name],axis=1)
                        energy_cri = 1000
                else:
                    energy_cri = 100
                mask_vdw = torch.abs(Energy_GAFF['vdw']) > energy_cri
                Energy_GAFF['vdw'] = Energy_GAFF['vdw'].masked_fill(mask_vdw,0)
                Energy_NN['vdw'] = Energy_NN['vdw'].masked_fill(mask_vdw,0)
                mask_bond = torch.abs(Energy_GAFF['bond']) > energy_cri
                Energy_GAFF['bond'] = Energy_GAFF['bond'].masked_fill(mask_bond,0)
                Energy_NN['bond'] = Energy_NN['bond'].masked_fill(mask_bond,0)
                mask_angle = torch.abs(Energy_GAFF['angle']) > energy_cri
                Energy_GAFF['angle'] = Energy_GAFF['angle'].masked_fill(mask_angle,0)
                Energy_NN['angle'] = Energy_NN['angle'].masked_fill(mask_angle,0)
                mask_charge = torch.abs(Energy_GAFF['charge']) > energy_cri
                Energy_GAFF['charge'] = Energy_GAFF['charge'].masked_fill(mask_charge,0)
                Energy_NN['charge'] = Energy_NN['charge'].masked_fill(mask_charge,0)

                loss7_tmp = loss_fn(Energy_NN['vdw'],Energy_GAFF['vdw'])
                if not np.isnan(loss7_tmp.item()):
                    loss7 += loss7_tmp / bsz
                loss8_tmp = loss_fn(Energy_NN['bond'],Energy_GAFF['bond'])
                if not np.isnan(loss8_tmp.item()):
                    loss8 += loss8_tmp / bsz
                loss9_tmp = loss_fn(Energy_NN['angle'],Energy_GAFF['angle'])
                if not np.isnan(loss9_tmp.item()):
                    loss9 += loss9_tmp / bsz
 
                loss10 += loss_fn(Energy_NN['imptors'],Energy_GAFF['imptors'])
    
                loss11_tmp = loss_fn(Energy_NN['torsion'],Energy_GAFF['torsion'])
                if not np.isnan(loss11_tmp.item()):
                    loss11 += loss11_tmp / bsz
                loss12_tmp = loss_fn(Energy_NN['charge'],Energy_GAFF['charge'])
                if not np.isnan(loss12_tmp.item()):
                    loss12 += loss12_tmp / bsz

            Energy_Loss += np.array([loss7.item(), loss8.item(), loss9.item(), 0, loss11.item(), loss12.item()])
            if not np.isnan(loss10.item()):
                Energy_Loss[3] += loss10.item()
            loss = loss1 + loss2 + loss3 + loss4 + loss5  + loss6  + loss7 + loss8 + loss9 + loss10 + loss11 + loss12

            loss.backward()
            optimizer.step()
#             zero the parameter gradients
            optimizer.zero_grad()

            if rate > 1 and optimizer.state_dict()['param_groups'][0]['lr'] > end_lr:
                bool_scheduler_step = False
            elif rate < 1 and optimizer.state_dict()['param_groups'][0]['lr'] < end_lr:
                bool_scheduler_step = False
            if bool_scheduler_step:
                scheduler.step()
                
        Params_Loss /= len(train_dl)
        Energy_Loss /= len(train_dl)

        val_params_loss, val_energy_loss = Test_GAFF(model, model2, val_dl, target_type=target_type)
        test_params_loss, test_energy_loss = Test_GAFF(model, model2, test_dl, target_type=target_type)

        elapsed = time.time() - epoch_start_time
        print('| {:3d}/{:3d} | lr {:.1e} | {:2.0f} s |'.format(
                epoch, epochs, optimizer.state_dict()['param_groups'][0]['lr'], elapsed))
        
        print('RMSE for parameters (VdW/Bond/Angle/Imptors/Torsion/Charge):')
        print('Train params: {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e}'.format(Params_Loss[0],Params_Loss[1],Params_Loss[2],Params_Loss[3],Params_Loss[4],Params_Loss[5]))
        print('Val   params: {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e}'.format(val_params_loss[0],val_params_loss[1],val_params_loss[2],val_params_loss[3],val_params_loss[4],val_params_loss[5]))
        print('Test  params: {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e}'.format(test_params_loss[0],test_params_loss[1],test_params_loss[2],test_params_loss[3],test_params_loss[4],test_params_loss[5]))
        print('RMSE for Energy (VdW/Bond/Angle/Imptors/Torsion/Charge):')
        print('Train params: {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e}'.format(Energy_Loss[0],Energy_Loss[1],Energy_Loss[2],Energy_Loss[3],Energy_Loss[4],Energy_Loss[5]))
        print('Val   params: {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e}'.format(val_energy_loss[0],val_energy_loss[1],val_energy_loss[2],val_energy_loss[3],val_energy_loss[4],val_energy_loss[5]))
        print('Test  params: {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} \n'.format(test_energy_loss[0],test_energy_loss[1],test_energy_loss[2],test_energy_loss[3],test_energy_loss[4],test_energy_loss[5]))

    print('The Training is finished')
    model.eval()
    return model

def Test_ANI1(model, model2, dl):
    model.eval()
    Loss = np.zeros(7)
    loss_fn = nn.MSELoss()
    loss_fn1 = nn.L1Loss()
    num_conf = 0
    num_mol = 0
    with torch.no_grad():
        for p,data in enumerate(dl):
            inputs, targets, coord_inputs, params_index = copy.deepcopy(data)
            bsz = len(params_index['atom']) - 1
            if torch.cuda.is_available():
                for name in inputs.keys():
                    inputs[name] = inputs[name].to('cuda')
                for name in targets.keys():
                    targets[name] = targets[name].to('cuda')
                for i in range(len(coord_inputs)):
                    for name in coord_inputs[i].keys():
                        coord_inputs[i][name] = coord_inputs[i][name].to('cuda')
    
            # forward + backward + optimize
            parameters = model(inputs) 
            Loss[0] += loss_fn(parameters['vdw'],targets['vdw']).item()
            Loss[1] += loss_fn(parameters['bond'],targets['bond']).item()
            Loss[2] += loss_fn(parameters['angle'],targets['angle']).item()
            if not np.isnan(loss_fn(parameters['imptors'],targets['imptors']).item()):
                Loss[3] += loss_fn(parameters['imptors'],targets['imptors']).item()
            Loss[4] += loss_fn(parameters['torsion'],targets['torsion']).item()
            
            for i in range(len(params_index['atom'])-1):
                mol_parameters = {}
                mol_targets = {}
                mol_parameters['vdw'] = parameters['vdw'][params_index['atom'][i]:params_index['atom'][i+1]]
                mol_parameters['bond'] = parameters['bond'][params_index['bond'][i]:params_index['bond'][i+1]]
                mol_parameters['angle'] = parameters['angle'][params_index['angle'][i]:params_index['angle'][i+1]]
                mol_parameters['imptors'] = parameters['imptors'][params_index['imptors'][i]:params_index['imptors'][i+1]]
                mol_parameters['torsion'] = parameters['torsion'][params_index['torsion'][i]:params_index['torsion'][i+1]]
                mol_parameters['charge'] = parameters['charge'][params_index['atom'][i]:params_index['atom'][i+1]]
                mol_targets['vdw'] = targets['vdw'][params_index['atom'][i]:params_index['atom'][i+1]]
                mol_targets['bond'] = targets['bond'][params_index['bond'][i]:params_index['bond'][i+1]]
                mol_targets['angle'] = targets['angle'][params_index['angle'][i]:params_index['angle'][i+1]]
                mol_targets['imptors'] = targets['imptors'][params_index['imptors'][i]:params_index['imptors'][i+1]]
                mol_targets['torsion'] = targets['torsion'][params_index['torsion'][i]:params_index['torsion'][i+1]]
                mol_targets['charge'] = targets['charge'][params_index['atom'][i]:params_index['atom'][i+1]]
                Loss[5] += loss_fn(mol_parameters['charge'],mol_targets['charge']).item() / bsz

                mol_parameters['charge'] = mol_targets['charge']
                                
                Energy_NN, _ = model2(coord_inputs[i], mol_parameters)
                Energy_GAFF, _ = model2(coord_inputs[i], mol_targets)
                
                mask_vdw = torch.sum(Energy_NN['vdw'] > 50, dim=1) > 0
                mask_bond = torch.sum(Energy_NN['bond'] > 100, dim=1) > 0
                mask_angle = torch.sum(Energy_NN['angle'] > 100, dim=1) > 0
                
                for name in Energy_NN:
                    Energy_NN[name] = torch.sum(Energy_NN[name],axis=1)
                    Energy_GAFF[name] = torch.sum(Energy_GAFF[name],axis=1)
                Energy_ref = targets['energy'][params_index['energy'][i]:params_index['energy'][i+1]]
                
                Energy_NN = Energy_NN['vdw'] + Energy_NN['bond'] + Energy_NN['angle'] + Energy_NN['UB'] + Energy_NN['imptors'] + Energy_NN['torsion'] + Energy_NN['charge']
                Energy_GAFF = Energy_GAFF['vdw'] + Energy_GAFF['bond'] + Energy_GAFF['angle'] + Energy_GAFF['UB'] + Energy_GAFF['imptors'] + Energy_GAFF['torsion'] + Energy_GAFF['charge']
    
                mask_energy = mask_vdw + mask_bond + mask_angle
                if 0 in mask_energy: 
                    Energy_NN = Energy_NN[~mask_energy]
                    Energy_GAFF = Energy_GAFF[~mask_energy]
                    Energy_ref = Energy_ref[~mask_energy]
                    minimun_energy = torch.argmin(Energy_NN)
                    Energy_NN = Energy_NN - Energy_NN[minimun_energy]
                    Energy_GAFF = Energy_GAFF - Energy_GAFF[minimun_energy]
                    Energy_ref = Energy_ref - Energy_ref[minimun_energy]
                    num_conf += len(Energy_ref) + 1
                    num_mol += 1
                    Loss[-1] += np.sqrt(loss_fn(Energy_NN,Energy_ref).item())

    Loss[:-1] /= len(dl)
    Loss[-1] = Loss[-1]/num_mol
    return Loss, num_conf, num_mol
        
def Train_ANI1(model, model2, train_dl, val_dl, test_dl, epochs=1, start_lr=1e-6, end_lr=0.1, rate=0.99, step_size=1, store_name=None):
    best_score = [0, 1000, 0]
    best_model_index = 0
    loss_fn = nn.MSELoss()
    loss_bce = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=rate)
    bool_scheduler_step = True
    
    for epoch in range(1,epochs+1):
        epoch_start_time = time.time()
        model.train()
        epoch_conf = 0
        epoch_mol = 0
        train_Loss = np.zeros(7)
        for p,data in enumerate(train_dl):
            inputs, targets, coord_inputs, params_index = copy.deepcopy(data)
            bsz = len(params_index['atom']) - 1
            if torch.cuda.is_available():
                for name in inputs.keys():
                    inputs[name] = inputs[name].to('cuda')
                for name in targets.keys():
                    targets[name] = targets[name].to('cuda')
                for i in range(len(coord_inputs)):
                    for name in coord_inputs[i].keys():
                        coord_inputs[i][name] = coord_inputs[i][name].to('cuda')

            # forward + backward + optimize
            parameters = model(inputs)
            loss1 = loss_fn(parameters['vdw'],targets['vdw'])
            loss2 = loss_fn(parameters['bond'],targets['bond'])
            loss3 = loss_fn(parameters['angle'],targets['angle'])
            loss4 = loss_fn(parameters['imptors'],targets['imptors'])
            loss5 = loss_fn(parameters['torsion'],targets['torsion']) + 0.01*torch.norm(parameters['torsion'], 2)
            loss6 = 0.01*torch.norm(parameters['transfer_charge'], 2)
     
            train_Loss += np.array([loss1.item(), loss2.item(), loss3.item(), 0, loss5.item(), 0, 0])
            if not np.isnan(loss4.item()):
                train_Loss[3] += loss4.item()
            loss7 = 0
            bsz_mol = 0
            for i in range(len(params_index['atom'])-1):
                mol_parameters = {}
                mol_targets = {}
                mol_parameters['vdw'] = parameters['vdw'][params_index['atom'][i]:params_index['atom'][i+1]]
                mol_parameters['bond'] = parameters['bond'][params_index['bond'][i]:params_index['bond'][i+1]]
                mol_parameters['angle'] = parameters['angle'][params_index['angle'][i]:params_index['angle'][i+1]]
                mol_parameters['imptors'] = parameters['imptors'][params_index['imptors'][i]:params_index['imptors'][i+1]]
                mol_parameters['torsion'] = parameters['torsion'][params_index['torsion'][i]:params_index['torsion'][i+1]]
                mol_parameters['charge'] = parameters['charge'][params_index['atom'][i]:params_index['atom'][i+1]]
                mol_targets['vdw'] = targets['vdw'][params_index['atom'][i]:params_index['atom'][i+1]]
                mol_targets['bond'] = targets['bond'][params_index['bond'][i]:params_index['bond'][i+1]]
                mol_targets['angle'] = targets['angle'][params_index['angle'][i]:params_index['angle'][i+1]]
                mol_targets['imptors'] = targets['imptors'][params_index['imptors'][i]:params_index['imptors'][i+1]]
                mol_targets['torsion'] = targets['torsion'][params_index['torsion'][i]:params_index['torsion'][i+1]]
                mol_targets['charge'] = targets['charge'][params_index['atom'][i]:params_index['atom'][i+1]]
                loss6 += loss_fn(mol_parameters['charge'],mol_targets['charge']) / bsz
                
                mol_parameters['charge'] = mol_targets['charge']
                
                Energy_NN, _ = model2(coord_inputs[i], mol_parameters)

                mask_vdw = torch.sum(Energy_NN['vdw'] > 50, dim=1) > 0
                mask_bond = torch.sum(Energy_NN['bond'] > 100, dim=1) > 0
                mask_angle = torch.sum(Energy_NN['angle'] > 100, dim=1) > 0
                
                for name in Energy_NN:
                    Energy_NN[name] = torch.sum(Energy_NN[name],axis=1)
                
                Energy_NN = Energy_NN['vdw'] + Energy_NN['bond'] + Energy_NN['angle'] + Energy_NN['UB'] + Energy_NN['imptors'] + Energy_NN['torsion'] + Energy_NN['charge']
                Energy_ref = targets['energy'][params_index['energy'][i]:params_index['energy'][i+1]]
        
                mask_energy = mask_angle + mask_vdw + mask_bond
                if 0 in mask_energy: 
                    Energy_NN = Energy_NN[~mask_energy]
                    Energy_ref = Energy_ref[~mask_energy]
                    minimun_energy = torch.argmin(Energy_NN)
                    Energy_NN = Energy_NN - Energy_NN[minimun_energy]
                    Energy_ref = Energy_ref - Energy_ref[minimun_energy]
                    epoch_mol += 1
                    epoch_conf += len(Energy_ref) + 1
                    bsz_mol += 1
                    loss7 += loss_fn(Energy_NN,Energy_ref)
                        
                        
            train_Loss[-2] += loss6.item()
            try:
                train_Loss[-1] += loss7.item()
            except:
                pass
            
            (100*loss1 + loss2 + loss3 + loss4 + 10*loss5 + 10*loss6 + loss7/bsz_mol).backward()
            optimizer.step()
#             zero the parameter gradients
            optimizer.zero_grad()

            if rate > 1 and optimizer.state_dict()['param_groups'][0]['lr'] > end_lr:
                bool_scheduler_step = False
            elif rate < 1 and optimizer.state_dict()['param_groups'][0]['lr'] < end_lr:
                bool_scheduler_step = False
            if bool_scheduler_step:
                scheduler.step()
                
        train_Loss[:-1] /= len(train_dl)
        train_Loss[-1] = np.sqrt(train_Loss[-1]/epoch_mol)
        
        torch.cuda.empty_cache()
        val_Loss, val_num_conf, val_num_mol = Test_ANI1(model, model2, val_dl)
        torch.cuda.empty_cache()
        test_Loss, test_num_conf, test_num_mol = Test_ANI1(model, model2, test_dl)
        torch.cuda.empty_cache()
        
        if store_name and val_Loss[-1] < best_score[1]:
            best_model_index = epoch
            best_score = [train_Loss[-1], val_Loss[-1], test_Loss[-1]]
            torch.save(model.to('cpu').state_dict(), './model/' + store_name)
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")
            print('New model ', store_name, ' saved!')
        elapsed = time.time() - epoch_start_time
        print('| {:3d}/{:3d} | lr {:.1e} | {:2.0f} s | molecules: {:6d} | conformations {:6d}'.format(epoch, epochs, optimizer.state_dict()['param_groups'][0]['lr'], elapsed, epoch_mol+val_num_mol+test_num_mol, epoch_conf+val_num_conf+test_num_conf))
        print('MSE for parameters (VdW/Bond/Angle/Imptors/Torsion/Charge) + RMSE for Energy:')
        print('Train Loss: {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.3e}'.format(train_Loss[0],train_Loss[1],train_Loss[2],train_Loss[3],train_Loss[4],train_Loss[5],train_Loss[6]))
        print('Val   Loss: {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.3e}'.format(val_Loss[0],val_Loss[1],val_Loss[2],val_Loss[3],val_Loss[4],val_Loss[5],val_Loss[6]))
        print('Test  Loss: {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.3e}\n'.format(test_Loss[0],test_Loss[1],test_Loss[2],test_Loss[3],test_Loss[4],test_Loss[5],test_Loss[6]))
        if (epoch - best_model_index + 1) >= 20:
            break
    print('The Training is finished')
    model.eval()
    return model
