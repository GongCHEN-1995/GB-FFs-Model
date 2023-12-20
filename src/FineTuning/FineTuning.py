import numpy as np
import copy
import time

import torch
import torch.nn as nn

def Test_FineTuning(model, model2, dl, dataset='both',):
    model.eval()
    Loss = np.zeros(8)
    loss_fn = nn.MSELoss()
    loss_fn1 = nn.L1Loss()
    num_conf = 0
    num_mol_SPICE = 0
    num_mol_DES370K = 0
    with torch.no_grad():
        for p,data in enumerate(dl):
            inputs, targets, coord_inputs, coord_inputs_grad, energy_force, params_index = copy.deepcopy(data)
            if torch.cuda.is_available():
                for name in inputs.keys():
                    inputs[name] = inputs[name].to('cuda')
                for name in targets.keys():
                    targets[name] = targets[name].to('cuda')
                for i in range(len(coord_inputs)):
                    for name in coord_inputs[i].keys():
                        coord_inputs[i][name] = coord_inputs[i][name].to('cuda')
                for i in range(len(energy_force)):
                    for name in energy_force[i].keys():
                        energy_force[i][name] = energy_force[i][name].to('cuda')
                for i in range(len(coord_inputs_grad)):
                    for name in coord_inputs_grad[i].keys():
                        coord_inputs_grad[i][name] = coord_inputs_grad[i][name].to('cuda')
                            
            # forward + backward + optimize
            parameters = model(inputs) 

            Loss[0] += loss_fn(parameters['vdw'],targets['vdw']).item()
            tmp = loss_fn(parameters['bond'],targets['bond']).item()
            if not np.isnan(tmp):
                Loss[1] += tmp 
            tmp = (loss_fn(parameters['angle'][:,:2],targets['angle'][:,:2])+loss_fn(parameters['angle'][:,3:],targets['angle'][:,3:])).item()
            if not np.isnan(tmp):
                Loss[2] += tmp 
            tmp = loss_fn(parameters['imptors'],targets['imptors']).item()
            if not np.isnan(tmp):
                Loss[3] += tmp
            tmp = loss_fn(parameters['torsion'],targets['torsion']).item()
            if not np.isnan(tmp):
                Loss[4] += tmp

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

                bool_SPICE = True if 'force' in energy_force[i] else False
                if dataset == 'SPICE' and not bool_SPICE:
                    continue
                elif dataset == 'DES370K' and bool_SPICE:
                    continue
                mol_targets['charge'] /= 1.1
                
                if bool_SPICE:
                    Energy_NN, Force_NN = model2(coord_inputs[i], mol_parameters, coord_inputs_grad[i])
                    Energy_GAFF, Force_GAFF = model2(coord_inputs[i], mol_targets, coord_inputs_grad[i])
                    Force_ref = energy_force[i]['force']
                    mask_force = torch.sum(torch.sum(torch.abs(Force_NN) > 200, dim=-1), dim=-1) > 0
                else:
                    Energy_NN, _ = model2(coord_inputs[i], mol_parameters, None)
                    Energy_GAFF, _ = model2(coord_inputs[i], mol_targets, None)
                
                mask_vdw = torch.sum(Energy_NN['vdw'] > 5, dim=1) > 0
                mask_bond = torch.sum(Energy_NN['bond'] > 50, dim=1) > 0
                mask_angle = torch.sum(Energy_NN['angle'] > 50, dim=1) > 0
                    
                for name in Energy_NN:
                    Energy_NN[name] = torch.sum(Energy_NN[name],axis=1)
                    Energy_GAFF[name] = torch.sum(Energy_GAFF[name],axis=1)
                Energy_ref = energy_force[i]['energy']

                Energy_NN = Energy_NN['vdw'] + Energy_NN['bond'] + Energy_NN['angle'] + Energy_NN['UB'] + Energy_NN['imptors'] + Energy_NN['torsion'] + Energy_NN['charge']
                Energy_GAFF = Energy_GAFF['vdw'] + Energy_GAFF['bond'] + Energy_GAFF['angle'] + Energy_GAFF['UB'] + Energy_GAFF['imptors'] + Energy_GAFF['torsion'] + Energy_GAFF['charge']
    
                mask_energy = mask_vdw + mask_bond + mask_angle
                if bool_SPICE:
                    mask_energy += mask_force
                if 0 in mask_energy: 
                    Energy_NN = Energy_NN[~mask_energy]
                    Energy_ref = Energy_ref[~mask_energy]
                    if bool_SPICE:
                        minimun_energy = torch.argmin(Energy_NN)
                        Energy_NN = Energy_NN - Energy_NN[minimun_energy]
                        Energy_ref = Energy_ref - Energy_ref[minimun_energy]
                    
                    num_conf += len(Energy_ref) + 1
                    if bool_SPICE:
                        num_mol_SPICE += 1
                    else:
                        num_mol_DES370K += 1
                    Loss[5] += np.sqrt(loss_fn(mol_parameters['charge'],mol_targets['charge']).item())
                    Loss[6] += np.sqrt(loss_fn(Energy_NN,Energy_ref).item())
                    if bool_SPICE:
                        Force_NN = Force_NN[~mask_energy]
                        Force_ref = Force_ref[~mask_energy]
                        Loss[7] += np.sqrt(loss_fn(Force_NN, Force_ref).item())

    Loss[:-3] /= len(dl)
    Loss[-3:-1] /= (num_mol_SPICE + num_mol_DES370K)
    Loss[-1] /= max(num_mol_SPICE, 1)
    return Loss, num_conf, num_mol_SPICE + num_mol_DES370K
        
def Train_FineTuning(model, model2, train_dl, val_dl, test_dl, force_weight=1, epochs=1, start_lr=1e-6, end_lr=0.1, rate=0.99, step_size=1, store_name=None):
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
        train_Loss = np.zeros(8)
        for p,data in enumerate(train_dl):
            inputs, targets, coord_inputs, coord_inputs_grad, energy_force, params_index = copy.deepcopy(data)
            bsz = len(params_index['atom']) - 1
            if torch.cuda.is_available():
                for name in inputs.keys():
                    inputs[name] = inputs[name].to('cuda')
                for name in targets.keys():
                    targets[name] = targets[name].to('cuda')
                for i in range(len(coord_inputs)):
                    for name in coord_inputs[i].keys():
                        coord_inputs[i][name] = coord_inputs[i][name].to('cuda')
                for i in range(len(energy_force)):
                    for name in energy_force[i].keys():
                        energy_force[i][name] = energy_force[i][name].to('cuda')
                for i in range(len(coord_inputs_grad)):
                    for name in coord_inputs_grad[i].keys():
                        coord_inputs_grad[i][name] = coord_inputs_grad[i][name].to('cuda')
                            
            # forward + backward + optimize
            parameters = model(inputs)
            loss1 = loss_fn(parameters['vdw'],targets['vdw'])
            loss2 = loss_fn(parameters['bond'][:,1:2],targets['bond'][:,1:2])
            loss3 = loss_fn(parameters['angle'][:,:2],targets['angle'][:,:2])+10000*loss_fn(parameters['angle'][:,3:],targets['angle'][:,3:])
            loss4 = loss_fn(parameters['imptors'],targets['imptors'])
            loss5 = loss_fn(parameters['torsion'],targets['torsion']) + 0.01*torch.norm(parameters['torsion'], 2)
            loss6 = 0.001*torch.norm(parameters['transfer_charge'], 2)
            train_Loss[0] += loss1.item()
            if not np.isnan(loss2.item()):
                train_Loss[1] += loss2.item()
            if not np.isnan(loss3.item()):
                train_Loss[2] += loss3.item()
            if not np.isnan(loss4.item()):
                train_Loss[3] += loss4.item()
            if not np.isnan(loss5.item()):
                train_Loss[4] += loss5.item()
                
            loss7 = 0
            loss_force = 0
            bsz_mol_SPICE = 0
            bsz_mol_DES370K = 0
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
                loss6 += loss_fn(mol_parameters['charge'],mol_targets['charge'])
                bool_SPICE = True if 'force' in energy_force[i] else False
             
                if bool_SPICE:
                    Energy_NN, Force_NN = model2(coord_inputs[i], mol_parameters, coord_inputs_grad[i])
                    Force_ref = energy_force[i]['force']
                    Force_ref_sum = torch.sqrt(torch.sum(Force_ref.pow(2), dim=1)).unsqueeze(1)
                    mask_force = torch.sum(torch.sum(torch.abs(Force_NN) > 200, dim=-1), dim=-1) > 0
                else:
                    Energy_NN, _ = model2(coord_inputs[i], mol_parameters, None)

                mask_vdw = torch.sum(Energy_NN['vdw'] > 5, dim=1) > 0
                mask_bond = torch.sum(Energy_NN['bond'] > 50, dim=1) > 0
                mask_angle = torch.sum(Energy_NN['angle'] > 50, dim=1) > 0
                
                for name in Energy_NN:
                    Energy_NN[name] = torch.sum(Energy_NN[name],axis=1)
                
                Energy_NN = Energy_NN['vdw'] + Energy_NN['bond'] + Energy_NN['angle'] + Energy_NN['UB'] + Energy_NN['imptors'] + Energy_NN['torsion'] + Energy_NN['charge']
                Energy_ref = energy_force[i]['energy']
        
                mask_energy = mask_vdw + mask_bond + mask_angle
                if bool_SPICE:
                    mask_energy += mask_force
                
                if 0 in mask_energy: 
                    Energy_NN = Energy_NN[~mask_energy]
                    Energy_ref = Energy_ref[~mask_energy]
                    if bool_SPICE:
                        minimun_energy = torch.argmin(Energy_NN)
                        Energy_ref = Energy_ref - Energy_ref[minimun_energy] + Energy_NN[minimun_energy]
                    
                    epoch_conf += len(Energy_ref) + 1
                    epoch_mol += 1
                    if bool_SPICE:
                        bsz_mol_SPICE += 1
                        loss7 += loss_fn(Energy_NN,Energy_ref)
                    else:
                        bsz_mol_DES370K += 1
                        loss7 += 100 * loss_fn(Energy_NN,Energy_ref)
                    if bool_SPICE:
                        Force_NN = Force_NN[~mask_energy]
                        Force_ref = Force_ref[~mask_energy]
                        loss_force += loss_fn(Force_NN, Force_ref)

            bsz_mol_SPICE = max(1,bsz_mol_SPICE)
            loss6 /= (bsz_mol_SPICE + bsz_mol_DES370K)
            loss7 /= (bsz_mol_SPICE + bsz_mol_DES370K)
            loss_force /= bsz_mol_SPICE
            
            train_Loss[5] += np.sqrt(loss6.item())
            loss_force *= force_weight
                
            try:
                train_Loss[-2] += np.sqrt(loss7.item())
            except:
                pass
            try:
                train_Loss[-1] += np.sqrt(loss_force.item())
            except:
                pass
            
            (1000*loss1 + loss2 + loss3 + loss4 + 10*loss5 + 100*loss6 + loss7 + loss_force).backward()

            optimizer.step()
#             zero the parameter gradients
            optimizer.zero_grad()

            if rate > 1 and optimizer.state_dict()['param_groups'][0]['lr'] > end_lr:
                bool_scheduler_step = False
            elif rate < 1 and optimizer.state_dict()['param_groups'][0]['lr'] < end_lr:
                bool_scheduler_step = False
            if bool_scheduler_step:
                scheduler.step()
                
        epoch_mol = max(epoch_mol, 1)
        train_Loss /= len(train_dl)
        
        torch.cuda.empty_cache()
        val_Loss, val_num_conf, val_num_mol = Test_FineTuning(model, model2, val_dl, dataset='both')
        torch.cuda.empty_cache()
        test_Loss, test_num_conf, test_num_mol = Test_FineTuning(model, model2, test_dl, dataset='both')
        torch.cuda.empty_cache()
        
        if store_name and (val_Loss[-2]+force_weight*val_Loss[-1]) < best_score[1]:
            best_model_index = epoch
            best_score = [train_Loss[-2]+force_weight*train_Loss[-1], val_Loss[-2]+force_weight*val_Loss[-1], test_Loss[-2]+force_weight*test_Loss[-1]]
            torch.save(model.to('cpu').state_dict(), './model/' + store_name)
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")
            print('New model ', store_name, ' saved!')
        elapsed = time.time() - epoch_start_time
        print('| {:3d}/{:3d} | lr {:.1e} | {:2.0f} s | molecules: {:6d} | conformations {:6d}'.format(epoch, epochs, optimizer.state_dict()['param_groups'][0]['lr'], elapsed, epoch_mol+val_num_mol+test_num_mol, epoch_conf+val_num_conf+test_num_conf))
        print('MSE for parameters (VdW/Bond/Angle/Imptors/Torsion/Charge) + RMSE for Energy/Force:')
        print('Train Loss: {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.3e} {:.3e}'.format(train_Loss[0],train_Loss[1],train_Loss[2],train_Loss[3],train_Loss[4],train_Loss[5],train_Loss[6],train_Loss[7]))
        print('Val   Loss: {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.3e} {:.3e}'.format(val_Loss[0],val_Loss[1],val_Loss[2],val_Loss[3],val_Loss[4],val_Loss[5],val_Loss[6],val_Loss[7]))
        print('Test  Loss: {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.3e} {:.3e}'.format(test_Loss[0],test_Loss[1],test_Loss[2],test_Loss[3],test_Loss[4],test_Loss[5],test_Loss[6],test_Loss[7]))

        if (epoch - best_model_index + 1) >= 20:
            break
    print('The Training is finished')
    model.eval()
    return model
