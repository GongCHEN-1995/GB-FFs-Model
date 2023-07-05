import numpy as np
import copy
import time

import torch
import torch.nn as nn

def Test_SPICE(model, model2, dl, charge_type='AM1-BCC', bool_force=False, angle_ub=False, print_info=False):
    model.eval()
    Loss = np.zeros(7)
    Loss_force = 0
    loss_fn = nn.MSELoss()
    loss_fn1 = nn.L1Loss()
    num_conf = 0
    num_mol = 0
    with torch.no_grad():
        for p,data in enumerate(dl):
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
                if bool_force:
                    for i in range(len(coord_inputs_grad)):
                        for name in coord_inputs_grad[i].keys():
                            coord_inputs_grad[i][name] = coord_inputs_grad[i][name].to('cuda')
                            
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
                if charge_type == 'AM1-BCC':
                    mol_parameters['charge'] = mol_targets['charge']
                
                if bool_force:
                    Energy_NN, Force_NN = model2(coord_inputs[i], mol_parameters, coord_inputs_grad[i])
                    Energy_GAFF, Force_GAFF = model2(coord_inputs[i], mol_targets, coord_inputs_grad[i])
                    Force_ref = energy_force[i]['force']
                    mask_force = torch.sum(torch.sum(Force_NN > 200, dim=-1), dim=-1) > 0
                else:
                    Energy_NN, _ = model2(coord_inputs[i], mol_parameters, None)
                    Energy_GAFF, _ = model2(coord_inputs[i], mol_targets, None)
                if print_info and i == print_info:
                    return coord_inputs[i], Energy_ref, Energy_NN, Energy_GAFF, mol_parameters, mol_targets
#                     return coord_inputs[i], Force_ref, Force_NN, Energy_GAFF, mol_parameters, mol_targets
                
                mask_vdw = torch.sum(Energy_NN['vdw'] > 10, dim=1) > 0
                mask_bond = torch.sum(Energy_NN['bond'] > 50, dim=1) > 0
                if angle_ub:
                    mask_angle = torch.sum(Energy_NN['angle'] > 100, dim=1) > 0
                else:
                    mask_angle = torch.sum(Energy_NN['angle'] > 50, dim=1) > 0
                    
                for name in Energy_NN:
                    Energy_NN[name] = torch.sum(Energy_NN[name],axis=1)
                    Energy_GAFF[name] = torch.sum(Energy_GAFF[name],axis=1)
                Energy_ref = energy_force[i]['energy']
#                 if print_info and i == print_info:
#                     return coord_inputs[i], Energy_ref, Energy_NN, Energy_GAFF, mol_parameters, mol_targets
                
                Energy_NN = Energy_NN['vdw'] + Energy_NN['bond'] + Energy_NN['angle'] + Energy_NN['imptors'] + Energy_NN['torsion'] + Energy_NN['charge']
                Energy_GAFF = Energy_GAFF['vdw'] + Energy_GAFF['bond'] + Energy_GAFF['angle'] + Energy_GAFF['imptors'] + Energy_GAFF['torsion'] + Energy_GAFF['charge']
#                 if print_info and i == print_info:
#                     return coord_inputs[i], Energy_ref, Energy_NN, Energy_GAFF, mol_parameters, mol_targets
    
                mask_energy = mask_vdw + mask_bond + mask_angle
                if bool_force:
                    mask_energy += mask_force
                ground_index = (mask_energy.detach()==0).nonzero()
#                 if print_info and i == print_info:
#                     return coord_inputs[i], Energy_ref, Energy_NN, Energy_GAFF, mol_parameters, mol_targets
#                 ground_index = torch.arange(len(Energy_NN)).cuda()
                if len(ground_index) > 1: 
                    minimun_energy = ground_index[torch.argmin(Energy_NN[ground_index])]
                    Energy_NN = Energy_NN - Energy_NN[minimun_energy]
                    Energy_GAFF = Energy_GAFF - Energy_GAFF[minimun_energy]
                    Energy_ref = Energy_ref - Energy_ref[minimun_energy]
                    
                    Energy_GAFF = Energy_GAFF.masked_fill(mask_energy,0)
                    Energy_NN = Energy_NN.masked_fill(mask_energy,0)
                    Energy_ref = Energy_ref.masked_fill(mask_energy,0)
                    if print_info and i == print_info:
                        return coord_inputs[i], Energy_ref, Energy_NN, Energy_GAFF, mol_parameters, mol_targets
                    Energy_GAFF = Energy_GAFF[Energy_GAFF.nonzero()]
                    Energy_NN = Energy_NN[Energy_NN.nonzero()]
                    Energy_ref = Energy_ref[Energy_ref.nonzero()]
                    
                    if len(Energy_NN) == len(Energy_ref):
                        num_conf += len(Energy_ref) + 1
                        num_mol += 1
                        Loss[-1] += np.sqrt(loss_fn(Energy_NN,Energy_ref).item())
                        if bool_force:
                            Force_NN = Force_NN.masked_fill(mask_energy.view(-1,1,1),0)
                            Force_ref = Force_ref.masked_fill(mask_energy.view(-1,1,1),0)
                            Loss_force += np.sqrt(loss_fn(Force_NN, Force_ref).item())

    Loss[:-1] /= len(dl)
    Loss[-1] = Loss[-1]/num_mol
    return Loss, Loss_force/num_mol, num_conf, num_mol
        
def Train_SPICE(model, model2, train_dl, val_dl, test_dl, charge_type='AM1-BCC', bool_force=False, force_weight=1, angle_ub=False, epochs=1, start_lr=1e-6, end_lr=0.1, rate=0.99, step_size=1, store_name=None):
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
        train_Loss_force = 0
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
                if bool_force:
                    for i in range(len(coord_inputs_grad)):
                        for name in coord_inputs_grad[i].keys():
                            coord_inputs_grad[i][name] = coord_inputs_grad[i][name].to('cuda')
                            
            # forward + backward + optimize
            parameters = model(inputs)
            loss1 = loss_fn(parameters['vdw'],targets['vdw'])
            loss2 = loss_fn(parameters['bond'],targets['bond'])
            loss3 = loss_fn(parameters['angle'],targets['angle'])
            loss4 = loss_fn(parameters['imptors'],targets['imptors'])
            loss5 = loss_fn(parameters['torsion'],targets['torsion']) + 0.01*torch.norm(parameters['torsion'], 2)
            loss6 = 0.001*torch.norm(parameters['transfer_charge'], 2)

            train_Loss += np.array([loss1.item(), loss2.item(), loss3.item(), 0, loss5.item(), 0, 0])
            if not np.isnan(loss4.item()):
                train_Loss[3] += loss4.item()
            loss7 = 0
            loss_force = 0
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
                if charge_type == 'AM1-BCC':
                    mol_parameters['charge'] = mol_targets['charge']
            
                if bool_force:
                    Energy_NN, Force_NN = model2(coord_inputs[i], mol_parameters, coord_inputs_grad[i])
                    Force_ref = energy_force[i]['force']
                    Force_ref_sum = torch.sqrt(torch.sum(Force_ref.pow(2), dim=1)).unsqueeze(1)
                    mask_force = torch.sum(torch.sum(Force_NN > 200, dim=-1), dim=-1) > 0
                else:
                    Energy_NN, _ = model2(coord_inputs[i], mol_parameters, None)

                mask_vdw = torch.sum(Energy_NN['vdw'] > 10, dim=1) > 0
                mask_bond = torch.sum(Energy_NN['bond'] > 50, dim=1) > 0
                if angle_ub:
                    mask_angle = torch.sum(Energy_NN['angle'] > 100, dim=1) > 0
                else:
                    mask_angle = torch.sum(Energy_NN['angle'] > 50, dim=1) > 0
                
                for name in Energy_NN:
                    Energy_NN[name] = torch.sum(Energy_NN[name],axis=1)
                
                Energy_NN = Energy_NN['vdw'] + Energy_NN['bond'] + Energy_NN['angle'] + Energy_NN['imptors'] + Energy_NN['torsion'] + Energy_NN['charge']
                Energy_ref = energy_force[i]['energy']
        
                mask_energy = mask_vdw + mask_bond + mask_angle
                if bool_force:
                    mask_energy += mask_force
                ground_index = (mask_energy.detach()==0).nonzero()

                if len(ground_index) > 1: 
                    minimun_energy = ground_index[torch.argmin(Energy_NN[ground_index])]
                    Energy_ref = Energy_ref - Energy_ref[minimun_energy] + Energy_NN[minimun_energy]
                    Energy_NN = Energy_NN.masked_fill(mask_energy,0)
                    Energy_ref = Energy_ref.masked_fill(mask_energy,0)
                    Energy_NN = Energy_NN[Energy_NN.nonzero()]
                    Energy_ref = Energy_ref[Energy_ref.nonzero()]
                    if len(Energy_NN) > 0 and len(Energy_NN) == len(Energy_ref):
                        epoch_conf += len(Energy_ref) + 1
                        epoch_mol += 1
                        bsz_mol += 1
                        loss7 += loss_fn(Energy_NN,Energy_ref)
                        
                        if bool_force:
                            Force_NN = Force_NN.masked_fill(mask_energy.view(-1,1,1),0)
                            Force_ref = Force_ref.masked_fill(mask_energy.view(-1,1,1),0)
                            loss_force += loss_fn(Force_NN, Force_ref)
                        
            train_Loss[-2] += loss6.item()
            if bool_force:
                loss_force *= force_weight
                
            try:
                train_Loss[-1] += np.sqrt(loss7.item()/bsz_mol)
            except:
                pass
            try:
                train_Loss_force += np.sqrt(loss_force.item()/bsz_mol)
            except:
                pass
            
            bsz_mol = max(1,bsz_mol)
            (100*loss1 + loss2 + loss3 + 10*loss4 + 10*loss5 + 100*loss6 + loss7/bsz_mol + loss_force/bsz_mol).backward()

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
        train_Loss[:-1] /= len(train_dl)
        train_Loss[-1] = train_Loss[-1]/len(train_dl)
        train_Loss_force = train_Loss_force/len(train_dl)
        
        torch.cuda.empty_cache()
        val_Loss, val_Loss_force, val_num_conf, val_num_mol = Test_SPICE(model, model2, val_dl, charge_type, bool_force, angle_ub, print_info=False)
        torch.cuda.empty_cache()
        test_Loss, test_Loss_force, test_num_conf, test_num_mol = Test_SPICE(model, model2, test_dl, charge_type, bool_force, angle_ub, print_info=False)
        torch.cuda.empty_cache()
        
        if store_name and (val_Loss[-1]+force_weight*val_Loss_force) < best_score[1]:
            best_model_index = epoch
            best_score = [train_Loss[-1]+force_weight*train_Loss_force, val_Loss[-1]+force_weight*val_Loss_force, test_Loss[-1]+force_weight*test_Loss_force]
            torch.save(model.to('cpu').state_dict(), './model/' + store_name)
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")
            print('New model ', store_name, ' saved!')
        elapsed = time.time() - epoch_start_time
        print('| {:3d}/{:3d} | lr {:.1e} | {:2.0f} s | molecules: {:6d} | conformations {:6d}'.format(epoch, epochs, optimizer.state_dict()['param_groups'][0]['lr'], elapsed, epoch_mol+val_num_mol+test_num_mol, epoch_conf+val_num_conf+test_num_conf))
        if bool_force:
            print('Train Loss: {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.3e} {:.3e}'.format(train_Loss[0],train_Loss[1],train_Loss[2],train_Loss[3],train_Loss[4],train_Loss[5],train_Loss[6],train_Loss_force))
            print('Val   Loss: {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.3e} {:.3e}'.format(val_Loss[0],val_Loss[1],val_Loss[2],val_Loss[3],val_Loss[4],val_Loss[5],val_Loss[6],val_Loss_force))
            print('Test  Loss: {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.3e} {:.3e}'.format(test_Loss[0],test_Loss[1],test_Loss[2],test_Loss[3],test_Loss[4],test_Loss[5],test_Loss[6],test_Loss_force))
        else:
            print('Train Loss: {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.3e}'.format(train_Loss[0],train_Loss[1],train_Loss[2],train_Loss[3],train_Loss[4],train_Loss[5],train_Loss[6]))
            print('Val   Loss: {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.3e}'.format(val_Loss[0],val_Loss[1],val_Loss[2],val_Loss[3],val_Loss[4],val_Loss[5],val_Loss[6]))
            print('Test  Loss: {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.3e}'.format(test_Loss[0],test_Loss[1],test_Loss[2],test_Loss[3],test_Loss[4],test_Loss[5],test_Loss[6]))

        if (epoch - best_model_index + 1) >= 20:
            break
    print('The Training is finished')
    model.eval()
    return model
