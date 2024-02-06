from fea_core import *


# definition and initialization
jobName = modelName = 'knee'
nonliplsTools = nonlipls_tools(jobName, modelName)

# defining the steps and field outputs
vars = ('S', 'PE', 'PEEQ', 'PEMAG', 'LE', 'U', 'RF', 'CF', 
        'CSTATUS', 'CDISP', 'COORD')

timer = timer_tools('lf', 'hf')
fileIo = file_io("gnn_datasets", modelName)
field_tools_params = {'nodeSetKey': cartilageSet, 'fileIo': fileIo, 'excludingSteps': {eqStepName}}
mesh_tools_params = {'nodeSetKey': cartilageSet, 'fileIo': fileIo}

def set_element_type(modelObj, elemCode, partName='tibia_cartilage_MED', elemLibrary=STANDARD):
    partObj = modelObj.parts[partName]
    partObj.setElementType(
        regions=(partObj.cells, ),
        elemTypes=(mesh.ElemType(elemCode=elemCode, elemLibrary=elemLibrary), )
        )

# defining the models used in surrogate modeling:
odbNames = {'lf': nonliplsTools.odbNameWithoutOptimizaion, 'hf': nonliplsTools.odbName}
open_odb_fn = {key: partial(open_odb, odbNames[key]) for key in odbNames}
jobNames = {'lf': nonliplsTools.jobNameWithoutOptimizaion, 'hf': nonliplsTools.jobName}
job_submit_fn = {key: partial(job_submit, jobNames[key], subroutineFile='NONLIPLS.for')
                 for key in jobNames}
modelNames = {'lf': nonliplsTools.modelNameWithoutOptimizaion, 'hf': nonliplsTools.modelName}


def perform_psa():
    # a helper function to run PSA and prepare for subsequent analysis.
    try:
        timer.start('hf')
        nonliplsTools.initialize_params('FEMUR_CARTILAGE')
        
        models['lf'].steps[eqStepName].resume()
        
        for predefinedFieldsObj in models['lf'].predefinedFields.values():
            predefinedFieldsObj.resume()
        
        if nonliplsTools.run_prestress_optimizer('ARTICULAR_CARTILAGE') == 'ABORTED':
            raise Exception('Error: pre-stressing is not complete!')
        
        timer.pause('hf')
        
    except Exception as ex:
        timer.ignore('hf')
        raise Exception("An error interrupted the operation: %s" % str(ex))
    
    models = {key: mdb.models[modelNames[key]] for key in modelNames}
    
    
    models['hf'].SoilsStep(utol=2e10, creep=OFF, matrixStorage=UNSYMMETRIC, **createStepParams)
    models['lf'].StaticStep(
        name='LOAD', previous='EQ', maintainAttributes=True, timePeriod=10.0,
        maxNumInc=1000, initialInc=10.0, minInc=0.00001, maxInc=10.0,
        matrixSolver=DIRECT, matrixStorage=UNSYMMETRIC, extrapolation=LINEAR
        )
    
    for predefinedFieldsObj in models['hf'].predefinedFields.values():
        predefinedFieldsObj.resume()
    
    for predefinedFieldsObj in models['lf'].predefinedFields.values():
        predefinedFieldsObj.suppress()
    
    for key in ['lf', 'hf']:
        for predefinedFieldsObj in models[key].predefinedFields.values():
            predefinedFieldsObj.resume()
    
    models['hf'].FieldOutputRequest(
        frequency=1, variables=vars+('SDV','POR',), **fieldOutputParams
        )
    models['hf'].fieldOutputRequests[eqStepName].deactivate(stepName)
    models['hf'].TimePoint(name=stepName, points=[[0]])
    models['lf'].FieldOutputRequest(
        timePoint=stepName, timeMarks=ON, variables=vars, **fieldOutputParams
        )
    
    for instanceName in ['tibia_cartilage_MED', 'femur_cartilage', 'tibia_cartilage_LAT']:
        set_element_type(models['hf'], C3D8P, instanceName)
        set_element_type(models['lf'], C3D8, instanceName)
    
    models['lf'].sections['CAR'].setValues(material='CAR', thickness=None)
    poreBcObj = models['lf'].boundaryConditions['PORE']
    
    constants = models['hf'].materials['CAR_UMAT'].userMaterial.mechanicalConstants
    constants[2] = 1.0
    models['hf'].materials['CAR_UMAT'].userMaterial.setValues(mechanicalConstants=constants)
    
    models['lf'].fieldOutputRequests[eqStepName].suppress()
    
    models['hf'].steps[eqStepName].resume()
        
    for predefinedFieldsObj in models['hf'].predefinedFields.values():
        predefinedFieldsObj.resume()
    
    if ~poreBcObj.suppressed:
        poreBcObj.suppress()
    
    return models

numIteration = 6

errorParams = []
surrogateParams = {}
n = 1

while True:
    try:
        models = perform_psa()
        surrogateParams['bcParams'] = {
            'cf2': np.random.uniform(0, -1000),
            }
        surrogateParams['t'] = np.random.uniform(0.001,100)
        surrogateParams['amp'] = np.array([0, 1], 'd')
        
        print "\n** Sample num: %s **\n" % (n)
        
        # updating the models
        stepParams['timePeriod'] = surrogateParams['t']
        stepParams['initialInc'] = stepParams['maxInc'] = stepParams['timePeriod']/10.0
        models['hf'].steps[stepName].setValues(**stepParams)
        amplitudeTool = amplitude_tools(surrogateParams['amp'], stepParams['timePeriod'])
        amplitudeTool.set(models['hf'])
        
        for key in ['lf', 'hf']:
            models[key].loads['LOAD'].setValuesInStep(
                stepName=stepName, **surrogateParams['bcParams']
                )
        
        # run the hf model, and if successful open it and get the syncing data
        timer.start('hf')
        jobStatus = job_submit_fn['hf']()
        
        if jobStatus in ['ERROR', 'TERMINATED']:
            timer.ignore('hf')
            continue
        else:
            odbs = {'hf': open_odb_fn['hf']()}
            instances = {'hf': odbs['hf'].rootAssembly.instances['PART-1-1']}
            fieldTools = {'hf': field_tools(
                odbs['hf'], instances['hf'], jobStatus=jobStatus, mainName='hf', **field_tools_params
                )}
            relativeTimes = fieldTools['hf'].relativeTimes
            
            if len(relativeTimes) == 0:
                timer.ignore('hf')
                continue
        
        # syncing the lf model with the hf model
        timePoints = relativeTimes[stepName]
        stepParams['timePeriod'] = lastTimePoint = timePoints[-1, 0] # get the last time point
        
        if stepParams['timePeriod'] < stepParams['initialInc']:
            stepParams['initialInc'] = stepParams['maxInc'] = stepParams['timePeriod']
        
        models['lf'].steps[stepName].setValues(**stepParams)
        models['lf'].TimePoint(name=stepName, points=timePoints)
        
        amplitudeTool.cut(lastTimePoint)
        amplitudeTool.set(models['lf'])
        
        timer.start('lf')
        jobStatus = job_submit_fn['lf']()
        
        if jobStatus in ['ERROR', 'TERMINATED']:
            timer.ignore('lf')
            timer.ignore('hf')
            continue
        else:
            odbs['lf'] = open_odb_fn['lf']()
            instances['lf'] = odbs['lf'].rootAssembly.instances['PART-1-1']
            fieldTools['lf'] = field_tools(
                odbs['lf'], instances['lf'], jobStatus=jobStatus, mainName='lf', **field_tools_params
                )
            relativeTimes = fieldTools['lf'].relativeTimes
            
            if len(relativeTimes) == 0:
                timer.ignore('lf')
                timer.ignore('hf')
                continue
                
            else:
                timer.stop('lf')
                timer.stop('hf')
                trajectoryLengthLf = len(fieldTools['lf'].frames)
                
                if trajectoryLengthLf < len(fieldTools['hf'].frames):
                    fieldTools['hf'].frames = fieldTools['hf'].frames[:trajectoryLengthLf]
                    fieldTools['hf'].times = fieldTools['hf'].times[:trajectoryLengthLf]
        
        # postprocessing
        pos = {key: fieldTools[key].get_coords() for key in odbs}
        meshTools = {
            key: mesh_tools(instances[key], mainName=key, **mesh_tools_params) for key in odbs
            }
        
        # some assertions:
        assert np.all(pos['hf']['labels'] == pos['lf']['labels']),\
            Exception('The labels of the lf and hf models do not match.')
        
        fieldNodesLabels = meshTools['lf'].correct_node_labels(pos['lf']['labels'][:, 0])
        assert np.all(meshTools['lf'].nodeIndeces == fieldNodesLabels ),\
            Exception('The nodal indeces from mesh_tools and field_tools do not match.')
        
        initialNodalDistances = np.abs(pos['hf']['data'][:, 0] - pos['lf']['data'][:, 0]).max()
        assert initialNodalDistances < pos['hf']['shortest_edge'], \
            Exception('Some of the corresponding lf and hf nodes have too different coordinates.')
        
        # recoding arrays
        senders, receivers = meshTools['lf'].get_topology(elementSetKey, abq_node_numbering)
        meshTools['lf'].record_augmented_topology(
            senders, receivers, unchangeableNodesKey, pos['lf']['data']
            )
        meshTools['lf'].record_node_type(nodeSetsDict)
        
        for k in ['lf', 'hf']:
            fieldTools[k].record_field('CSTATUS', 'contact_status', dtype='int32')
            fieldTools[k].record_field('LE', 'strain', invariantsList)
            fieldTools[k].record_field('S', 'stress', invariantsList)
        
        fieldTools['hf'].record_field('POR', 'pore_pressure', relative=True)
        fieldTools['hf'].record_field('SDV5', 'gag_stress', relative=True)
        fieldTools['hf'].record_field('SDV6', 'depth')
        fieldTools['hf'].make_tensor_from_sdvs(range(70,76), 'FIB')
        fieldTools['hf'].make_tensor_from_sdvs(range(76,82), 'NONFIB')
        fieldTools['hf'].make_tensor_from_sdvs(range(16,19), 'VEC1')
        fieldTools['hf'].make_tensor_from_sdvs(range(19,22), 'VEC2')
        fieldTools['hf'].record_field('FIB', 'fibrilar_stress', invariantsList, relative=True)
        fieldTools['hf'].record_field('NONFIB', 'non_fibrilar_stress', invariantsList, relative=True)
        fieldTools['hf'].record_field('VEC1', 'vec_1')
        fieldTools['hf'].record_field('VEC2', 'vec_2')
        fieldTools['hf'].record_time()
        fieldTools['hf'].record_trajectory_length()
        
        close_all_odbs()
        fileIo.store_meta()
        
        if n != numIteration:
            n += 1
        else:
            break
        
    except Exception as ex:
        print 'Warning: An error is catched in the surrogate loop (i.e., %s)!' % ex
        errorParams.append(surrogateParams)
        raise Exception('ERROR: surrogate loop is intrupted (i.e., %s)!' % ex)

fileIo.record_csv(timer.get())
