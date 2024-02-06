from abaqus import *
from abaqusConstants import *
import mesh, job, step, assembly, part
from shutil import copyfile, rmtree
import copy, json, warnings, sys, os, time
import numpy as np
from scipy.spatial import KDTree
from functools import partial
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import OrderedDict
import displayGroupOdbToolset as dgo
import pickle, random, string


def job_submit(
    jobName,
    SDVINI=True, # if SDVINI subroutine is used
    nodalOutputPrecision=SINGLE, # or FULL
    subroutineFile='', # if having subroutine, assign its name, e.g., "subroutine.for".
    numCpus=1, # avoid multiprocessing when using common blocks in Fortran.
    numGPUs=0,
    intruptWithError=False,
    ensureOdbIsClosed=True,
    ):
    # the possible output is the job status, e.g., 'ERROR', 'ABORTED', 'TERMINATED', 'COMPLETED'.
    
    odbName = jobName + '.odb'
    staName = jobName + '.sta'
    lckName = jobName + '.lck'
    
    if os.path.exists(lckName):
        raise Exception('Error: an lck file is detected!')
    
    if ensureOdbIsClosed and os.path.exists(odbName):
        odbName = jobName+'.odb'
        
        if odbName in session.odbs:
            close_odb(odbName)
    
    if os.path.exists(staName):
        try:
            os.remove(staName)
        except OSError:
            print 'Warning: failed to delete the sta file (%s)! Trying once more after 30 Sec.'
            time.sleep(30)
            os.remove(staName)
    
    if SDVINI:
        modelName = mdb.jobs[jobName].model
        keywordBlock = mdb.models[modelName].keywordBlock
        keywordBlock.setValues(edited = 0)
        keywordBlock.synchVersions(storeNodesAndElements=False)
        
        def _give_idx(searchCriteria):
            return [line.startswith(searchCriteria) for line in keywordBlock.sieBlocks].index(True)
        try:
            idx = _give_idx('** ----------------------------------------'\
                            '------------------------\n** \n** STEP:')
            keywordBlock.insert(idx-1, '\n*INITIAL CONDITIONS, TYPE=SOLUTION, USER')
        except:
            raise Exception(
                'Error: "job_submit" function cannot insert the relavant SDVINI keyboard. '
                'Either fix the function, or insert SDVINI = False, if possible.'
                )
    
    job = mdb.jobs[jobName]
    job.setValues(
        nodalOutputPrecision=nodalOutputPrecision,
        userSubroutine=os.path.join(os.getcwd(), subroutineFile),
        numCpus=numCpus,
        numGPUs=numGPUs
        )
    
    try:
        job.submit(consistencyChecking=ON)
        # job.waitForCompletion() should not be placed just after
        # job.submit(), as it might cause rare strange errors.
        # Instead, the code checks constantly the lck file. 
        # job.waitForCompletion() is still neccessary, otherwise
        # the status of the job in CAE may not be updated sometimes.
        print '%s is just submitted!'%jobName
        
        while not os.path.exists(lckName):
            time.sleep(2)
        
        # printing the sta during submission, while waiting for the lck to be deleted.
        pos = 0L
        
        while os.path.exists(lckName):
            time.sleep(5)
            
            if os.path.exists(staName):
                try:
                    with open(staName, "r") as f:
                        f.seek(pos)
                        
                        for line in f.readlines():
                            print line.strip()
                        
                        pos = f.tell()
                except OSError as ex:
                    print 'Warning: an os error was catched (%s)!' % ex
                finally:
                    f.close()
        
        job.waitForCompletion()
        status = str(job.status)
        
    except AbaqusException as ex:
        job.waitForCompletion()
        
        if intruptWithError == True:
            raise ex
        else:
            print 'Warning: an error is avoided during job submitting (i.e., %s)' % ex
            
            while os.path.exists(lckName):
                print 'Waiting for the lck file to be deleted!'
                time.sleep(10)
            
            status = 'ERROR'
    
    time.sleep(5)
    
    return status


def copy_model(NameOfNewModel, nameOfCopiedModel):
    mdb.Model(name = NameOfNewModel, objectToCopy = mdb.models[nameOfCopiedModel])


def copy_job(NameOfNewJob, nameOfCopiedJob):
    mdb.Job(name = NameOfNewJob, objectToCopy = mdb.jobs[nameOfCopiedJob])


def open_odb(odbName, readOnly=True):
    _open_odb_fn = lambda: session.openOdb(name=odbName, readOnly=readOnly)
    try:
        return _open_odb_fn()
    except Exception as ex1: # if it does not work try again after 5Sec.
        try:
            return _open_odb_fn()
        except Exception as ex2:
            raise Exception('Error: open_odb() did not work. \n%s\n%s'%(ex1, ex2))


def close_odb(odbName, saveOdb = False):
    try:
        odbObj = session.odbs[odbName]
        
        if saveOdb == True:
            odbObj.save()
        
        odbObj.close()
    except Exception as ex:
        raise Exception('Error: close_odb() did not work.\n%s'%(ex))


def close_all_odbs():
    for odbObj in session.odbs.values():
        odbObj.close()


def unit_vector(vector):
    norm = np.linalg.norm(vector)
    if (norm==0):
        vector += 1e-6
        norm = np.linalg.norm(vector)
    
    return vector / norm


class nonlipls_tools():
    """This class contains all the pre-stressing and relevant intialization code,
    where the inital state is claculate and later used for calculation correct pre-stress
    state. The way a high-fidelity cartilage model can then be run."""
    
    def __init__(
        self,
        jobName,
        modelName,
        materialName='CAR_UMAT',
        sectionName='CAR',
        stepName='EQ',
        txtFolderName='txt',
        subroutineFile='subroutines.for',
        numSdv=200,
        ):
        
        # ensuring the step is not surrpressed.
        modelObj = mdb.models[modelName]
        stepObj = modelObj.steps[stepName]
        
        if stepObj.suppressed:
            stepObj.resume()
        
        # fixing the outputs viriables of the fieldOutputRequest.
        variables = modelObj.steps[stepName].fieldOutputRequestStates[stepName].variables
        variables = set(variables) | {'LE', 'U', 'UR', 'SDV', 'COORD'}
        
        modelObj.fieldOutputRequests[stepName].setValues(variables=tuple(variables))
        
        copy_model(modelName + '-Backup', modelName)
        
        self.odbNameWithoutOptimizaion = jobName +'.odb'
        self.jobNameWithoutOptimizaion = jobName
        self.modelNameWithoutOptimizaion = modelName
        self.modelName = modelName + '-withEQ'
        self.odbName = jobName + '-withEQ.odb'
        self.jobName = jobName + '-withEQ'
        self.txtFolderName = txtFolderName
        self.modelNameInverse = modelName + '-Inverse'
        self.odbNameInverse = jobName + '-Inverse.odb'
        self.jobNameInverse = jobName + '-Inverse'
        self.stepName = stepName
        self.materialName = materialName
        self.numSdv = numSdv
        self.sectionName = sectionName
        
        self.job_submit = partial(
            job_submit, subroutineFile=subroutineFile, intruptWithError=True
            )
    
    def _set_umat_key(self, modelObj, umat_key):
        # ensure correct material and section definitions
        modelObj.fieldOutputRequests[self.stepName].setValues(position=INTEGRATION_POINTS)
        modelObj.sections[self.sectionName].setValues(material=self.materialName, thickness=None)
        materialObj = modelObj.materials[self.materialName]
        materialObj.Depvar(n = self.numSdv)
        userMaterialObj = materialObj.userMaterial
        props = copy.deepcopy(userMaterialObj.mechanicalConstants)
        props[2] = umat_key
        userMaterialObj.setValues(mechanicalConstants=props)
    
    def _focus_on_first_step(self, modelObj):
        
        for key in modelObj.steps.keys():
            stepObj = modelObj.steps[key]
            if key == 'Initial':
                continue
            if key == self.stepName:
                stepObj.resume()
            else:
                stepObj.suppress()
    
    def _resume_all_steps(self, modelObj):
        
        for key in modelObj.steps.keys():
            stepObj = modelObj.steps[key]
            if key != 'Initial':
                stepObj.resume()
    
    def initialize_params(self, *keys):
        # finding the normalized depth and local plan, and initializing the parameters.
        
        if os.path.exists(self.txtFolderName):
            rmtree(self.txtFolderName)
    
        os.makedirs(self.txtFolderName)
        
        with open(os.path.join(self.txtFolderName, 'DATA.txt'), "w") as f:
            f.write('0\n')
        
        modelObj = mdb.models[self.modelNameWithoutOptimizaion]
        self._set_umat_key(modelObj, 0.0)
        self._focus_on_first_step(modelObj)
        
        self.job_submit(jobName=self.jobNameWithoutOptimizaion)
        
        odb = open_odb(self.odbNameWithoutOptimizaion)
        
        assemblyObj = odb.rootAssembly
        frameObj0 = odb.steps[self.stepName].frames[0]
        
        # get the instance name as it is converted by Abaqus to a general instance for all the model in ODB
        instanceOdbName = odb.rootAssembly.instances.keys()[0]
        
        # get all the nodeset region object
        regionNodeSets = odb.rootAssembly.instances[instanceOdbName].nodeSets
        regionElementSets = odb.rootAssembly.instances[instanceOdbName].elementSets
        
        def _get_coords_of_a_nodeset(nodeSetName):
            valueObj = frameObj0.fieldOutputs['COORD'].getSubset(region=regionNodeSets[nodeSetName],
                                                                 position=NODAL).values
            return np.array([i.data for i in valueObj], dtype=np.float32)
        
        def _point_data_of_a_substructure(cartilageKey = 'LAT_CARTILAGE'):
            
            topCoords = _get_coords_of_a_nodeset('TOP_'+cartilageKey)
            bottomCoords = _get_coords_of_a_nodeset('BOTTOM_'+cartilageKey)
            bottomTree = KDTree(bottomCoords)
            topTree = KDTree(topCoords)
            centralPoint = np.mean(np.concatenate((topCoords, bottomCoords)), axis = 0, dtype=np.float32)
            centralPoint += 1e-8 # just to avoid zero devision
            
            ####### extracting integration point data #######
            pointSetCoords = []
            for fieldName in ['SDV91', 'SDV92', 'SDV93']:
                valueObj = frameObj0.fieldOutputs[fieldName].getSubset(region=regionElementSets[cartilageKey],
                                                                       position=INTEGRATION_POINT).values
                if fieldName == 'SDV91': # only once in the loop:
                    coordsFromSdv = np.array([[i.elementLabel, i.integrationPoint, i.data] for i in valueObj],
                                             dtype=np.float32)
                else:
                    coordsFromSdv = np.array([[i.data] for i in valueObj], dtype=np.float32)
                
                pointSetCoords.append(coordsFromSdv)
            
            temp1, temp2, temp3 = np.hsplit(pointSetCoords[0], 3)
            pointSetCoords = np.array([temp1, temp2, temp3, pointSetCoords[1], pointSetCoords[2]],
                                      dtype=np.float32)
            pointSetCoords = np.squeeze(pointSetCoords).T # deleting redundant axis and transposing
            
            def _get_one_unit(point, numPoints = 1):
                '''A helper function:
                   "point" is the array of points,
                   "numPoints" is the number of nodal points averaged'''
                topDistance , topId = topTree.query(point, numPoints)
                bottomDistance , bottomId = bottomTree.query(point, numPoints)
                
                topDistance = np.mean(topDistance)
                bottomDistance = np.mean(bottomDistance)
                
                totalDistance = topDistance + bottomDistance
                depth = topDistance / totalDistance
                
                topPoint = topCoords[topId]
                bottomPoint = bottomCoords[bottomId]
                unit1 = unit_vector(bottomPoint - topPoint)   # unit of the depth (first important unit)
                vec3 = np.cross(unit1, topPoint - centralPoint)   # perpendicular to the surface of central point and depth vector.
                unit3 = unit_vector(vec3)
                vec2 = np.cross(unit3, unit1) # in the surface of central point and depth vector but perpendicular to the depth vector
                unit2 = unit_vector(vec2) # second important unit
                
                return np.hstack([depth, unit1, unit2, unit3])
            
            # Output is element label, integration point lable, depth, units for each point
            return np.array([np.concatenate([pointData[0:2], _get_one_unit(pointData[2:])])
                             for pointData in pointSetCoords], dtype=np.float32)
        
        def _point_sdv_data(point_data):
            '''It takes point_data, i.e., [element, integration point, depth, and units]
               and returns [element, integration point, and relevant sdvs]'''
            
            element = point_data[0]
            integrationPoint = point_data[1]
            depth = point_data[2]
            unit1 = point_data[3:6]
            unit2 = point_data[6:9]
            unit3 = point_data[9:12]
            
            sdv2=1.4 * (depth ** 2) - 1.1 * depth + 0.59 # FIBER CONSTANT
            sdv3=0.1 + 0.2 * depth # SOLID MATERIAL CONSTANT
            alpha1=[None] * 10
            alpha1[0]=0.005
            alpha1[1]=0.01
            alpha1[2]=0.025
            alpha1[3]=0.035
            alpha1[4]=0.042
            alpha1[5]=0.048
            alpha1[6]=0.053
            alpha1[7]=0.058
            alpha1[8]=0.06
            alpha1[9]=0.06  # the deepest points have index 9 but it is still part its upper leyer (with index 8).
            sdv1=alpha1[int(depth*9)] # GAG CONSTANT
            
            # for primary fibril vectors 1 and 2
            pFibrilVec1 = depth*unit1 + (1.0-depth)*unit2
            pFibrilVec2 = depth*unit1 - (1.0-depth)*unit2
            pFibrilVec1 = unit_vector(pFibrilVec1)
            pFibrilVec2 = unit_vector(pFibrilVec2)
            
            sFibrilVec1 = unit1
            sFibrilVec2 = unit2
            sFibrilVec3 = unit3
            sFibrilVec4 = unit_vector( unit1 + unit2 + unit3)
            sFibrilVec5 = unit_vector(-unit1 + unit2 + unit3)
            sFibrilVec6 = unit_vector( unit1 - unit2 + unit3)
            sFibrilVec7 = unit_vector( unit1 + unit2 - unit3)
            
            return np.hstack([
                element, integrationPoint, sdv1, sdv2, sdv3, pFibrilVec1,
                pFibrilVec2, sFibrilVec1, sFibrilVec2, sFibrilVec3, sFibrilVec4,
                sFibrilVec5, sFibrilVec6, sFibrilVec7
                ])
    	
        data = []
        for key in keys:
            subData = _point_data_of_a_substructure(key)
            
            # Depth is normalized within [0,1] (as it is already around 0.1 to 0.9)
            subData[:,2] = (subData[:,2] - np.min(subData[:,2]))/np.ptp(subData[:,2])
            
            data.append(
                np.array([_point_sdv_data(point_data) for point_data in subData], dtype=np.float32)
                )
        
        data = np.concatenate(data, axis = 0)
        
        elementArray = np.unique(data[:,0])
        
        for element in elementArray:
            np.savetxt(os.path.join(self.txtFolderName, '%i.txt'%(element)),
                       data[data[:,0] == element][:,1:],
                       delimiter=',',
                       fmt='%10.7f')
        
        with open(os.path.join(self.txtFolderName, 'DATA.txt'), "w") as f:
            f.write('1\n')
        
        self._resume_all_steps(modelObj)
        self._set_umat_key(modelObj, 1.0)
        
        print 'INITIALIZATION IS COMPLETED! \n'
    
    def run_prestress_optimizer(
            self,
            key,
            sdvList=['SDV%s'%(i) for i in (range(1,4) + range(16,43))],
            zeta=1.0, 
            breakPoint=0, 
            errorLimit=1e-3,
            maxiteration=50,
            eta=4.0,
            ):
        # main pre-stress function
        zeta = float(zeta) # avoiding problem with integer division
        modelWithoutOptimizaionObj = mdb.models[self.modelNameWithoutOptimizaion]
        self._set_umat_key(modelWithoutOptimizaionObj, 1.0)
        self._focus_on_first_step(modelWithoutOptimizaionObj)
        
        # ensure the correct naming of all cartilage sets.
        nodeSet = key + '_NODES'
        elementSet = key + '_ELEMENTS'
        
        # creating helper models, jobs, sets, and BCs
        rootAssemblyObj = modelWithoutOptimizaionObj.rootAssembly
        nodeObj = rootAssemblyObj.sets[nodeSet].nodes
        nodeNameList = ['TEMP-%s'%(i) for i in xrange(1, len(nodeObj)+1)]
        
        for i in xrange(len(nodeNameList)):
            rootAssemblyObj.Set(nodes=(nodeObj[i:i+1],), name=nodeNameList[i])
        
        for name in [self.modelName, self.modelNameInverse]:
            copy_model(name, self.modelNameWithoutOptimizaion)
        
        for name in [self.jobName, self.jobNameInverse]:
            copy_job(name, self.jobNameWithoutOptimizaion)
        
        mdb.jobs[self.jobNameInverse].setValues(model=self.modelNameInverse)
        mdb.jobs[self.jobName].setValues(model=self.modelName)
        steps = mdb.models[self.modelName].steps
        stepsWithoutEQ = modelWithoutOptimizaionObj.steps
        
        BCobj = mdb.models[self.modelNameInverse].boundaryConditions
        BCkeys = BCobj.keys()
        
        for i in BCkeys:
            BCobj[i].suppress()
        
        modelObjTemp = mdb.models[self.modelNameInverse]
        
        for i in xrange(len(nodeNameList)):
            modelObjTemp.VelocityBC(name=nodeNameList[i], # bc name and node names are the same.
                                    createStepName='Initial',
                                    region=modelObjTemp.rootAssembly.sets[nodeNameList[i]],
                                    v1=SET,
                                    v2=SET,
                                    v3=SET,
                                    amplitude=UNSET,
                                    localCsys=None,
                                    distributionType=UNIFORM,
                                    fieldName='')
        
        # some helper functions
        def _integration_points_values(odb, parameters=['SDV3'], frameNum=-1):
            
            instanceOdbName = odb.rootAssembly.instances.keys()[0] # refers to all
            regionElementSets = odb.rootAssembly.instances[instanceOdbName].elementSets
            frameObj = odb.steps[self.stepName].frames[frameNum]
            
            return [np.ravel([[item.data, item.elementLabel, item.integrationPoint]
                    for item in frameObj.fieldOutputs[sdvName].getSubset(region=
                                regionElementSets[elementSet], position=
                                INTEGRATION_POINT).values])
                    for sdvName in parameters]
        
        def _extract_coords_values(odb, frameNum = -1):
            instanceOdbName = odb.rootAssembly.instances.keys()[0] # refers to all
            regionNodeSets = odb.rootAssembly.instances[instanceOdbName].nodeSets
            frameObj = odb.steps[self.stepName].frames[frameNum]
            
            # a list of [[node label], [node coord 1, node coord 2, ...], ...],
            # then flatten the list.
            return [nodeDataElement
                    for nodeName in nodeNameList
                    for nodeData in ([nodeName],
                                     frameObj.fieldOutputs['COORD'].getSubset(region
                                        =regionNodeSets[nodeName], position=NODAL
                                        ).values[0].data.tolist()
                                    )
                    for nodeDataElement in nodeData]
        
        def _edit_node_by_offset(displacementFromInitial, modelName):
            rootAssemblyObj = mdb.models[modelName].rootAssembly
            num = 0
            for i in displacementFromInitial:
                num += 1
                if num % 4 == 1:
                    nodeLabel = i
                elif num % 4 == 2:
                    u1 = -i
                elif num % 4 == 3:
                    u2 = -i
                elif num % 4 == 0:
                    u3 = -i
                    rootAssemblyObj.editNode(nodes=rootAssemblyObj.sets[nodeLabel].nodes[0],
	                                         offset1=u1,
	                                         offset2=u2,
	                                         offset3=u3,
	                                         projectToGeometry=OFF)
        
        def _inverse_run(displacementFromInitial):
            ModelObjTemp = mdb.models[self.modelNameInverse]
            bcStateObj = ModelObjTemp.steps[self.stepName].boundaryConditionStates
            num = 0
            for i in displacementFromInitial:
                num += 1
                if num % 4 == 1:
                    bcLabel = i
                elif num % 4 == 2:
                    # v1 = bcStateObj[bcLabel].v1-i
                    v1 = -i
                elif num % 4 == 3:
                    # v2 = bcStateObj[bcLabel].v2-i
                    v2 = -i
                elif num % 4 == 0:
                    # v3 = bcStateObj[bcLabel].v3-i
                    v3 = -i
                    ModelObjTemp.boundaryConditions[bcLabel].setValuesInStep(stepName=self.stepName,
                                                                             v1=v1,
                                                                             v2=v2,
                                                                             v3=v3)
            
            with open(os.path.join(self.txtFolderName, 'DATA.txt'), "w") as f:
                f.write('-1\n')
            
            self.job_submit(self.jobNameInverse)
            
            with open(os.path.join(self.txtFolderName, 'DATA.txt'), "w") as f:
                f.write('1\n')
        
        def _new_SDV_in_fortran(newSdvData):
            IntegrationPointArray = np.unique(newSdvData[0][2::3]) # [1.0, 2.0, 3.0, 4.0, ...]
            IntegrationCount = IntegrationPointArray[-1].max() # number of all integration points
            _, ElementsIdx = np.unique(newSdvData[0][1::3], return_index=True)  # e.g., [0L, 27L, 54L, ...]
            elementCount = ElementsIdx[1] # all elements have the same number of nodes
            
            for ElementsIdxItem in ElementsIdx:
                elementIdxArray = ElementsIdxItem*3 + 1 + np.arange(0, 3*IntegrationCount, 3, dtype=int)
                ValueIdxArray = elementIdxArray - 1
                IntegrationPointIdxArray = elementIdxArray + 1
                elementItem = newSdvData[0][elementIdxArray[0]]
                sdvDataList = np.concatenate((IntegrationPointArray.reshape(1,-1),
                                              np.take(newSdvData, ValueIdxArray, axis = -1)))
                np.savetxt(os.path.join(self.txtFolderName, '%i.txt'%(elementItem)),
                           sdvDataList.T,
                           delimiter=',',
                           fmt='%10.7f')
        
        if self.job_submit(self.jobNameWithoutOptimizaion) == 'ABORTED':
            raise Exception('ERROR! TOTALLY UNSTABLE MODEL')
        
        odbObjWithoutOptimization = open_odb(self.odbNameWithoutOptimizaion)
        initialNodalCoords = _extract_coords_values(odbObjWithoutOptimization, 0)
        copyfile(self.odbNameWithoutOptimizaion, self.odbName)
        
        def _calculate_r_u(zeta):
            odb = open_odb(self.odbName)
            newNodalCoords = _extract_coords_values(odb, -1)
            close_odb(self.odbName)
            
            displacementFromInitial = [(newNodalCoords[i]-initialNodalCoords[i])*zeta
                                       if i % 4 != 0
                                       else newNodalCoords[i] # just the label
                                       for i in xrange(len(newNodalCoords))]
            
            newError = np.linalg.norm(np.array(
                [displacementFromInitial[i] for i in xrange(len(displacementFromInitial)) if i % 4 != 0]
                )) / zeta
            
            return newError, displacementFromInitial
            
        newError, displacementFromInitial = _calculate_r_u(zeta)
        
        self.optimizerStatus = {'step': [1], 'error': [newError], 'zeta': [zeta]}
        
        failed=False
        iterationNumber = 1
        
        newSdvDataBackup = _integration_points_values(odbObjWithoutOptimization, sdvList, 0)
        close_odb(self.odbNameWithoutOptimizaion)
        previousError = newError
        
        while True:
            
            if iterationNumber == maxiteration:
                failed = True
                break
            else:
                iterationNumber += 1
            
            copy_model(self.modelName + '-Backup', self.modelName)
            _edit_node_by_offset(displacementFromInitial, self.modelName)
            copy_model(self.modelNameInverse + '-Backup', self.modelNameInverse)
            _inverse_run(displacementFromInitial)
            _edit_node_by_offset(displacementFromInitial, self.modelNameInverse)
            odbInverse = open_odb(self.odbNameInverse)
            newSdvData = _integration_points_values(odbInverse, sdvList, -1)
            close_odb(self.odbNameInverse)
            _new_SDV_in_fortran(newSdvData)
            
            
            if self.job_submit(self.jobName) != 'ABORTED':
                newError, displacementFromInitial = _calculate_r_u(zeta)
                if previousError < newError:
                    successfulStep = False
                else:
                    successfulStep = True
                
            else:
                successfulStep = False
            
            print '\n** #STEP: %s | ERROR: %s | ZETA: %s **\n' % (iterationNumber, newError, zeta)
            
            self.optimizerStatus['step'].append(iterationNumber)
            self.optimizerStatus['error'].append(newError)
            self.optimizerStatus['zeta'].append(zeta)
            
            if errorLimit > newError:
                failed = False
                break
            
            if successfulStep == True:
                newSdvDataBackup = copy.deepcopy(newSdvData)
                previousError = newError
            
            else:
                _new_SDV_in_fortran(newSdvDataBackup)
                zeta = zeta/eta
                
                if zeta < 0.0001:
                    failed = True
                    break
                
                copy_model(self.modelName, self.modelName + '-Backup')
                copy_model(self.modelNameInverse, self.modelNameInverse + '-Backup')
        
        # finish_optimization
        modelsObj = mdb.models
        self._resume_all_steps(modelsObj[self.modelName])
        self._resume_all_steps(modelsObj[self.modelNameWithoutOptimizaion])
        
        del modelsObj[self.modelName+'-Backup']
        del modelsObj[self.modelNameInverse]
        del modelsObj[self.modelNameInverse+'-Backup']
        del mdb.jobs[self.jobNameInverse]
        
        for tempSetName in nodeNameList:
            for modelName in [self.modelName, self.modelNameWithoutOptimizaion]:
                del modelsObj[modelName].rootAssembly.sets[tempSetName]
        
        if failed == True:
            print 'PRE_STRESSING HAS NOT BEEN FULLY CONVERGED! \n'
            return 'ABORTED'
        else:
            print 'PRE_STRESSING HAS BEEN COMPLETED! \n'
            return 'COMPLETED'




class file_io():
    
    def __init__(self, mainDir, subDir):
        
        writePath = os.path.join(mainDir, subDir)
        
        if os.path.exists(writePath):
            rmtree(writePath)
            os.makedirs(writePath)
        else:
            os.makedirs(writePath)
        
        self.new_address = lambda filePath: os.path.join(writePath, filePath)
        
        self.reset()
    
    def reset(self):
        self._sample_num = 1
        self.mataOfArrays, self.metaOfDicts, self._seen_keys = {}, {}, {}
    
    def record_csv(self, data_dict):
        for key in data_dict:
            with open(self.new_address(key) + ".csv", "w") as f:
                np.savetxt(f, data_dict[key], delimiter=",")
    
    def _new_sample(self):
        self._sample_num += 1
        self._seen_keys = {key: False for key in self._seen_keys}
    
    def store_data(self, key, array):
        # The input array can be a single array or a dictionary of arrays, with a shape
        # [num nodes or edges or 1, num frames or 1, ...], where only num frames can be varied.
        
        if self._sample_num == 1:
            # if a key is repeated make new sample otherwise add that key to self._seen_keys.
            if key in self._seen_keys:
                self._new_sample()
            else:
                self._seen_keys[key] = True
        else:
            if key in self._seen_keys:
                # in a step, if a key was seen, make a new sample
                # but before that verify the other keys were already seen.
                if self._seen_keys[key] == False:
                    self._seen_keys[key] = True
                else:
                    if False in self._seen_keys.values():
                        raise Exception('Error: file_io got an already seen key (' + key +
                                        ') before seeing the others in a step num > 1.')
                    else:
                        self._new_sample()
            else:
                raise Exception('Error: file_io got a new key ('+ key +
				                '), which was not seen before in a step num > 1.')
        
        def _store_single_array(key, array):
            # save array and update mataOfArrays, 
            assert array.ndim > 1, Exception('array.ndim in store_data() should be more than 2.')
            
            dtype = array.dtype
            
            if dtype == np.float64:
                array = array.astype(np.float32)
            elif dtype == np.int64:
                array = array.astype(np.int32)
            
            dtypeName = array.dtype.name
            flat_array = array.reshape(-1)
            
            if key not in self.mataOfArrays:
                
                shape = array.shape
                
                if shape[1] == 1:
                    numFrames = 1L
                else:
                    numFrames = -1L
                
                shape = (shape[0], numFrames, ) + shape[2:]
                
                # update meta
                self.mataOfArrays[key] = {"shape": shape, "ndim": array.ndim, "dtype": dtypeName}
                openMode = 'wb'
            else:
                openMode = 'ab'
            
            # self.record({key: array})
            with open(self.new_address(key)+'.npy', openMode) as f:
                np.save(f, flat_array)
        
        if type(array) == dict:
            # store keys as long integers that can be dumped as JSON.
            self.metaOfDicts[key] = [long(k) for k in array.keys()]
            
            for k, val in array.items():
                _store_single_array(key+"_"+str(k), val)
            
        else:
            _store_single_array(key, array)
    
    def store_meta(self, **additionalMetaData):
        # adding possible addtional meta data and storing them
        meta = {
            "arrays": self.mataOfArrays,
            "dicts": self.metaOfDicts,
            "array_names": self.mataOfArrays.keys(),
            "total_num_samples": self._sample_num
            }
        meta.update(additionalMetaData)
        
        with open(self.new_address("meta.json"), "w") as f:
            json.dump(meta, f, indent=2, separators=(", ", ": "))


class timer_tools():
    # set different timers, each can repeatedly.
    def __init__(self, *args):
        
        assert len(args) != 0, Exception(
            'Error: Feed an arbitrary input string to the timer contructor, e.g., "fea".'
            )
        self.times = {key: [] for key in args}
        self.keys = args
        # for the first start
        self.isStarted = False
        self.delta = 0
    
    def start(self, key):
        assert ~self.isStarted, Exception('Error: The timer has already been started.')
        self.t0 = time.time()
        self.isStarted = True
    
    def pause(self, key):
        assert self.isStarted, Exception('Error: The timer shoud be started to be paused.')
        self.delta += time.time() - self.t0
        self.isStarted = False
    
    def stop(self, key):
        assert self.isStarted, Exception('Error: The timer shoud be started to be stopped.')
        self.pause(key)
        self.times[key].append(self.delta)
        self.delta = 0 # for the next possible start
    
    def ignore(self, key):
        self.isStarted = False
    
    def get(self, with_pre=True):
        # use this method to get all the time measures in the end.
        if with_pre == True:
            time_pre = 'time_'
        else:
            time_pre = ''
        
        return {time_pre+key: np.array(self.times[key], dtype="float32") for key in self.keys}


class field_tools():
    # tools to make new calculated field from the obtained results
    def __init__(
        self,
        odb,
        instance,
        nodeSetKey,
        fileIo,
        excludingSteps=set(),
        position=NODAL,
        jobStatus='COMPLETED',
        mainName=''
        ):
        assert type(excludingSteps) == set, Exception('Error: excludingSteps argument contains set!')
        stepsObj = odb.steps
        stepskeys = stepsObj.keys()
        
        # some of the excludingSteps might not exists which are eleminated
        excludingSteps = excludingSteps.intersection(set(stepskeys))
        baseTime = max([0]+[stepsObj[k].totalTime+stepsObj[k].timePeriod for k in excludingSteps])
        
        frames, times, relativeTimesDict = [], [], OrderedDict()
        
        for step in stepsObj.values():
            stepName = step.name
            
            if stepName not in excludingSteps:
                initialTime = step.totalTime - baseTime
                # step.frames is an ODB Sequence does not support slicing, thus covert it to list.
                stepFrames = [frame for frame in step.frames]
                
                if initialTime != 0.0:
                    stepFrames = stepFrames[1:]
                
                relativeTimes = np.array([frame.frameValue for frame in stepFrames])
                times.append(initialTime + relativeTimes)
                frames.extend(stepFrames)
                
                if len(relativeTimes) > 1:
                    relativeTimesDict[stepName] = np.array(relativeTimes).reshape(-1, 1)
        
        times = np.concatenate(times, axis=0)
        
        # if job was aborted, the last frame was not converged and should be deleted
        if jobStatus == 'ABORTED':
            lastStepName = stepskeys[-1]
            frames = frames[:-1]
            times = times[:-1]
            
            if lastStepName in relativeTimesDict:
                relativeTimesDict[lastStepName] = relativeTimesDict[lastStepName][:-1]
        
        
        self.frames = frames
        self.times = times
        self.relativeTimes = relativeTimesDict
        self.region = instance.nodeSets[nodeSetKey]
        self.instance = instance
        self.position = position
        self.odb = odb
        
        if mainName != '':
            self.mainName = mainName + '_'
        else:
            self.mainName = mainName
        
        self.store_data = fileIo.store_data
    
    def record_trajectory_length(self):
        # record total simulation time of each frame
        self.store_data('trajectory_length', np.array(len(self.frames)).reshape([1, 1, 1]))
    
    def record_time(self):
        # record total simulation time of each frame
        self.store_data(self.mainName+'time', self.times.reshape([1, -1, 1]))
    
    def get_values(self, field, frame):
        return frame.fieldOutputs[field].getSubset(region=self.region, position=self.position).values
    
    def extract_data(self, field, component='data', frame_first=False):
        
        data = np.array([
            [getattr(value, component) for value in self.get_values(field, frame)] for frame in self.frames
        ])
        
        if data.ndim == 2:
            data = np.expand_dims(data, axis = -1)
        
        if frame_first==False:
            data = data.transpose([1, 0, 2])
        
        return data
    
    def get_sample_value_obj(self, field, frameNum = -1):
        return self.get_values(field, self.frames[frameNum])
    
    def show_components(self, field, frameNum = -1):
        components = dir(self.get_sample_value_obj(field)[0]) 
        return [component for component in components if component[:2] != '__']
    
    def make_tensor_from_sdvs(self, sdvNumsList, name):
        
        numComponents = len(sdvNumsList)
        
        if numComponents == 6:
            validInvariants = (MISES, TRESCA, PRESS, INV3, MAX_PRINCIPAL, MID_PRINCIPAL, MIN_PRINCIPAL,)
            tensor_type=TENSOR_3D_FULL
        elif numComponents in [3 , 2]:
            validInvariants = (MAGNITUDE, )
            tensor_type=VECTOR
        else:
            raise Exception('The "make_tensor_from_sdvs" function does not support this tensor type')
        
        data = [self.extract_data('SDV'+str(sdvNum), frame_first=True) for sdvNum in sdvNumsList]
        data = np.concatenate(data, axis = -1)
        nodeLabels = [value.nodeLabel for value in self.get_sample_value_obj('SDV'+str(sdvNumsList[0]))]
        
        # frame.frameId might not be sequential, so avoid it, instead I use:
        num = 0
        
        for frame in self.frames:
            custom_field = frame.FieldOutput(
                name=name, description=name, type=tensor_type, 
                )
            custom_field.addData(
                position=NODAL, instance=self.instance, labels=nodeLabels, data=data[num]
                )
            num += 1
        
    def record_field(
        self, field, name, invariantsList=None, relative=False, dtype='float32', return_data=False
	    ):
        
        data = self.extract_data(field)
        
        if relative == True:
            data -= data[:,0:1]
        
        name = self.mainName + name
        
        self.store_data(name, data.astype(dtype))
        
        if return_data==True:
            data_dict = {name: data}
        
        if invariantsList != None:
            name += "_"
            
            for invariant in invariantsList:
                data = self.extract_data(field, invariant)
                
                if relative == True:
                    data -= data[:,0:1]
                
                invariantName = name + invariant
                self.store_data(invariantName, data.astype(dtype))
                
                if return_data==True:
                    data_dict[invariantName] = data
        
        if return_data==True:
            if len(data_dict.keys()) == 1: # when has one key return only its value.
                return list(data_dict.values())[0]
            else:
                return data_dict
    
    def get_coords(self):
        # getting nodal coordinates (and storing them), nodal labels, and shortest edge distance
        labels = self.extract_data('COORD', 'nodeLabel')
        coords = self.extract_data('COORD', 'data')
        x0 = coords[:, 0]
        distance_matrix = np.sqrt(np.sum((x0[:, None] - x0)**2, axis=-1))
        np.fill_diagonal(distance_matrix, np.inf)
        shortest_edge = np.min(distance_matrix)
        return {'data': coords, 'labels': np.squeeze(labels), 'shortest_edge': shortest_edge}


class mesh_tools():
    # tools to get the mesh and nodal information, used to obtain "alowable edges"
    def __init__(self, instance, nodeSetKey, fileIo, mainName=''):
        self.instance = instance
        # renumbering relevant nodes like [0, 1, 2, ...]
        nodeLabels = self.get_nodal_indeces(nodeSetKey, correctLabels=False)
        self.uniqueLabels = np.unique(nodeLabels)
        self.nodeIndeces = self.correct_node_labels(nodeLabels)
        self.store_data = fileIo.store_data
        
        if mainName != '':
            self.mainName = mainName + '_'
        else:
            self.mainName = mainName
    
    def get_nodal_indeces(self, nodeSetKey, correctLabels=True):
        nodesObj = self.instance.nodeSets[nodeSetKey].nodes
        nodes = np.array([node.label for node in nodesObj], dtype=int)
        if correctLabels==True:
            return self.correct_node_labels(nodes)
        else:
            return nodes
    
    def correct_node_labels(self, labels):
        return np.searchsorted(self.uniqueLabels, labels)
    
    def record_k_nearest(self, ref_pos, augemented_pos, k=10):
        assert ref_pos.ndim >= 3 and augemented_pos.ndim >= 3, Exception('input positions'
            'should be of dimension 3 or more. You missed probably the frame axis.')
        tree = KDTree(ref_pos[:, 0])
        _, idx = tree.query(augemented_pos[:, 0], k=k)
        # the index of node labels might be different from the tree indeces.
        idx = np.expand_dims(self.nodeIndeces[idx], axis=1)
        self.store_data(self.mainName+"nearest_nodes", idx,)
    
    def get_topology(self, elementSetKey, abq_node_numbering, record=True):
        
        # firstly, get the edge idx in each cell
        node_numbering_idx = abq_node_numbering - 1
        edge_numbering_idx = np.concatenate(
            [node_numbering_idx, node_numbering_idx[:,::-1]], axis = 0
            )
        
        # secondly, get cell data
        elements = self.instance.elementSets[elementSetKey].elements
        cells = np.array([element.connectivity for element in elements], dtype = int)
        
        # now fix the node labels as they might not be sequential or start from 0
        cells = self.correct_node_labels(cells)
        
        # getting topology
        edges = np.unique(
            cells[:, edge_numbering_idx].reshape(-1, 2),
            axis=0
            )
        senders, recievers = edges[:, 0], edges[:, 1]
        
        
        # senders_occurence shows the number of repetitions of each idx,
        # showing the number of edges for each node.
        senders_occurence = np.bincount(senders)[senders]
        # edge_idx_dict is a dic with keys showing the number of edges and
        # values showing the idx of each edge (not the nodes')
        edge_idx_dict = {
            key: np.squeeze(np.argwhere(senders_occurence == key))
            for key in np.unique(senders_occurence)
            }
        keys = sorted(edge_idx_dict.keys())
        # now, edge_idx_dict is used to group senders/receivers depending
        # on their number of connecting edges to each sender
        senders_dict = {key: senders[edge_idx_dict[key]] for key in keys}
        receivers_dict = {key: recievers[edge_idx_dict[key]] for key in keys}
        
        # storing topology
        if record == True:
            edges_dict = {}
            for key in keys:
                edges_dict[key] = np.stack([senders_dict[key], receivers_dict[key]], axis=-1)
                edges_dict[key] = np.expand_dims(edges_dict[key], axis=1)
            
            self.store_data(self.mainName+"edges", edges_dict,)
        
        return senders_dict, receivers_dict
    
    def record_augmented_topology(self, senders_dict, receivers_dict, unchangeableNodesKey, pos):
        
        unchangeableNodes = self.get_nodal_indeces(unchangeableNodesKey)
        keys = sorted(senders_dict.keys())
        # finding multiplier_dict that can be 0, 1, 2 (e.g., 1 refers to half the edge)
        multiplier_dict = {}
        previous_keys = []
        
        for key in keys:
            
            args_valid_edges = ~np.isin(senders_dict[key], unchangeableNodes)
            # multiplier_dict[key] = np.where(args, 0, 1)
            
            if np.all(~args_valid_edges):
                multiplier_dict[key] = np.zeros_like(args_valid_edges, dtype=int)
            else:
                args_valid_edges *= np.isin(receivers_dict[key], senders_dict[key])
                multiplier_dict[key] = np.where(args_valid_edges, 1, 0)
                
                for previous_key in previous_keys:
                    args = np.isin(receivers_dict[key], senders_dict[previous_key])
                    multiplier_dict[key] += np.where(args, 2, 0)
                    assert (3 not in multiplier_dict[key]), Exception('You have '
                        'multiplier(s) 3 in multiplier_dict[key] that should be fixed.')
                
            previous_keys.append(key)
        
        # deleting invalid edges (corresponding to multiplier 0) and fixing shapes
        for key in keys:
            if multiplier_dict[key].sum() == 0:
                del senders_dict[key], receivers_dict[key], multiplier_dict[key]
            else:
                valid_edges = (multiplier_dict[key] != 0)
                senders_dict[key] = senders_dict[key][valid_edges]
                _, counts = np.unique(senders_dict[key], return_counts=True)
                dim = np.unique(counts)
                assert dim.shape[0] == 1, Exception('Reshaping is not possible.')
                senders_dict[key] = senders_dict[key].reshape(-1, dim[0])
                assert np.all(senders_dict[key] == senders_dict[key][:, :1]), Exception(
                    'Senders of each nodes are not the same as the corresponding node.'
                    )
                # similarly
                multiplier_dict[key] = multiplier_dict[key][valid_edges].reshape(-1, dim[0])
                receivers_dict[key] = receivers_dict[key][valid_edges].reshape(-1, dim[0])
        
        # correcting the keys as they refer to the shape, which is changed now
        correct_key = lambda array: {value.shape[1]: value for value in array.values()}
        multiplier_dict = correct_key(multiplier_dict)
        senders_dict = correct_key(senders_dict)
        receivers_dict = correct_key(receivers_dict)
        
        # getting x0 and dx (for augmentation)
        senders = {0: unchangeableNodes.reshape([-1, 1, 1])}
        x0 = {0: np.expand_dims(pos[unchangeableNodes], axis=-2)}
        dx = {0: np.zeros_like(x0[0])}
        receivers = {0: np.full(senders[0].shape, -1)}
        
        for key in senders_dict:
            x0[key] = pos[senders_dict[key]].transpose([0, 2, 1, 3])
            x1 = pos[receivers_dict[key]].transpose([0, 2, 1, 3])
            multiplier = multiplier_dict[key][:, np.newaxis, :, np.newaxis]
            dx[key] = (x1 - x0[key]) * (multiplier)/2.0
            # senders_dict is like [[1, 1], [5, 5] , ...] that should be stored like [1, 5, ...]
            senders[key] = np.expand_dims(senders_dict[key][:,0:1], axis=1)
            receivers[key] = np.expand_dims(receivers_dict[key], axis=1)
        
        # storing them all
        self.store_data(self.mainName+"augmented_senders", senders)
        self.store_data(self.mainName+"augmented_receivers", receivers)
        self.store_data(self.mainName+"augmented_dx", dx)
        self.store_data(self.mainName+"augmented_x0", x0)
    
    def record_node_type(self, nodeSubsetDict):
        # combine nodal labels and record them
        node_type = []
        mask = []
        for setId in nodeSubsetDict:
            labels = self.get_nodal_indeces(nodeSubsetDict[setId])
            mask.append(np.isin(self.nodeIndeces, labels))
            node_type.append(np.where(mask[-1], 1, 0).reshape(-1, 1))
        
        mask_others = np.sum(mask, axis=0, dtype='bool')
        node_type_others = [np.where(mask_others, 0, 1).reshape(-1,1)] 
        node_type = np.concatenate(node_type_others + node_type, axis=-1)
        assert np.all(np.count_nonzero(node_type, axis=-1) == 1), Exception(
            'The one-hot array has several ones. Verify the nodes are not coincided.'
            )
        node_type = np.expand_dims(node_type, axis=1)
        self.store_data(self.mainName+"node_type", node_type)


class amplitude_tools():
    # we used this class to further randomize the boundary conditions by
    # applying amplitude data and interpolating them
    def __init__(self, amp, timePeriod):
        
        self.amplitudeName = 'amplitude'
        t = np.linspace(0, timePeriod, amp.shape[0])
        cs = CubicSpline(t, amp)
        self.t = np.linspace(0, timePeriod, 100)
        y = cs(self.t)
        # scaling
        y_min = np.min(y)
        y_max = np.max(y)
        self.amp = (y - y_min) / (y_max - y_min + (1e-8))
    
    def cut(self, timePeriod):
        
        amp = self.amp
        t = self.t
        
        if np.isclose(t[-1], timePeriod):
            t[-1] = timePeriod
            
        elif timePeriod <= t[-1]:
            idx = np.searchsorted(t, timePeriod, side='right')
            gradient = (amp[idx-1] - amp[idx])/(t[idx-1] - t[idx])
            amp[idx-1] += gradient*(timePeriod - t[idx-1])
            t[idx] = timePeriod
            amp = amp[:idx+1]
            t = t[:idx+1]
            
        else:
            raise Exception('Error: timePeriod >> t[-1] in the cut method of amplitude_tools')
        
        if np.isclose(t[-1], t[-2]):
            t = np.delete(t, -2)
            amp = np.delete(amp, -2)
        
        self.t = t
        self.amp = amp
    
    def set(self, modelObj):
        
        data = np.stack([self.t, self.amp], axis = -1)
        modelObj.amplitudes[self.amplitudeName].setValues(timeSpan=STEP, data=data)
    
    def plot(self):
        # just for verification
        plt.plot(self.t, self.amp)
        plt.xlabel('Time (t)')
        plt.ylabel('Amplitude (-)')
        plt.show()


caeFile = 'fea.cae'
subFolder = 'DA'

dirAddress = os.path.join('C:\\Temp\\', subFolder)
os.chdir(dirAddress)
openMdb(pathName = caeFile)

invariantsList = [
    'inv3',
    'press',
    'mises',
    'minPrincipal',
    'midPrincipal',
    'maxPrincipal'
    ]
nodeSetsDict = {
    1: 'TOP_FEMUR_CARTILAGE',
    2: 'BOTTOM_FEMUR_CARTILAGE',
    }
cartilageSet= 'ARTICULAR_CARTILAGE_NODES'
elementSetKey = 'ARTICULAR_CARTILAGE_ELEMENTS'
unchangeableNodesKey = 'UNCHANGEABLE_NODES'
stepName = 'LOAD'
stepParams = {
    'matrixSolver': DIRECT,
    'matrixStorage': UNSYMMETRIC,
    'amplitude': RAMP,
    'maxNumInc': 1000,
    'minInc': 1e-6,
    'extrapolation': PARABOLIC,
    }
fieldOutputParams = {
    'name': stepName,
    'createStepName': stepName,
    'position': AVERAGED_AT_NODES,
    }
abq_node_numbering = np.array(
    [[1,2], [2,3], [3,4], [4,1], [5,6], [6,7], [7,8], [8,5], [1,5], [2,6], [4,8], [3,7]]
    )
eqStepName = 'EQ'
createStepParams = {
    'name': stepName,
    'previous': eqStepName,
    'maintainAttributes': True,
    }
fieldOutputParams = {
    'name': stepName,
    'createStepName': stepName,
    'position': AVERAGED_AT_NODES,
    }