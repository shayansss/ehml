from fea_core import *

# definition and initialization
jobName = modelName = 'knee'


with open('results.pkl', 'rb') as file:
    pmse_dict = pickle.load(file)


def visualize(random_string):
    # get the visualization of the random_string field and store it into a file
    odb = open_odb('knee.odb')
    viewportObj = session.viewports['Viewport: 1']
    viewportObj.setValues(displayedObject=odb)
    viewportObj.makeCurrent()
    
    leaf = dgo.LeafFromOdbElementMaterials(elementMaterials=('CAR', ))
    viewportObj.odbDisplay.displayGroup.replace(leaf=leaf)
    viewportObj.odbDisplay.setFrame(step=1, frame=0)
    viewportObj.odbDisplay.display.setValues(plotState=CONTOURS_ON_DEF)
    
    viewportObj.odbDisplay.setPrimaryVariable(
        variableLabel=random_string, outputPosition=NODAL,
        )
    viewportObj.viewportAnnotationOptions.setValues(
        triad=OFF, legend=OFF, title=OFF, state=OFF, annotations=OFF, compass=OFF
        )
    viewportObj.view.fitView()
    viewportObj.odbDisplay.contourOptions.setValues(
        minValue=0, maxValue=2, maxAutoCompute=OFF,
        minAutoCompute=OFF, outsideLimitsAboveColor='#FF0000'
        )
    
    session.pngOptions.setValues(imageSize=(2000, 1123))
    session.printOptions.setValues(vpDecorations=OFF, reduceColors=False)
    session.printToFile(
        fileName='result.png', format=PNG, canvasObjects=(viewportObj, )
        )
    close_odb('knee.odb')


def generate_random_string(length=8):
    # Generate a random string of fixed length
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(length))


def make_field_with_random_name(results):
    # generated abaqus field from input results
    random_string = generate_random_string(10)
    
    cartilageSet= 'ARTICULAR_CARTILAGE_NODES'
    eqStepName = 'EQ'
    fileIo = file_io("gnn_datasets", modelName)
    field_tools_params = {'nodeSetKey': cartilageSet, 'fileIo': fileIo, 'excludingSteps': {eqStepName}}
    
    odb = open_odb('knee.odb', readOnly=False)
    fieldTool = field_tools(
        odb, odb.rootAssembly.instances['PART-1-1'], jobStatus='COMPLETED', **field_tools_params
        )
    
    nodeLabels = [value.nodeLabel for value in fieldTool.get_sample_value_obj('S')]
    results = np.mean(np.array(results).reshape(len(nodeLabels), -1), axis=-1)[:, np.newaxis]
    frame = fieldTool.frames[0]
    custom_field = frame.FieldOutput(name=random_string, description='PMSE', type=SCALAR)
    custom_field.addData(
        position=NODAL, instance=fieldTool.instance, labels=nodeLabels, data=results
        )
    close_odb('knee.odb', saveOdb = True)
    return random_string


keys = [u'static_subgraph', u'top_10', u'dynamic_subgraph', u'static_weight',
        u'top_1', u'dynamic_weight', u'mse']

column_1 = {k: [] for k in keys}
column_2 = {k: [] for k in keys}
column_3 = {k: [] for k in keys}
column_4 = {k: [] for k in keys}

for k1 in pmse_dict:
    if k1[-1] == '2':
        for k2 in pmse_dict[k1]:
            if k2 == 'With augmentation':
                for k3 in pmse_dict[k1][k2]:
                    for k4 in pmse_dict[k1][k2][k3]:
                        column_1[k4].append(pmse_dict[k1][k2][k3][k4])
            else:
                for k3 in pmse_dict[k1][k2]:
                    for k4 in pmse_dict[k1][k2][k3]:
                        column_2[k4].append(pmse_dict[k1][k2][k3][k4])
    else:
        for k2 in pmse_dict[k1]:
            if k2 == 'With augmentation':
                for k3 in pmse_dict[k1][k2]:
                    for k4 in pmse_dict[k1][k2][k3]:
                        column_3[k4].append(pmse_dict[k1][k2][k3][k4])
            else:
                for k3 in pmse_dict[k1][k2]:
                    for k4 in pmse_dict[k1][k2][k3]:
                        column_4[k4].append(pmse_dict[k1][k2][k3][k4])

for k in keys:
    column_1[k] = np.mean(np.array(column_1[k]).transpose(), axis=-1, keepdims=True)
    column_2[k] = np.mean(np.array(column_2[k]).transpose(), axis=-1, keepdims=True)
    column_3[k] = np.mean(np.array(column_3[k]).transpose(), axis=-1, keepdims=True)
    column_4[k] = np.mean(np.array(column_4[k]).transpose(), axis=-1, keepdims=True)

def get_img(column, loss = 'mse'):
    results = column[loss]
    random_string = make_field_with_random_name(results)
    visualize(random_string)
    start_row1, end_row1, start_col1, end_col1 = 150, -150, 550, -500
    img = mpimg.imread('result.png')[start_row1:end_row1, start_col1:end_col1]
    return img


losses = {
    0:'dynamic_subgraph',
    1:'static_subgraph',
    2:'dynamic_weight',
    3:'static_weight',
    4:'top_1',
    5:'mse',
    }

losses_names = {
    0:'Daynamic subgraphing',
    1:'Static subgraphing',
    2:'Dynamic weighting',
    3:'Static weighting',
    4:'Maximal loss',
    5:'L2 loss',
    }

n_rows = 1 + 4 * len(losses)
n_cols = 4
height_per_row = 3 
width = 12 
fontsize = 25
pad = 20
set_title_params = {'fontsize':fontsize, 'pad':pad, 'fontweight':'bold'}
text_params = {'fontsize':fontsize, 'fontweight':'bold'}


plt.figure(figsize=(2*width,  2*height_per_row * n_rows))

ax = []

for i, loss in losses.items():
    
    img = get_img(column_1, loss)
    
    ax.append(plt.subplot(1+4*len(losses), 4, 1+4*i))
    ax[-1].imshow(img)
    ax[-1].axis('off')
    if i == 0: ax[-1].set_title('With augmentation\nand 2 message passings', **set_title_params)
    
    img = get_img(column_2, loss)
    
    ax.append(plt.subplot(1+4*len(losses), 4, 2+4*i))
    ax[-1].imshow(img)
    ax[-1].axis('off')
    if i == 0: ax[-1].set_title('Without augmentation\nand 2 message passings', **set_title_params)
    
    img = get_img(column_3, loss)
    
    ax.append(plt.subplot(1+4*len(losses), 4, 3+4*i))
    ax[-1].imshow(img)
    ax[-1].axis('off')
    if i == 0: ax[-1].set_title('With augmentation\nand 5 message passings', **set_title_params) 
    
    img = get_img(column_4, loss)
    
    ax.append(plt.subplot(1+4*len(losses), 4, 4+4*i))
    ax[-1].imshow(img)
    ax[-1].axis('off')
    if i == 0: ax[-1].set_title('Without augmentation\nand 5 message passings', **set_title_params)  
    plt.text(-3400, 400 + height_per_row * 2 * i + height_per_row / 2, losses_names[i], 
        rotation='vertical', verticalalignment='center', horizontalalignment='center', **text_params)


plt.tight_layout()
plt.savefig('3d_inductive.pdf', format='pdf', dpi=600)