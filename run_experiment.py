from ml_core import *

loss_types = [
    'mse',
    'top_1',
    'top_10',
    'static_weight',
    'dynamic_weight',
    'static_subgraph',
    'dynamic_subgraph'
    ]

# loading the experiments
with open('experiment.json', 'r') as file:
    json_dict = json.load(file)
    compression_ratio = json_dict['compression_ratio']
    frame_length = json_dict['frame_length']
    num_message_passings = json_dict['num_message_passings']
    rotation = json_dict['rotation']
    maxK = json_dict['maxK']

with_transform_stats = load_transform_stats('with_transform_stats')

exp_num = int(sys.argv[1])

modelTools = model_tools(
    frame_length=frame_length[exp_num],
    transformation=with_transform_stats,
    rotation = rotation[exp_num], maxK=maxK[exp_num]
    )
modelTools.prepare_ds(num_message_passings=0)

for loss_type in loss_types:
    modelTools.initialize_training(
        compression_ratio=compression_ratio[exp_num],
        exp_num=exp_num,
        loss_type=loss_type,
        model_type='dae'
        )
    modelTools.fit()
    modelTools.store_data()

modelTools.prepare_ds(num_message_passings=num_message_passings[exp_num])
for loss_type in loss_types:
    modelTools.load_data(loss_type, exp_num, 'dae')
    modelTools.initialize_training(
        compression_ratio=compression_ratio[exp_num],
        exp_num=exp_num,
        loss_type=loss_type,
        model_type='combined'
        )
    modelTools.fit()
    modelTools.store_data()
    modelTools.initialize_training(
        compression_ratio=compression_ratio[exp_num],
        exp_num=exp_num,
        loss_type=loss_type,
        model_type='gnn',
        )
    modelTools.fit()
    modelTools.store_data()