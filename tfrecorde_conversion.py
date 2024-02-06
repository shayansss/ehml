from ml_core import *

def create_tfrecord(modelName = 'Simple_3D', dataSetName='gnn_datasets'):
    # covert npy files to tfrecord for efficiency

    BytesList = tf.train.BytesList
    Feature = tf.train.Feature
    Features = tf.train.Features
    Example = tf.train.Example

    path_fn = partial(os.path.join, dataSetName, modelName)

    with open(path_fn("meta.json"), "r") as f:
        meta = json.load(f)

    arrayFiles = {k: open(path_fn(k+'.npy'), 'rb')
                  for k in meta['array_names']}

    with tf.io.TFRecordWriter(path_fn('data.tfrecord')) as f:
        for _ in tqdm(range(meta['total_num_samples'])):
            arrays = {k: np.load(f) for k, f in arrayFiles.items()}
            feature = {arrayName: Feature(bytes_list=BytesList(
                       value=[arrays[arrayName].tobytes()]))
                       for arrayName in meta["array_names"]}
            person_example = Example(features=Features(feature=feature))
            f.write(person_example.SerializeToString())

    for k in meta['array_names']:
        arrayFiles[k].close()
        os.remove(path_fn(k+'.npy'))

create_tfrecord() # small-scale dataset
create_tfrecord('knee') # # large-scale dataset
