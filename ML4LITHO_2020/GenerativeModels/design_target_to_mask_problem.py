import numpy as np
from skimage.transform import rescale
#from npy2TFRecord import convert_to

#def ingest_images(a_clips, b_clips, name):

#    convert_to(a_clips, b_clips, name )

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape

    arr=arr[: h- h % nrows, :w - w % ncols]
    h, w = arr.shape

    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def load_clip_sets(path, height, width):

    X_data = np.load(path)
    clips=[]
    for data in X_data:
        clips.append(blockshaped(data, height, width))
    return np.array(clips)

def load_clips(path, height, width):
    clip_sets=load_clip_sets(path,height,width)
    clips=[]
    for clip_set in clip_sets: #each clip set is an array of clips
        clip_set = np.reshape(clip_set,(12,12,48,48))
        for clip_row in clip_set[1:-1,1:-1]:
            for clip_element in clip_row:
                clips.append(clip_element)
    return np.array(clips)

def load_contexts(path, height, width):
    clip_sets=load_clip_sets(path,height,width)
    contexts=[]
    for clip_set in clip_sets: #each clip set is an array of clips
        clip_set = np.reshape(clip_set,(12,12,48,48))
        for context_row in range(clip_set.shape[0]-2): #each clip set is an array of clips
            for context_column in range(clip_set.shape[0]-2): #each clip set is an array of clips
                context_region=np.block([[clip_set[context_row,context_column],clip_set[context_row,context_column+1],clip_set[context_row,context_column+2]],
                                         [clip_set[context_row+1,context_column],np.zeros((48,48)),clip_set[context_row+1,context_column+2]],
                                         [clip_set[context_row+2,context_column],clip_set[context_row+2,context_column+1],clip_set[context_row+2,context_column+2]]])
                context_region=rescale(context_region,(1./3,1./3))
                contexts.append(context_region) 
    return np.array(contexts)

def data_shuffle(data_sets_org, percent_of_train, min_test_data=80, shuffle_data=False):
    """Divided the data to train and test and shuffle it"""
    perc = lambda i, t: np.rint((i * t) / 100).astype(np.int32)
    C = type('type_C', (object,), {})
    data_sets = C()
    data_sets.n_samples = data_sets_org.masks.shape[0]
    stop_train_index = perc(percent_of_train, data_sets_org.masks.shape[0])
    start_test_index = stop_train_index
    if percent_of_train > min_test_data:
        start_test_index = perc(min_test_data, data_sets_org.masks.shape[0])
    data_sets.train = C()
    data_sets.test = C()
    if shuffle_data:
        shuffled_masks, shuffled_designs, shuffled_contexts = shuffle_in_unison_inplace(data_sets_org.masks, data_sets_org.designs, data_sets_org.contexts)
    else:
        shuffled_masks, shuffled_designs, shuffled_contexts = data_sets_org.masks, data_sets_org.designs,data_sets_org.contexts
    print('everybodys shuffling : shuf? {} perc {} start_test_index {} data shape {}'.format(
        shuffle_data,percent_of_train,start_test_index,data_sets_org.masks.shape[0]))
    print('shuffled shape {}'.format(shuffled_masks.shape))
    data_sets.train.masks = shuffled_masks[:stop_train_index, :]
    data_sets.train.designs = shuffled_designs[:stop_train_index, :]
    data_sets.train.contexts=shuffled_contexts[:stop_train_index, :]
    data_sets.test.masks = shuffled_masks[start_test_index:, :]
    data_sets.test.designs = shuffled_designs[start_test_index:, :]
    data_sets.test.contexts = shuffled_contexts[start_test_index:, :]
    return data_sets

def shuffle_in_unison_inplace(a, b, c):
	"""Shuffle the arrays randomly"""
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p], c[p]

def mask_design_and_context_data_generation(FLAGS):
    designs = load_clips('{}\\designs.npy'.format(FLAGS.data_dir),FLAGS.image_dim,FLAGS.image_dim )/255.
    masks   = load_clips('{}\\masks.npy'.format(FLAGS.data_dir),FLAGS.image_dim,FLAGS.image_dim  )/255.
    contexts=load_contexts('{}\\designs.npy'.format(FLAGS.data_dir),FLAGS.image_dim,FLAGS.image_dim )/255.
    contexts=contexts/np.amax(contexts+1e-7)
    C = type('type_C', (object,), {})
    data_sets = C()
    if FLAGS.sample_size:
        n_samples=min(FLAGS.sample_size, designs.shape[0], masks.shape[0], contexts.shape[0])
    else:
        n_samples=designs.shape[0]
    data_sets.masks=masks[:n_samples]
    masks=[]
    data_sets.designs=designs[:n_samples]
    designs=[]
    data_sets.contexts=contexts[:n_samples]
    contexts=[]

    data_sets.n_samples=n_samples
    print('loaded masks, designs and contects of shapes {} {} {} and {} elements.'.format( data_sets.masks.shape, data_sets.designs.shape, data_sets.contexts.shape, data_sets.n_samples))
    data_sets = data_shuffle(data_sets, percent_of_train=FLAGS.percent_of_train, shuffle_data=FLAGS.shuffle_data)
    return data_sets