import os
import json
import numpy as np
import scipy.sparse
import scipy.io as sio
import random
from imdb import Imdb

class tiger(Imdb):
    def __init__(self, image_set,shuffle) :
        super(tiger, self).__init__('tiger_' + image_set)
        self.image_set = image_set
        self.classes = ('__background',
            'vehicle', 'pedestrian', 'cyclist',
            'traffic lights')
        self.num_classes = 5
        #self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self.config = {'use_difficult': True,
                       'comp_id': 'comp4',
                       'padding': 30}
        self.image_ext = '.jpg'

        self._root_path = '/home/howardsu/data/Tiger'
        if image_set == 'test' :
            self._data_path = os.path.join(self._root_path, 'testing')
        else :
            self._data_path = os.path.join(self._root_path, 'training')

        assert os.path.exists(self._root_path), \
                'Tiger root path does not exist: {}'.format(self._root_path)
        assert os.path.exists(self._data_path), \
                'Tiger data path does not exist: {}'.format(self._data_path)

        self.image_set_index = self._load_image_set_index(shuffle)
        assert self.image_set == 'test' or self._gt_roidb != None
        self.num_images = len(self.image_set_index)


    def _load_roi(self, x):
        """
        Convert JSON format to ROI data structure
        """
        data = json.loads(x)
        key = data.keys()[0]
        objs = data[key]
        id = os.path.splitext(os.path.basename(key))[0]
        num_objs = len(objs)
        if num_objs == 0:
            return None

        boxes = [] 
        for ix, entry in enumerate(objs):
                if entry[4] == 20:
                    cls = 4
                else:
                    cls = entry[4]
                x1 = entry[0] / 640
                y1 = entry[1] / 360
                x2 = entry[2] / 640
                y2 = entry[3] / 360

                boxes.append([cls, x1, y1, x2, y2])

        return {'index' : id,
                'boxes' : np.array(boxes)}

    def _load_image_set_index(self, shuffle):
        """
        Load the indexes listed in this dataset's image set file.
        """
        if self.image_set == 'test' :
            # enum the current folder
            image_index = [\
                    os.path.splitext(os.path.basename(f))[0] for f in os.listdir(self._data_path) \
                    if os.path.isfile(os.path.join(self._data_path, f)) and os.path.splitext(f)[1] == self.image_ext]
        else :
            image_set_file = os.path.join(self._data_path, 'label.idl')
            assert os.path.exists(image_set_file), \
                    'Path does not exist: {}'.format(image_set_file)

            with open(image_set_file) as f:
                self._gt_roidb = [self._load_roi(x) for x in f.readlines()]
                self._gt_roidb = [x for x in self._gt_roidb if x != None]
                if shuffle:
                    random.shuffle(self._gt_roidb)
            #if self.image_set == 'trainval' :
            image_index = [key["index"] for key in self._gt_roidb]
	    # add padding
	    max_objects = 29 

            assert max_objects > 0, "No objects found for any of the images"
            assert max_objects <= self.config['padding'], "# obj exceed padding"
            self.padding = self.config['padding']
            labels = []
            for x in self._gt_roidb:
		label = x['boxes']
            	label = np.lib.pad(label, ((0, self.padding-label.shape[0]), (0,0)), \
                               'constant', constant_values=(-1, -1))
            	labels.append(label)

       	    self.labels =  np.array(labels)
        return image_index

    def image_path_from_index(self, index):
        """
        load image full path given specified index

        Parameters:
        ----------
        index : int
            index of image requested in dataset

        Returns:
        ----------
        full path of specified image
        """
        key = self.image_set_index[index]
        image_path = os.path.join(self._data_path,
                                  key + self.image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def label_from_index(self, index):
        """
        load ground-truth of image given specified index

        Parameters:
        ----------
        index : int
            index of image requested in dataset

        Returns:
        ----------
        object ground-truths, in format
        numpy.array([id, xmin, ymin, xmax, ymax]...)
        """
	labels = self.labels[index, :, :]
        return labels
if __name__ == '__main__':
    from datasets.tiger import tiger
    d = tiger('trainval')
    res = d.roidb
    from IPython import embed; embed()
