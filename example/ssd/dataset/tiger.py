import os
import json
import numpy as np
import scipy.sparse
import scipy.io as sio
import random
from imdb import Imdb
import json

WIDTH = 640
HEIGHT = 360

class tiger(Imdb):
    def __init__(self, image_set, shuffle=False) :
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
                x1 = entry[0] / WIDTH
                y1 = entry[1] / HEIGHT 
                x2 = entry[2] / WIDTH
                y2 = entry[3] / HEIGHT

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
            if self.image_set == 'trainval':
                labelfile = 'label.idl'
            elif self.image_set == 'train':
                labelfile = 'train.idl'
            elif self.image_set == 'val':
                labelfile = 'val.idl'

            image_set_file = os.path.join(self._data_path, labelfile)
            assert os.path.exists(image_set_file), \
                    'Path does not exist: {}'.format(image_set_file)

            with open(image_set_file) as f:
                self._gt_roidb = [self._load_roi(x) for x in f.readlines()]
                self._gt_roidb = [x for x in self._gt_roidb if x != None]
                if shuffle:
                    random.shuffle(self._gt_roidb)
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

    def evaluate_detections(self, detections):
        """
        top level evaluations
        Parameters:
        ----------
        detections: list
            result list, each entry is a matrix of detections
        Returns:
        ----------
            None
        """
        # make all these folders for results
        result = {}
        for img, dets in enumerate(detections):
           l = [] 
	   inds = np.where(dets[:, 1] >= 0.1)[0]
	   for i in inds:
	       score = round(float(dets[i, 1]),2)
	       clsid = int(dets[i, 0])
	       l.append([
                   round(float(dets[i, 2] * WIDTH),2) + 1,
                   round(float(dets[i, 3] * HEIGHT),2) + 1,
		   round(float(dets[i, 4] * WIDTH),2) + 1,
                   round(float(dets[i, 5] * HEIGHT),2) + 1,
		       clsid, score])
           result[self.image_set_index[img]+self.image_ext] = l
        print json.dumps(result)
