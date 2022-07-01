import numpy as np
from cryoEM_config import CryoEMConfig

class CryoEMObject:
    def __init__(self, name='', idx=-1, parent_idx=-1, container=[]):
        assert type(container) == list
        self._name = name
        self._idx = idx
        self._parent_idx=parent_idx
        self._container = container

    def __getitem__(self, idx):
        return self._container[idx]

    def __len__(self):
        return len(self._container)

    @property
    def name(self):
        return self._name

    @property
    def idx(self):
        return self._idx

    @property
    def parent_idx(self):
        return self._parent_idx

    @property
    def container(self):
        return self._container

class CTFValue:
    def __init__(self, value=-100, confidence=0.0):
        self._value = value
        self._confidence = confidence

    @property
    def value(self):
        return self._value

    @property
    def confidence(self):
        return self._confidence

class CTFCategory:
    def __init__(self, value=(0.0, 0.0), confidence=0.0):
        self._value = value
        self._confidence = confidence

    @property
    def value(self):
        return self._value

    @property
    def confidence(self):
        return self._confidence

'''
# Data structure for CryoEMHole
class CryoEMHole(CryoEMObject):
    def __init__(self, name='', idx=-1, parent_idx=-1, ctf=CTF(), gt_ctf=CTF()):
        super().__init__(name, idx, parent_idx)
        self._visited = False
        self._ctf = ctf
        self._gt_ctf = gt_ctf

    def __str__(self):
        return '      HOLE {} ({} {}) ctf ({:5.2f} {:5.2f}) gt_ctf ({:5.2f} {:5.2f}) v {:}'.format(self.name, self.parent_idx, \
                                                                        self.idx, self.ctf.value, self.ctf.confidence, self.gt_ctf.value, self.gt_ctf.confidence, self.status)

    @property
    def is_visited(self):
        return self._visited

    @property
    def ctf(self):
        return self._ctf

    @property
    def gt_ctf(self):
        return self._gt_ctf

    @property
    def status(self):
        return self._visited

    def set_status(self, status=False):
        self._visited = status
'''

# Data structure for CryoEMHole
class CryoEMHole(CryoEMObject):
    def __init__(self, name='', idx=-1, parent_idx=-1, ctf=None, gt_ctf=CTFValue(), category_bins=(0, 6,CryoEMConfig.MAX_CTF_VALUE), ctf_category=None):
        super().__init__(name, idx, parent_idx)
        self._visited = False
        self._ctf = ctf
        self._gt_ctf = gt_ctf
        self._category = ctf_category
        self._gt_category = self._create_category(self._gt_ctf, category_bins)

    def __str__(self):
        hole_str ='      HOLE ({} ({} {}) gt_ctf ({:5.2f} {:5.2f}) gt_category ({} {:5.2f}) v {:}'.format(self.name, self.parent_idx, \
                                                                        self.idx, self.gt_ctf.value, self.gt_ctf.confidence, \
                                                                        self.gt_category.value, self.gt_category.confidence, self.status)
        if self._ctf:
            hole_str += '\n        ctf ({:5.2f} {:5.2f})'.format(self.ctf.value, self.ctf.confidence)

        if self._gt_category:
            hole_str += '\n        category ({} {:5.2f})'.format(self.category.value, self.category.confidence)

        return hole_str

    def _create_category(self, ctf, category_bins):
        hist, _ = np.histogram([ctf.value], category_bins)
        return CTFCategory(tuple(hist), ctf.confidence)

    @property
    def is_visited(self):
        return self._visited

    @property
    def ctf(self):
        return self._ctf

    @property
    def gt_ctf(self):
        return self._gt_ctf

    @property
    def status(self):
        return self._visited

    @property
    def category(self):
        return self._category

    @property
    def gt_category(self):
        return self._gt_category

    def set_status(self, status=False):
        self._visited = status

# Data structure for CryoEMPatch (i.e hole-level image)
class CryoEMPatch(CryoEMObject):
    def __init__(self, name='', idx=-1, parent_idx=-1, holeList=[]):
        super().__init__(name, idx, parent_idx, holeList)

    def __str__(self):
        return '    PATCH {} ({} {}) #holes {} \n'.format(self.name, self.parent_idx, self.idx, len(self)) + '    \n'.join([str(item) for item in self.container])

    # default setting returns all ctfs
    def get_ctfs(self, threshold=CryoEMConfig.MAX_CTF_VALUE, prediction=True, visited=True):
        if prediction:
            return [hole.ctf for hole in self.container if hole.ctf.value <= threshold and hole.status == True] if visited else \
                [hole.ctf for hole in self.container if hole.ctf.value <= threshold and hole.status == False]
        else:
            return [hole.gt_ctf for hole in self.container if hole.gt_ctf.value <= threshold and hole.status == True] if visited else \
                [hole.gt_ctf for hole in self.container if hole.gt_ctf.value <= threshold and hole.status == False]

    def get_categories(self, prediction=True, visited=True):
        # print(f"prediciton type: {prediction}")
        if prediction:
            return [hole.category for hole in self.container if hole.status == True] if visited else \
                [hole.category for hole in self.container if hole.status == False]
        else:
            return [hole.gt_category for hole in self.container if hole.status == True] if visited else \
                [hole.gt_category for hole in self.container if hole.status == False]

    '''
    def ctf_histogram(self, bins=[0,3,5,7,9, CryoEMConfig.MAX_CTF_VALUE]): # computing features
        hist, _= np.histogram(self.get_ctfs(), bins)
        return hist
    '''

# Data structure for CryoEMSquare
class CryoEMSquare(CryoEMObject):
    def __init__(self, name='', idx=-1, parent_idx=-1, patchList=[]):
        super().__init__(name, idx, parent_idx, patchList)

    def __str__(self):
        return '  SQUARE {} ({} {}) #patches {}\n'.format(self.name, self.parent_idx, self.idx, len(self)) + '   \n'.join([str(item) for item in self.container])

    def hole_counts(self):
        return sum([len(patch) for patch in self.container])

    def visited_hole_counts(self):
        return sum([patch.visited_hole_counts for patch in self.container])

    '''
    def ctf_histogram(self, bins=[6, CryoEMConfig.MAX_CTF_VALUE]): # computing features
        hist, _= np.histogram(self.get_ctfs(), bins)
        return hist
    '''

# Data structure for CryoEMGrid
class CryoEMGrid(CryoEMObject):
    def __init__(self, name='', idx=-1, squareList=[]):
        super().__init__(name, idx, -1, squareList)

    def __str__(self):
        return 'GRID {} ({} {}) #squares {}\n'.format(self.name, self.parent_idx, self.idx, len(self)) + '  \n'.join([str(item) for item in self.container])
