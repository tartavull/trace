import trace.common as com

from .cremi_dataset import CREMIDataset
from .snemi_dataset import SNEMI3DDataset
from .isbi_dataset import ISBIDataset

DATASET_DICT = {
    com.CREMI_A: CREMIDataset,
    com.CREMI_B: CREMIDataset,
    com.CREMI_C: CREMIDataset,
    com.ISBI: ISBIDataset,
    com.SNEMI3D: SNEMI3DDataset,
}
