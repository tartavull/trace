import numpy as np
from scipy import ndimage

class Clefts:

    def __init__(self, test, truth):

        test_clefts = test
        truth_clefts = truth

        self.test_clefts_mask = np.equal(test_clefts, 0xffffffffffffffff)
        self.truth_clefts_mask = np.equal(truth_clefts, 0xffffffffffffffff)
    
        self.test_clefts_edt = ndimage.distance_transform_edt(self.test_clefts_mask, sampling=test_clefts.resolution)
        self.truth_clefts_edt = ndimage.distance_transform_edt(self.truth_clefts_mask, sampling=truth_clefts.resolution)

    # higher threshold -> fewer false positives
    def count_false_positives(self, threshold=200):
        # predicted clefts
        mask1 = np.invert(self.test_clefts_mask)
        # find all background pixels + threshold
        mask2 = self.truth_clefts_edt > threshold 
        # finds all places where a cleft is predicted but 
        # it's also part of the background of the truth
        false_positives = self.truth_clefts_edt[np.logical_and(mask1, mask2)] 
        return false_positives.size

    # higher threshold -> fewer false negatives
    def count_false_negatives(self, threshold=200):
        # all places with true clefts
        mask1 = np.invert(self.truth_clefts_mask)
        # all background pixels of prediction + threshold
        mask2 = self.test_clefts_edt > threshold
        # find all places where there should be a true cleft but 
        # we've predicted that it's part of the background
        false_negatives = self.test_clefts_edt[np.logical_and(mask1, mask2)]
        return false_negatives.size


    def count_true_positives(self, threshold=200):
        # all places with true clefts
        mask1 = np.invert(self.truth_clefts_mask)
        # all foreground pixels within the threshold distance from 
        # an actual foreground pixel
        mask2 = self.test_clefts_edt < threshold
        # finds all places with actual cleft and predictions within a threshold
        true_positives = self.test_clefts_edt[np.logical_and(mask1, mask2)]
        return true_positives.size

    def f_score(self, tp, fn, fp, beta=1.5):
        return ((1 + beta**2) * tp)/((1 + beta**2) * tp + beta**2 * fn + fp)

    def acc_false_positives(self):

        mask = np.invert(self.test_clefts_mask)
        false_positives = self.truth_clefts_edt[mask]
        stats = {
            'mean': np.mean(false_positives),
            'std': np.std(false_positives),
            'max': np.amax(false_positives),
            'count': false_positives.size,
            'median': np.median(false_positives)}
        return stats

    def acc_false_negatives(self):

        mask = np.invert(self.truth_clefts_mask)
        false_negatives = self.test_clefts_edt[mask]
        stats = {
            'mean': np.mean(false_negatives),
            'std': np.std(false_negatives),
            'max': np.amax(false_negatives),
            'count': false_negatives.size,
            'median': np.median(false_negatives)}
        return stats