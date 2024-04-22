class ContrastiveTransformations:
    """
    To allow efficient training, we need to prepare the data loading such that we sample two different, random augmentations for each image in the batch. The easiest way to do this is by creating a transformation that, when being called, applies a set of data augmentations to an image twice. (source: https://lightning.ai/docs/pytorch/1.8.1/notebooks/course_UvA-DL/13-contrastive-learning.html)
    """
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]
