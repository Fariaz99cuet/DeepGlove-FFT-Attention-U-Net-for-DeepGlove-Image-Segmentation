import tensorflow.keras.backend as K

################################################################################
# Helper Functions
################################################################################
def average(x, class_weights=None):
    if class_weights is not None:
        x = x * class_weights
    return K.mean(x)

def gather_channels(*xs):
    return xs

def round_if_needed(x, threshold):
    if threshold is not None:
        x = K.greater(x, threshold)
        x = K.cast(x, K.floatx())
    return x

################################################################################
# Metric Functions
################################################################################
def iou_score(y_true, y_pred, class_weights=1., smooth=1e-5, threshold=None):    
    # y_true = K.one_hot(K.squeeze(K.cast(y_true, tf.int32), axis=-1), n_classes)

    y_true, y_pred = gather_channels(y_true, y_pred)
    y_pred = round_if_needed(y_pred, threshold)
    axes = [1, 2] if K.image_data_format() == "channels_last" else [2, 3]
    
    intersection = K.sum(y_true * y_pred, axis=axes)
    union = K.sum(y_true + y_pred, axis=axes) - intersection

    score = (intersection + smooth) / (union + smooth)
    score = average(score, class_weights)

    return score

def dice_coefficient(y_true, y_pred, beta=1.0, class_weights=1., smooth=1e-5, threshold=None):
    # print(y_pred)
    y_true, y_pred = gather_channels(y_true, y_pred)
    y_pred = round_if_needed(y_pred, threshold)
    axes = [1, 2] if K.image_data_format() == "channels_last" else [2, 3]
    
    tp = K.sum(y_true * y_pred, axis=axes)
    fp = K.sum(y_pred, axis=axes) - tp
    fn = K.sum(y_true, axis=axes) - tp

    score = ((1.0 + beta) * tp + smooth) / ((1.0 + beta) * tp + (beta ** 2.0) * fn + fp + smooth)
    # print("Score, wo avg: " + str(score))
    score = average(score, class_weights)
    # print("Score: " + str(score))

    return score

def precision(y_true, y_pred, class_weights=1., smooth=1e-5, threshold=None):
    y_true, y_pred = gather_channels(y_true, y_pred)
    y_pred = round_if_needed(y_pred, threshold)
    axes = [1, 2] if K.image_data_format() == "channels_last" else [2, 3]
    
    tp = K.sum(y_true * y_pred, axis=axes)
    fp = K.sum(y_pred, axis=axes) - tp

    score = (tp + smooth) / (tp + fp + smooth)
    score = average(score, class_weights)

    return score

def recall(y_true, y_pred, class_weights=1., smooth=1e-5, threshold=None):
    y_true, y_pred = gather_channels(y_true, y_pred)
    y_pred = round_if_needed(y_pred, threshold)
    axes = [1, 2] if K.image_data_format() == "channels_last" else [2, 3]
    
    tp = K.sum(y_true * y_pred, axis=axes)
    fn = K.sum(y_true, axis=axes) - tp

    score = (tp + smooth) / (tp + fn + smooth)
    score = average(score, class_weights)

    return score

def tversky(y_true, y_pred, alpha=0.7, class_weights=1., smooth=1e-5, threshold=None):
    y_true, y_pred = gather_channels(y_true, y_pred)
    y_pred = round_if_needed(y_pred, threshold)
    axes = [1, 2] if K.image_data_format() == "channels_last" else [2, 3]
    
    tp = K.sum(y_true * y_pred, axis=axes)
    fp = K.sum(y_pred, axis=axes) - tp
    fn = K.sum(y_true, axis=axes) - tp

    score = (tp + smooth) / (tp + alpha * fn + (1 - alpha) * fp + smooth)
    score = average(score, class_weights)

    return score


################################################################################
# Loss Functions
################################################################################
def categorical_crossentropy(y_true, y_pred, class_weights=1.):
    y_true, y_pred = gather_channels(y_true, y_pred)

    axis = 3 if K.image_data_format() == "channels_last" else 1
    y_pred /= K.sum(y_pred, axis=axis, keepdims=True)

    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

    loss = y_true * K.log(y_pred) * class_weights
    return - K.mean(loss)

def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred))

def categorical_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_true, y_pred = gather_channels(y_true, y_pred)
    y_true = K.cast(y_true, K.floatx())
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

    loss = - y_true * (alpha * K.pow((1 - y_pred), gamma) * K.log(y_pred))

    return K.mean(loss)

# def categorical_focal_dice_loss(y_true, y_pred, gamma=2.0, alpha=0.25, beta=1.0, class_weights=1., smooth=1e-5, threshold=None):
#     dice_score = dice_coefficient(y_true, y_pred, beta=beta, class_weights=class_weights, smooth=smooth, threshold=threshold)

#     cat_focal_loss = categorical_focal_loss(y_true, y_pred, gamma=gamma, alpha=alpha)
#     return dice_loss + cat_focal_loss

def binary_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

    loss_a = - y_true * (alpha * K.pow((1 - y_pred), gamma) * K.log(y_pred))
    loss_b = - (1 - y_true) * ((1 - alpha) * K.pow((y_pred), gamma) * K.log(1 - y_pred))
    
    return K.mean(loss_a + loss_b)

def combo(y_true, y_pred, alpha=0.5, beta=1.0, ce_ratio=0.5, class_weights=1., smooth=1e-5, threshold=None):
    # alpha < 0.5 penalizes FP more, alpha > 0.5 penalizes FN more

    y_true, y_pred = gather_channels(y_true, y_pred)
    y_pred = round_if_needed(y_pred, threshold)
    axes = [1, 2] if K.image_data_format() == "channels_last" else [2, 3]
    
    tp = K.sum(y_true * y_pred, axis=axes)
    fp = K.sum(y_pred, axis=axes) - tp
    fn = K.sum(y_true, axis=axes) - tp

    dice = ((1.0 + beta) * tp + smooth) / ((1.0 + beta) * tp + (beta ** 2.0) * fn + fp + smooth)

    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

    ce = - (alpha * (y_true * K.log(y_pred))) + ((1 - alpha) * (1.0 - y_true) * K.log(1.0 - y_pred))
    ce = K.mean(ce, axis=axes)

    combo = (ce_ratio * ce) - ((1 - ce_ratio) * dice)
    loss = average(combo, class_weights)

    return loss
class KerasObject:
    def __init__(self, name=None):
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            return self.__class__.__name__
        return self._name

    @property
    def name(self):
        return self.__name__

    @name.setter
    def name(self, name):
        self._name = name

class Metric(KerasObject):
    pass

class Loss(KerasObject):

    def __add__(self, other):
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError('Loss should be inherited from `Loss` class')

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            return MultipliedLoss(self, value)
        else:
            raise ValueError('Loss should be inherited from `BaseLoss` class')

    def __rmul__(self, other):
        return self.__mul__(other)
    #from .base import Loss


################################################################################
# Losses
################################################################################
class JaccardLoss(Loss):
    def __init__(self, class_weights=None, smooth=1e-5):
        super().__init__(name="jaccard_loss")
        self.class_weights = class_weights if class_weights is not None else 1.0
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        return 1.0 - iou_score(
            y_true,
            y_pred,
            class_weights=self.class_weights,
            smooth=self.smooth,
            threshold=None
        )

class DiceLoss(Loss):
    def __init__(self, beta=1.0, class_weights=None, smooth=1e-5):
        super().__init__(name="dice_loss")
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1.0
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        # print(y_pred)
        return 1.0 - dice_coefficient(
            y_true,
            y_pred,
            beta=self.beta,
            class_weights=self.class_weights,
            smooth=self.smooth,
            threshold=None
        )


class TverskyLoss(Loss):
    def __init__(self, alpha=0.7, class_weights=None, smooth=1e-5):
        super().__init__(name="tversky_loss")
        self.alpha = alpha
        self.class_weights = class_weights if class_weights is not None else 1.0
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        return 1.0 - tversky(
            y_true,
            y_pred,
            alpha=self.alpha,
            class_weights=self.class_weights,
            smooth=self.smooth,
            threshold=None
        )


class FocalTverskyLoss(Loss):
    def __init__(self, alpha=0.7, gamma=1.25, class_weights=None, smooth=1e-5):
        super().__init__(name="focal_tversky_loss")
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights if class_weights is not None else 1.0
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        return K.pow((1.0 - tversky(
            y_true,
            y_pred,
            alpha=self.alpha,
            class_weights=self.class_weights,
            smooth=self.smooth,
            threshold=None
        )),
        self.gamma
        )


class BinaryCELoss(Loss):
    def __init__(self):
        super().__init__(name="binary_crossentropy")

    def __call__(self, y_true, y_pred):
        return binary_crossentropy(
            y_true,
            y_pred
        )


class CategoricalCELoss(Loss):
    def __init__(self, class_weights=None):
        super().__init__(name="categorical_crossentropy")
        self.class_weights = class_weights

    def __call__(self, y_true, y_pred):
        return categorical_crossentropy(
            y_true,
            y_pred,
            class_weights=self.class_weights
        )

class CategoricalFocalLoss(Loss):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__(name="focal_loss")
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, y_true, y_pred):
        return categorical_focal_loss(
            y_true,
            y_pred,
            alpha=self.alpha,
            gamma=self.gamma
        )

class BinaryFocalLoss(Loss):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__(name='binary_focal_loss')
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, y_true, y_pred):
        return binary_focal_loss(y_true, y_pred, alpha=self.alpha, gamma=self.gamma)


class ComboLoss(Loss):
    def __init__(self, alpha=0.5, beta=1.0, ce_ratio=0.5, class_weights=None, smooth=1e-5):
        super().__init__(name="combo_loss")
        self.alpha = alpha
        self.beta = beta
        self.ce_ratio = ce_ratio
        self.class_weights = class_weights if class_weights is not None else 1.0
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        return combo(
            y_true,
            y_pred,
            alpha=self.alpha,
            beta=self.beta,
            ce_ratio=self.ce_ratio,
            class_weights=self.class_weights,
            smooth=self.smooth,
            threshold=None
        )