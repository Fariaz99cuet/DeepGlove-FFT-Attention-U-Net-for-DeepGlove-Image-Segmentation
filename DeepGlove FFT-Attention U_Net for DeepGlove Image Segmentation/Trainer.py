def pixel_accuracy(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)
    matches = tf.equal(y_true, y_pred)
    accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))
    return accuracy
def build_mo
################################################################################
# Metrics
################################################################################
class IOUScore(Metric):
    def __init__(self, class_weights=None, threshold=None, smooth=1e-5, name=None):
        name = name or "iou_score"
        super().__init__(name=name)
        self.class_weights = class_weights if class_weights is not None else 1.
        self.threshold = threshold
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        return iou_score(
            y_true,
            y_pred,
            class_weights=self.class_weights,
            smooth=self.smooth,
            threshold=self.threshold
        )

class FScore(Metric):
    def __init__(self, beta=1, class_weights=None, threshold=None, smooth=1e-5, name=None):
        name = name or "f{}-score".format(beta)
        super().__init__(name=name)
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1.
        self.threshold = threshold
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        return dice_coefficient(
            y_true,
            y_pred,
            beta=self.beta,
            class_weights=self.class_weights,
            smooth=self.smooth,
            threshold=self.threshold
        )

class Precision(Metric):
    def __init__(self, class_weights=None, threshold=None, smooth=1e-5, name=None):
        name = name or "precision"
        super().__init__(name=name)
        self.class_weights = class_weights if class_weights is not None else 1.
        self.threshold = threshold
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        return precision(
            y_true,
            y_pred,
            class_weights=self.class_weights,
            smooth=self.smooth,
            threshold=self.threshold
        )

class Recall(Metric):
    def __init__(self, class_weights=None, threshold=None, smooth=1e-5, name=None):
        name = name or "recall"
        super().__init__(name=name)
        self.class_weights = class_weights if class_weights is not None else 1.
        self.threshold = threshold
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        return iou_score(
            y_true,
            y_pred,
            class_weights=self.class_weights,
            smooth=self.smooth,
            threshold=self.threshold
        )
        def trainModel(train_dataset, val_dataset, img_size, epochs):
    """
    Обучает модель U-Net для задачи семантической сегментации на заданных наборах данных.

    Args:
        train_dataset (tf.data.Dataset): Тренировочный набор данных, подготовленный для обучения модели.
        val_dataset (tf.data.Dataset): Валидационный набор данных для оценки модели во время обучения.
        img_size (int): Размер изображения (высота и ширина), используемый в модели.
        batch_size (int): Размер батча, используемый в процессе обучения.
        epochs (int): Количество эпох для обучения модели.

    Returns:
        tf.keras.callbacks.History: История обучения модели, содержащая информацию о значениях потерь и метрик на каждой эпохе.
    """
    model = build_model((img_size, img_size, 3))
    model.summary()
    # model = build_tasm_model() # Загрузка готовой модели из библиотеки tasm
    checkpoint_callback = ModelCheckpoint('seg_model.weights.h5', 
                                          monitor='val_loss',
                                          save_best_only=True,
                                          save_weights_only=True,
                                          verbose=1)
    history = model.fit(train_dataset,
                        epochs=epochs,
                        callbacks = [checkpoint_callback],
                        validation_data=val_dataset)
    return historyclass SemanticSegmentationDataset:
    """
    Класс для создания датасета для задачи семантической сегментации.

    Args:
        image_paths (list): Список путей к изображениям.
        mask_paths (list): Список путей к маскам.
        img_size (int): Желаемый размер изображений.
        unique_classes (list): Список уникальных классов вашей задачи.
        num_classes (int): Количество классов для сегментации.
        batch_size (int, optional): Размер батча. По умолчанию 32.
        shuffle (bool, optional): Нужно ли перемешивать данные. По умолчанию True.
        transforms (list, optional): Применение функции преобразований над изображениями (аугментация). По умолчанию None.

    Methods:
        __call__(): Создает и возвращает tf.data.Dataset для обучения или валидации/тестирования.
        open_paths(image_path, mask_path): Открывает изображение и маску по заданным путям.
        apply_transform(image, mask): Применяет преобразования (аугментацию) для текущего набора данных.
        process_mask(image, mask): Обрабатывает маску, преобразуя её в бинарный формат (для бинарной сегментации) или one-hot кодирование (для многоклассовой сегментации).
    """
    def __init__(self,
                 image_paths,
                 mask_paths,
                 img_size,
                 unique_classes,
                 num_classes,
                 batch_size=32,
                 shuffle=True,
                 transforms=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.unique_classes = unique_classes
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transforms = transforms
      
    def __call__(self):
        # 1. Создаем список пар (image, mask)
        dataset = tf.data.Dataset.from_tensor_slices((self.image_paths, self.mask_paths))
        
        # 2. Перемешивание (если требуется)
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.image_paths))
        
        # 3. Открытие и получение изображения с соответствующей маской
        dataset = dataset.map(lambda x, y: self.open_paths(x, y), num_parallel_calls=tf.data.AUTOTUNE)
        
        # 4. Аугментация (если указана)
        if self.transforms:
            dataset = dataset.map(lambda x, y: self.apply_transform(x, y), num_parallel_calls=tf.data.AUTOTUNE)
        
        # 5. Обработка маски (преобразование в бинарный / one-hot вид)
        dataset = dataset.map(lambda x, y: self.process_mask(x, y), num_parallel_calls=tf.data.AUTOTUNE)
        
        # 6. Формирование батчей
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.batch_size)
        
        return dataset

    def open_paths(self, image_path, mask_path):
        image = tf.io.read_file(image_path) # открытие image
        image = tf.image.decode_jpeg(image, channels=3)
        mask = tf.io.read_file(mask_path) # открытие mask
        mask = tf.image.decode_png(mask, channels=1)
        return image, mask
    
    def apply_transform(self, image, mask):
        def train_tranform(image, mask):
            image = np.array(image)
            data = {'image': image, 'mask': mask} # для albumentations необходимо подавать данные в формате словаря с указанием image и mask
            augmented = self.transforms(**data) # возвращает две картинки в augmented
            image, mask = augmented['image'], augmented['mask']
            return tf.convert_to_tensor(image, tf.float32), tf.convert_to_tensor(mask, tf.float32)
        
        image, mask = tf.numpy_function(func=train_tranform, inp=[image, mask], Tout=[tf.float32, tf.float32])
        image.set_shape([self.img_size, self.img_size, 3])
        mask.set_shape([self.img_size, self.img_size, self.num_classes])
        return image, mask
    
    def process_mask(self, image, mask):
        if num_classes == 1:
            # Преобразование маски в бинарную (0 или 1)
            mask = tf.where(mask > 0, 1, 0)
        else:
            # One-Hot кодирование
            mask = tf.equal(mask, tf.constant(list(self.unique_classes), dtype=tf.float32))
        mask = tf.cast(mask, tf.float32)
        return image, mask
 