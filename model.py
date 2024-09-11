import tensorflow as tf
from pathlib import Path
import cv2
import numpy as np

class CNN_Model(object):
    def __init__(self, weight_path=None):
        # Khởi tạo mô hình CNN với đường dẫn trọng số được cung cấp (nếu có)
        self.weight_path = weight_path
        self.model = None

    def build_model(self, rt=False):
        # Xây dựng mô hình CNN
        self.model = tf.keras.Sequential() 
        # Lớp Convolutional 1
        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
        # Lớp Convolutional 2
        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
        # Lớp MaxPooling
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # Lớp Dropout để tránh overfitting
        self.model.add(tf.keras.layers.Dropout(0.25))

        # Lớp Convolutional 3
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        # Lớp Convolutional 4
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        # Lớp MaxPooling
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # Lớp Dropout để tránh overfitting
        self.model.add(tf.keras.layers.Dropout(0.25))

        # Lớp Convolutional 5
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        # Lớp Convolutional 6
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        # Lớp MaxPooling
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # Lớp Dropout để tránh overfitting
        self.model.add(tf.keras.layers.Dropout(0.25))

        # Lớp Flatten
        self.model.add(tf.keras.layers.Flatten())
        # Lớp Dense kết nối đầy đủ
        self.model.add(tf.keras.layers.Dense(512, activation='relu'))
        # Lớp Dropout để tránh overfitting
        self.model.add(tf.keras.layers.Dropout(0.5))
        # Lớp Dense kết nối đầy đủ
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        # Lớp Dropout để tránh overfitting
        self.model.add(tf.keras.layers.Dropout(0.5))
        # Lớp Dense với softmax để dự đoán xác suất của từng lớp
        self.model.add(tf.keras.layers.Dense(2, activation='softmax'))

        # Nếu có trọng số được cung cấp, tải chúng
        if self.weight_path is not None:
            self.model.load_weights(self.weight_path)
        # model.summary()
        if rt:
            return self.model

    @staticmethod
    def load_data():
        # Tải và tiền xử lý dữ liệu hình ảnh
        dataset_dir = './datasets/'
        images = []
        labels = []

        # Load hình ảnh và nhãn của nhóm "unchoice"
        for img_path in Path(dataset_dir + 'unchoice/').glob("*.png"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
            img = img.reshape((28, 28, 1))
            label = tf.keras.utils.to_categorical(0, num_classes=2)  # Nhãn 0 cho nhóm "unchoice"
            images.append(img / 255.0)
            labels.append(label)

        # Load hình ảnh và nhãn của nhóm "choice"
        for img_path in Path(dataset_dir + 'choice/').glob("*.png"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
            img = img.reshape((28, 28, 1))
            label = tf.keras.utils.to_categorical(1, num_classes=2)  # Nhãn 1 cho nhóm "choice"
            images.append(img / 255.0)
            labels.append(label)

        # Xáo trộn dữ liệu
        datasets = list(zip(images, labels))
        np.random.shuffle(datasets)
        images, labels = zip(*datasets)
        images = np.array(images)
        labels = np.array(labels)

        return images, labels

    def train(self):
        images, labels = self.load_data()

        # Xây dựng mô hình
        self.build_model(rt=False)

        # Biên dịch mô hình
        self.model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(1e-3), metrics=['acc'])

        # Giảm learning rate khi không cải thiện
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, verbose=1, )

        # Lưu model tốt nhất
        cpt_save = tf.keras.callbacks.ModelCheckpoint('./weight.h5', save_best_only=True, monitor='val_acc', mode='max')

        print("Training......")
        # Huấn luyện mô hình
        self.model.fit(images, labels, callbacks=[cpt_save, reduce_lr], verbose=1, epochs=10, validation_split=0.15, batch_size=32,
                       shuffle=True)
# Chú thích mã bằng tiếng Việt với các chi tiết cụ thể
