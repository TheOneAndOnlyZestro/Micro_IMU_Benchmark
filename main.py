import tensorflow as tf
import keras
from dataset import load_dataset, load_dataset_conv
from model import  create_conv_model, create_dense_model, create_lstm_model, create_conv_lstm_model, create_residual_conv_lstm_model
import os




if __name__ == "__main__":
    imu_train_file = ["./dataset/IMU_Data_1.csv", "./dataset/IMU_Data_2.csv", "./dataset/IMU_Data_3.csv", "./dataset/IMU_Data_4.csv"]
    imu_test_file = ["./dataset/IMU_Data_5.csv"]

    imu_dataset_train: tf.data.Dataset = load_dataset_conv(imu_file= imu_train_file, sequence_length=10, batch_size=32)
    imu_dataset_test: tf.data.Dataset = load_dataset_conv(imu_file=imu_test_file,sequence_length=10, batch_size=3)
    
    model_conv: keras.Model = create_residual_conv_lstm_model(sequence_length=10)
    model_conv.compile(
        optimizer= keras.optimizers.Adam(learning_rate=1e-3),
        loss= keras.losses.MeanSquaredError(),
        metrics=['accuracy']
    )

    for x, y in imu_dataset_train:
        print(f"{x.shape} {y.shape}")
        break

    model_conv.fit(imu_dataset_train, epochs=5)

    loss, accuracy = model_conv.evaluate(imu_dataset_test)

    model_conv.save('./model_weights/conv_model.keras')
    print("Loss :", loss)
    print("Accuracy :", accuracy)




