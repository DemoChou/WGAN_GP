import numpy as np
import tqdm
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
n_epochs=1000

def generate_and_save_images(model, epoch,save=True, is_flatten=False):
    '''
        Generate images and plot it.
    '''
    for e in range(epoch):
        noise = tf.random.normal(shape=[1, d])
        predictions = model.predict(noise)
        if is_flatten:
            predictions = predictions.reshape(-1, 256, 256, 3).astype('float32')
        
        if not os.path.exists('test'):
            os.makedirs('test')
        
        for i in range(predictions.shape[0]):
            generated_image = (predictions[i] * 0.5 + 0.5).squeeze() * 255.0
            generated_image = generated_image.astype(np.uint8)
            generated_image=cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join('test', f'epoch_{e}_image_{i}.jpg'), generated_image)

        if save:
            print(f"Images saved at {'test'}")
        else:
            print("Image generation complete")
tf.random.set_seed(87)
np.random.seed(87)
d=100
generator = tf.keras.models.load_model(f'outputs\model_epoch5000\model_final.h5')
generate_and_save_images(generator,100)