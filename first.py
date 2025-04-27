import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.applications import VGG19

tf.keras.mixed_precision.set_global_policy('float32')

def load_and_process_image(image_path, img_size=512):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = np.expand_dims(img, axis=0).astype('float32')  # Ensure float32 dtype
    return tf.keras.applications.vgg19.preprocess_input(img)

def deprocess_image(img):
    img = img.reshape((img.shape[1], img.shape[2], 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = np.clip(img, 0, 255).astype('uint8')
    return img

def get_vgg19_model():
    model = VGG19(weights='imagenet', include_top=False)
    model.trainable = False
    return model

def compute_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

def gram_matrix(tensor):
    tensor = tf.cast(tensor, tf.float32)  
    channels = int(tensor.shape[-1])
    a = tf.reshape(tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True) / tf.cast(n, tf.float32)
    return gram

def compute_style_loss(base_style, target_style):
    return tf.reduce_mean(tf.square(gram_matrix(base_style) - gram_matrix(target_style)))

def total_variation_loss(img):
    x_deltas = img[:, :-1, :-1, :] - img[:, 1:, :-1, :]
    y_deltas = img[:, :-1, :-1, :] - img[:, :-1, 1:, :]
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

def get_features(image, model):
    layers = ['block4_conv2']  # Content layer
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    outputs = [model.get_layer(name).output for name in layers + style_layers]
    feature_model = tf.keras.Model(inputs=model.input, outputs=outputs)
    features = feature_model(image)
    return features[:1], features[1:]

def style_transfer(content_path, style_path, iterations=500, alpha=1e4, beta=1e-2):  
    print("🔄 Loading images...")  
    content_image = load_and_process_image(content_path)
    style_image = load_and_process_image(style_path)

    print("🔄 Loading VGG19 model...")
    vgg = get_vgg19_model()
    content_features, style_features = get_features(content_image, vgg)
    _, target_style_features = get_features(style_image, vgg)

    generated_image = tf.Variable(content_image, dtype=tf.float32)

    optimizer = tf.optimizers.Adam(learning_rate=5.0)

    print("🚀 Starting style transfer...")
    for i in range(iterations):
        with tf.GradientTape() as tape:
            gen_content_features, gen_style_features = get_features(generated_image, vgg)
            content_loss = compute_content_loss(content_features[0], gen_content_features[0])
            style_loss = sum(compute_style_loss(gs, ts) for gs, ts in zip(gen_style_features, target_style_features))
            tv_loss = total_variation_loss(generated_image)
            total_loss = alpha * content_loss + beta * style_loss + 30 * tv_loss

        grads = tape.gradient(total_loss, generated_image)
        optimizer.apply_gradients([(grads, generated_image)])
        generated_image.assign(tf.clip_by_value(generated_image, -103.939, 255 - 103.939))

        if i % 50 == 0:
            print(f"Iteration {i}: Loss {total_loss.numpy()}")

    print("Style transfer complete!")
    return deprocess_image(generated_image.numpy())

content_image_path = "D:\\INTERNSHIP CODTECH\\TASK 3\\content_image.jpg"
style_image_path = "D:\\INTERNSHIP CODTECH\\TASK 3\\style_image.jpg"

result_image = style_transfer(content_image_path, style_image_path)
plt.imshow(result_image)
plt.axis('off')
plt.show()
