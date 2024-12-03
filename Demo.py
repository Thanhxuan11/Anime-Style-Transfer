import streamlit as st
from pathlib import Path
import PIL
import cv2
import numpy as np
import tensorflow as tf
from net import generator
from GuidedFilter import guided_filter
import os

# Đặt cấu hình trang
st.set_page_config(
    page_title="Anime Style Transfer",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Ứng dụng chuyển đổi phong cách anime dựa trên AnimeGANv3."
    }
)

# Đặt biến môi trường
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Hàm hỗ trợ
def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def save_images(images, image_path, hw):
    # Normal hóa và chuyển về kiểu uint8
    images = (images.squeeze() + 1.) / 2 * 255
    images = np.clip(images, 0, 255).astype(np.uint8)
    images = cv2.resize(images, (hw[1], hw[0]))
    
    # Đảm bảo lưu ảnh ở định dạng RGB
    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    cv2.imwrite(image_path, images)
def preprocessing(img, x8=True):
    h, w = img.shape[:2]
    if x8:  # Resize image to multiple of 8
        def to_x8s(x):
            return 256 if x < 256 else x - x % 8
        img = cv2.resize(img, (to_x8s(w), to_x8s(h)))
    return img / 127.5 - 1.0

def load_test_data(image, x8=True):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = preprocessing(img, x8)
    img = np.expand_dims(img, axis=0)
    return img, image.shape[:2]

def sigm_out_scale(x):
    x = (x + 1.0) / 2.0
    return tf.clip_by_value(x, 0.0, 1.0)

def tanh_out_scale(x):
    x = (x - 0.5) * 2.0
    return tf.clip_by_value(x, -1.0, 1.0)

def test(checkpoint_dir, save_dir, image):
    result_dir = check_folder(save_dir)
    test_real = tf.placeholder(tf.float32, [1, None, None, 3], name='AnimeGANv3_input')
    with tf.variable_scope("generator", reuse=False):
        _, _ = generator.G_net(test_real, True)
    with tf.variable_scope("generator", reuse=True):
        test_s0, test_m = generator.G_net(test_real, False)
        test_s1 = tanh_out_scale(guided_filter(sigm_out_scale(test_s0), sigm_out_scale(test_s0), 2, 0.01))

    variables = tf.contrib.framework.get_variables_to_restore()
    generator_var = [var for var in variables if var.name.startswith('generator') and
                     ('main' in var.name or 'base' in var.name) and 'Adam' not in var.name and 'support' not in var.name]
    saver = tf.train.Saver(generator_var)

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        sess.run(tf.global_variables_initializer())
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        else:
            print(" [*] Failed to find a checkpoint")
            return

        img, scale = load_test_data(image)
        real, s1, s0, m = sess.run([test_real, test_s1, test_s0, test_m], feed_dict={test_real: img})
        save_images(real, result_dir + '/a_result.png', scale)
        save_images(s1, result_dir + '/b_result.png', scale)
        save_images(s0, result_dir + '/c_result.png', scale)
        save_images(m, result_dir + '/d_result.png', scale)

        return result_dir + '/a_result.png', result_dir + '/b_result.png', result_dir + '/c_result.png', result_dir + '/d_result.png'


# Giao diện Streamlit
st.title("ỨNG DỤNG CHUYỂN ĐỔI ẢNH THẾ GIỚI THỰC THÀNH HÌNH ẢNH PHONG CÁCH ANIME ")

# Sidebar
st.sidebar.header("⚙️ Configuration")
source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png"))
style_option = st.sidebar.selectbox("Select Style", ["Tabe", "Shinkai"])

if source_img:
    # Đọc ảnh từ Streamlit uploader
    uploaded_image = PIL.Image.open(source_img)
    image = np.array(uploaded_image)  # Chuyển PIL Image sang NumPy array (RGB)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Apply Style Transfer"):
        with st.spinner("Processing..."):
            # Chọn checkpoint theo style
            if style_option == "Tabe":
                checkpoint_dir = 'checkpoint/generator_v3_Tabe_weight'
            else:
                checkpoint_dir = 'checkpoint/generator_v3_Shinkai_weight'

            save_dir = 'style_results/'
            
            # Thực hiện chuyển đổi style
            results = test(checkpoint_dir, save_dir, image)

            # Hiển thị ảnh đã stylize
            stylized_image = cv2.imread(results[3]) 
            st.image(stylized_image, caption="Stylized Image (m)", use_column_width=True)