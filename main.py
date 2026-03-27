import cv2
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets
from google.colab import files
from PIL import Image
import io

print("🩺 أداة السونار عالي الدقة (High-Resolution Detail Enhancer)")
print("الرجاء رفع صورة سونار...")
uploaded = files.upload()

if uploaded:
    file_name = list(uploaded.keys())[0]
    image_bytes = uploaded[file_name]
    img = Image.open(io.BytesIO(image_bytes))
    img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    
    def sharpen_ultrasound(details_level, sharpness, theme):
        # 1. إبراز التفاصيل الدقيقة (CLAHE)
        # كلما زاد الرقم، برزت الأنسجة المخفية أكثر بدون تشويه
        clahe = cv2.createCLAHE(clipLimit=details_level, tileGridSize=(8,8))
        detailed_img = clahe.apply(img_gray)
        
        # 2. زيادة حدة الحواف (Unsharp Masking)
        # نبرز العظام وحدود الأعضاء بقوة
        if sharpness > 0:
            blurred = cv2.GaussianBlur(detailed_img, (0, 0), 3)
            sharpened = cv2.addWeighted(detailed_img, 1.0 + sharpness, blurred, -sharpness, 0)
        else:
            sharpened = detailed_img
            
        # 3. التلوين الحراري
        if theme == 'Heatmap (Jet)':
            cmap_type = cv2.COLORMAP_JET
        elif theme == 'Bone (X-Ray style)':
            cmap_type = cv2.COLORMAP_BONE
        else:
            cmap_type = cv2.COLORMAP_MAGMA
            
        colored = cv2.applyColorMap(sharpened, cmap_type)
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

        # 4. عرض النتائج
        plt.figure(figsize=(18, 9))
        
        plt.subplot(1, 3, 1)
        plt.imshow(img_gray, cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(sharpened, cmap='gray')
        plt.title(f'Ultra-Sharp Details\n(CLAHE: {details_level}, Sharpness: {sharpness})')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(colored_rgb)
        plt.title(f'Colorized Mapped\n(Theme: {theme})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    # واجهة التحكم للحدة والتفاصيل
    interact(sharpen_ultrasound, 
             details_level=widgets.FloatSlider(min=1.0, max=5.0, step=0.5, value=2.0, description='عمق التفاصيل:'),
             sharpness=widgets.FloatSlider(min=0.0, max=3.0, step=0.5, value=1.0, description='حدة الحواف:'),
             theme=widgets.Dropdown(options=['Heatmap (Jet)', 'Bone (X-Ray style)', 'Magma'], value='Heatmap (Jet)', description='الثيم:'))
